#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import time
import os
from contextlib import nullcontext
from pprint import pformat
from typing import Any
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig





def setup_distributed():
    """Setup distributed training"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.distributed.barrier()
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def update_policy_distributed(
    train_tracker: MetricsTracker,
    policy: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    use_amp: bool = False,
) -> tuple[MetricsTracker, dict[str, Any]]:
    """Update policy with distributed training support"""
    start_time = time.perf_counter()
    
    policy.train()
    
    # Forward pass with optional AMP
    with torch.autocast(device_type="cuda", enabled=use_amp):
        loss, output_dict = policy.forward(batch)
    
    # Backward pass with gradient scaling
    grad_scaler.scale(loss).backward()
    
    # Gradient clipping
    if grad_clip_norm is not None:
        grad_scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        # Calculate grad norm for logging
        total_norm = 0.0
        for p in policy.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        grad_norm = torch.tensor(total_norm ** (1.0 / 2))
    
    # Optimizer step
    grad_scaler.step(optimizer)
    grad_scaler.update()
    
    if lr_scheduler is not None:
        lr_scheduler.step()
    
    optimizer.zero_grad()
    
    # Update metrics
    train_tracker.loss = loss.item()
    train_tracker.grad_norm = grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm
    train_tracker.lr = optimizer.param_groups[0]["lr"]
    train_tracker.update_s = time.perf_counter() - start_time
    
    return train_tracker, output_dict


def create_distributed_dataloader(dataset, cfg, rank, world_size, sampler=None):
    """Create distributed dataloader"""
    if world_size > 1:
        if sampler is not None:
            # For EpisodeAwareSampler, we need to make it distributed
            dist_sampler = DistributedSampler(
                dataset, 
                num_replicas=world_size, 
                rank=rank,
                shuffle=True
            )
        else:
            dist_sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True
            )
        shuffle = False
    else:
        dist_sampler = sampler
        shuffle = sampler is None
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size // world_size,  # Split batch across GPUs
        shuffle=shuffle,
        sampler=dist_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,  # Important for distributed training
        persistent_workers=cfg.num_workers > 0,
    )
    
    return dataloader


@parser.wrap()
def train(cfg: ExtendedTrainPipelineConfig):
    """Multi-GPU training with FSDP"""
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    is_main_process = rank == 0
    
    # Setup device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    
    # Only log from main process
    if is_main_process:
        cfg.validate()
        logging.info(pformat(cfg.to_dict()))
        logging.info(f"Using {world_size} GPUs for training")
    
    # WandB only on main process
    wandb_logger = None
    if is_main_process and cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    elif is_main_process:
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))
    
    if cfg.seed is not None:
        set_seed(cfg.seed + rank)  # Different seed per rank
    
    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    if is_main_process:
        logging.info("Creating dataset")
    dataset = make_dataset(cfg)
    
    # Create environment only on main process for evaluation
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None and is_main_process:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)
    
    if is_main_process:
        logging.info("Creating policy")
    
    # Create policy
    if cfg.policy.type == "pi0":
        cfg.policy.pretrained_path = 'lerobot/pi0'
    elif cfg.policy.type == 'smolvla':
        cfg.policy.pretrained_path = 'lerobot/smolvla_base'
    
    # Ensure same random seed for consistent model initialization across processes
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cfg.seed)
            torch.cuda.manual_seed_all(cfg.seed)
    
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)
    policy = policy.to(device)
    
    # Synchronize model parameters across all processes
    if world_size > 1 and dist.is_initialized():
        with torch.no_grad():
            for param in policy.parameters():
                dist.broadcast(param.data, src=0)
            for buffer in policy.buffers():
                dist.broadcast(buffer.data, src=0)
        
        # Wait for all processes to complete synchronization
        dist.barrier()
    
    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    
    # Setup DDP for multi-GPU training
    if world_size > 1:
        policy = DDP(
            policy,
            device_ids=[local_rank] if device.type == 'cuda' else None,
            output_device=local_rank if device.type == 'cuda' else None,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            broadcast_buffers=True,
        )
        if is_main_process:
            logging.info("Policy wrapped with DDP")
    
    # Disable AMP for BFloat16 compatibility
    use_amp = cfg.policy.use_amp and not any(p.dtype == torch.bfloat16 for p in policy.parameters())
    grad_scaler = GradScaler(device.type, enabled=use_amp)
    
    if is_main_process and not use_amp and cfg.policy.use_amp:
        logging.warning("AMP disabled due to BFloat16 parameters in model")
    
    step = 0
    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)
    
    # Log model info only on main process
    if is_main_process:
        num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        num_total_params = sum(p.numel() for p in policy.parameters())
        
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")
    
    # Create distributed dataloader
    if hasattr(cfg.policy, "drop_n_last_frames"):
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        sampler = None
    
    dataloader = create_distributed_dataloader(dataset, cfg, rank, world_size, sampler)
    dl_iter = cycle(dataloader)
    
    policy.train()
    
    # Training metrics
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    
    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )
    
    if is_main_process:
        logging.info("Start offline training on a fixed dataset")
    
    # Training loop
    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time
        
        # Move batch to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)
        
        # Update policy
        if world_size > 1:
            train_tracker, output_dict = update_policy_distributed(
                train_tracker,
                policy,
                batch,
                optimizer,
                cfg.optimizer.grad_clip_norm,
                grad_scaler=grad_scaler,
                lr_scheduler=lr_scheduler,
                use_amp=use_amp,
            )
        else:
            # Use original update function for single GPU
            from train_model import update_policy
            train_tracker, output_dict = update_policy(
                train_tracker,
                policy,
                batch,
                optimizer,
                cfg.optimizer.grad_clip_norm,
                grad_scaler=grad_scaler,
                lr_scheduler=lr_scheduler,
                use_amp=use_amp,
            )
        
        step += 1
        train_tracker.step()
        
        # Logging and checkpointing only on main process
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0
        
        if is_main_process and is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()
        
        # Synchronize before checkpointing
        if world_size > 1:
            dist.barrier()
        
        if is_main_process and cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            
            # For FSDP, use state_dict with full_state_dict context
            if world_size > 1:
                with FSDP.state_dict_type(policy, FSDP.StateDictType.FULL_STATE_DICT):
                    save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
            else:
                save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
            
            update_last_checkpoint(checkpoint_dir)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)
        
        # Evaluation only on main process
        if is_main_process and cfg.env and is_eval_step:
            step_id = get_step_identifier(step, cfg.steps)
            logging.info(f"Eval policy at step {step}")
            
            # Set policy to eval mode
            policy.eval()
            
            with (
                torch.no_grad(),
                torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
            ):
                from train_model import eval_policy
                eval_info = eval_policy(
                    eval_env,
                    policy,
                    cfg.eval.n_episodes,
                    videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                    max_episodes_rendered=4,
                    start_seed=cfg.seed,
                )
            
            # Back to train mode
            policy.train()
            
            eval_metrics = {
                "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                "pc_success": AverageMeter("success", ":.1f"),
                "eval_s": AverageMeter("eval_s", ":.3f"),
            }
            eval_tracker = MetricsTracker(
                cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step
            )
            eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
            eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
            eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
            logging.info(eval_tracker)
            if wandb_logger:
                wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")
    
    # Cleanup
    if eval_env:
        eval_env.close()
    
    if world_size > 1:
        cleanup_distributed()
    
    if is_main_process:
        logging.info("End of training")


if __name__ == "__main__":
    init_logging()
    train()
