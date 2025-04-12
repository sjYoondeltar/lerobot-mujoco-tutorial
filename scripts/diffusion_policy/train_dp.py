#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train Diffusion Policy for robot manipulation tasks.
This script trains a Diffusion Policy model on demonstration data.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse  # Added for command-line arguments
from torchvision import transforms
from typing import Dict, Tuple, List, Any

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.datasets.factory import resolve_delta_timestamps


class AddGaussianNoise(object):
    """
    Adds Gaussian noise to a tensor.
    """
    def __init__(self, mean=0., std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # Adds noise: tensor remains a tensor.
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class EpisodeSampler(torch.utils.data.Sampler):
    """
    Sample frames from a specific episode.
    """
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


def create_or_load_policy(ckpt_dir: str, action_type: str = 'joint', load_ckpt: bool = False, root_dir: str = './demo_data') -> Tuple[DiffusionPolicy, LeRobotDatasetMetadata, str]:
    """
    Create a new policy or load an existing one.
    
    Args:
        ckpt_dir: Directory to save/load checkpoints
        action_type: Type of action to train with ('joint', 'eef_pose', or 'delta_q')
        load_ckpt: Whether to load an existing checkpoint
        root_dir: Root directory for dataset
        
    Returns:
        policy: DiffusionPolicy instance
        dataset_metadata: Dataset metadata
        action_type_ckpt_dir: Checkpoint directory for the specific action type
    """
    # 액션 타입에 따라 데이터셋 경로 설정
    dataset_root = os.path.join(root_dir, action_type)
    dataset_metadata = LeRobotDatasetMetadata("omy_pnp", root=dataset_root)
    features = dataset_to_policy_features(dataset_metadata.features)
    
    # 디버깅: 사용 가능한 특성 출력
    print(f"Available features: {list(features.keys())}")
    print("Feature types:")
    for k, v in features.items():
        print(f"  - {k}: type={v.type if hasattr(v, 'type') else 'None'}, shape={v.shape if hasattr(v, 'shape') else 'None'}")
    
    # 액션 관련 feature 찾기
    output_features = {k: v for k, v in features.items() if k == "action" and v.type is FeatureType.ACTION}
    
    # output_features가 비어있는지 확인
    if not output_features:
        print(f"WARNING: No output features found for action_type '{action_type}'")
        print("Dataset might not contain the 'action' key.")
        print(f"Make sure you've collected data with action_type='{action_type}'.")
        raise ValueError(f"No features found for action type: {action_type}")
    
    # 입력 특성 설정
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    # Create policy configuration
    cfg = DiffusionConfig(
        input_features=input_features, 
        output_features=output_features, 
        horizon=16,  # Must be multiple of 8 due to downsampling factor
        n_action_steps=1
    )
    
    # Adjust the checkpoint directory to include the action type
    action_type_ckpt_dir = os.path.join(ckpt_dir, action_type)
    
    # Create or load policy
    if load_ckpt and os.path.exists(action_type_ckpt_dir):
        print(f"Loading policy from {action_type_ckpt_dir}")
        policy = DiffusionPolicy.from_pretrained(action_type_ckpt_dir)
    else:
        print(f"Creating new policy for action type: {action_type}")
        try:
            policy = DiffusionPolicy(cfg, dataset_stats=dataset_metadata.stats)
            print("DiffusionPolicy successfully created")
        except Exception as e:
            print(f"Error creating DiffusionPolicy: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    return policy, dataset_metadata, action_type_ckpt_dir


def prepare_data(dataset_metadata: LeRobotDatasetMetadata, cfg: DiffusionConfig, action_type: str, root_dir: str = './demo_data') -> LeRobotDataset:
    """
    Prepare dataset for training.
    
    Args:
        dataset_metadata: Metadata for the dataset
        cfg: Policy configuration
        action_type: Type of action to train with
        root_dir: Root directory for dataset
        
    Returns:
        dataset: Dataset ready for training
    """
    # Resolve delta timestamps for action chunking
    delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)
    
    # Create data transformation pipeline with enhanced augmentations
    transform = transforms.Compose([
        # Enhanced image augmentations (matching scripts/train.py)
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
        # Add 5 RandomErasing masks (random masking)
        transforms.RandomErasing(p=1.0, scale=(0.01, 0.015), ratio=(0.95, 1.05), value=0),
        transforms.RandomErasing(p=1.0, scale=(0.01, 0.015), ratio=(0.95, 1.05), value=0),
        transforms.RandomErasing(p=1.0, scale=(0.01, 0.015), ratio=(0.95, 1.05), value=0),
        transforms.RandomErasing(p=1.0, scale=(0.01, 0.015), ratio=(0.95, 1.05), value=0),
        transforms.RandomErasing(p=1.0, scale=(0.01, 0.015), ratio=(0.95, 1.05), value=0),
        # Original noise augmentation
        AddGaussianNoise(mean=0., std=0.02),
        transforms.Lambda(lambda x: x.clamp(0, 1))
    ])
    
    # 액션 타입에 따라 데이터셋 경로 설정
    dataset_root = os.path.join(root_dir, action_type)
    
    # Create dataset
    dataset = LeRobotDataset(
        "omy_pnp", 
        delta_timestamps=delta_timestamps, 
        root=dataset_root, 
        image_transforms=transform
    )
    
    # Log dataset information for debugging
    print(f"Dataset loaded with {dataset.num_episodes} episodes from {dataset_root}")
    print(f"Dataset has {len(dataset)} samples")
    
    return dataset


def train_policy(
    policy: DiffusionPolicy, 
    dataset: LeRobotDataset, 
    ckpt_dir: str, 
    device: torch.device,
    training_steps: int = 5000,
    log_freq: int = 100,
    eval_freq: int = 100
) -> List[float]:
    """
    Train the policy on the dataset.
    
    Args:
        policy: Policy to train
        dataset: Dataset to train on
        ckpt_dir: Directory to save checkpoints
        device: Device to train on
        training_steps: Number of training steps
        log_freq: Frequency of logging
        eval_freq: Frequency of evaluation
        
    Returns:
        losses: List of training losses
    """
    # Ensure checkpoint directory exists
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Set up optimizer and dataloader
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=128,  # Increased from 64 to match scripts/train.py
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )
    
    # Training loop
    policy.train()
    policy.to(device)
    losses = []
    mean_errors = []  # Track evaluation metrics
    step = 0
    done = False
    current_epoch = 0
    
    print("Starting training...")
    while not done:
        current_epoch += 1
        for batch in dataloader:
            inp_batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            loss, _ = policy.forward(inp_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            
            if step % log_freq == 0:
                print(f"Step: {step}, Epoch: {current_epoch}, Loss: {loss.item():.3f}")
            
            # 평가 부분 수정
            if step % eval_freq == 0 and step > 0:
                step_ckpt_dir = os.path.join(ckpt_dir, f'step_{step}')
                os.makedirs(step_ckpt_dir, exist_ok=True)
                policy.save_pretrained(step_ckpt_dir)
                
                # 에피소드 구분 없이 단일 MSE 평가
                print(f"\n--- Evaluating at step {step} ---")
                
                # Save current training mode and set to eval
                training = policy.training
                policy.eval()
                
                # 에피소드 구분 없이 한번에 평가
                mse = evaluate_policy_simple(policy, dataset, device)
                if mse is not None:
                    mean_errors.append((step, mse))
                    print(f"MSE: {mse:.5f}")
                
                # Restore training mode
                if training:
                    policy.train()
                
                # 손실 그래프 저장
                plt.figure(figsize=(12, 8))
                plt.subplot(2, 1, 1)
                plt.plot(losses)
                plt.xlabel('Steps')
                plt.ylabel('Loss')
                plt.title(f'Training Loss (Step {step})')
                
                # Plot mean error if available
                if mean_errors:
                    plt.subplot(2, 1, 2)
                    steps, errors = zip(*mean_errors)
                    plt.plot(steps, errors, 'r-')
                    plt.xlabel('Steps')
                    plt.ylabel('MSE')
                    plt.title('Evaluation MSE')
                
                plt.tight_layout()
                plt.savefig(os.path.join(step_ckpt_dir, f'metrics_step_{step}.png'))
                plt.close()
                
                print(f"Saved checkpoint and evaluation at step {step}")
            
            step += 1
            if step >= training_steps:
                done = True
                break
    
    # 최종 모델 저장
    final_ckpt_dir = os.path.join(ckpt_dir, 'final')
    os.makedirs(final_ckpt_dir, exist_ok=True)
    policy.save_pretrained(final_ckpt_dir)
    
    # 최종 평가
    print("\n--- Final Evaluation ---")
    policy.eval()
    mse = evaluate_policy_simple(policy, dataset, device)
    if mse is not None:
        mean_errors.append((step, mse))
        print(f"Final MSE: {mse:.5f}")
    
    # Save combined loss and error graph
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(losses)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss (Final)')
    
    # Plot mean error if available
    if mean_errors:
        plt.subplot(2, 1, 2)
        steps, errors = zip(*mean_errors)
        plt.plot(steps, errors, 'r-')
        plt.xlabel('Steps')
        plt.ylabel('MSE')
        plt.title('Evaluation MSE')
    
    plt.tight_layout()
    plt.savefig(os.path.join(final_ckpt_dir, 'final_metrics.png'))
    plt.close()
    
    print(f"Saving final policy to {final_ckpt_dir}")
    
    return losses


def evaluate_policy_simple(
    policy: DiffusionPolicy, 
    dataset: LeRobotDataset, 
    device: torch.device,
    max_samples: int = 100  # 최대 평가 샘플 수 (모든 데이터를 평가하지 않고 일부만)
) -> float:
    """에피소드 구분 없이 데이터셋의 랜덤 샘플에 대해 정책을 평가합니다."""
    policy.eval()
    
    # 수정: 랜덤 샘플링 방식 변경
    # 텐서 방식에서 리스트 방식으로 변경하여 오류 방지
    dataset_size = len(dataset)
    sample_size = min(max_samples, dataset_size)
    # 리스트 인덱스로 변환하여 사용
    indices = list(range(dataset_size))
    np.random.shuffle(indices)  # numpy로 셔플
    indices = indices[:sample_size]  # 필요한 수만큼만 선택
    
    test_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        # 리스트로 샘플러 인덱스 전달
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
        num_workers=0,  # 디버깅을 위해 worker 수를 0으로 변경
        pin_memory=device.type != "cpu",
    )
    
    policy.reset()
    print(f"Evaluating policy on {sample_size} random samples...")
    
    all_errors = []
    sample_count = 0
    # 디버깅용 변수들
    has_nonzero_error = False
    first_nonzero_idx = -1
    first_gt_vals = None
    first_pred_vals = None
    
    for batch in test_dataloader:
        inp_batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        
        # 디버깅: 입력 batch의 구조와 'action' 값 확인
        if sample_count == 0:
            print("\nInput batch keys:", list(inp_batch.keys()))
            if 'action' in inp_batch:
                action_shape = inp_batch['action'].shape
                print(f"GT action shape: {action_shape}")
                # GT 값의 통계 출력
                action_stats = {
                    'min': inp_batch['action'].min().item(),
                    'max': inp_batch['action'].max().item(),
                    'mean': inp_batch['action'].mean().item(),
                    'std': inp_batch['action'].std().item() if inp_batch['action'].numel() > 1 else 0,
                    'zeros': torch.sum(inp_batch['action'] == 0).item(),
                    'total_elements': inp_batch['action'].numel()
                }
                print(f"GT action stats: {action_stats}")
                if action_stats['zeros'] == action_stats['total_elements']:
                    print("WARNING: GT action is all zeros!")
            else:
                print("WARNING: 'action' key not found in batch!")
                print("Available keys:", list(inp_batch.keys()))
        
        try:
            # 모델 예측 시도
            action = policy.select_action(inp_batch)
            
            # 예측값과 실제값 비교
            if len(action.shape) == 3:  # (batch, time, action_dim)
                pred = action[:, 0, :]  # 첫 번째 타임스텝만 사용
            else:
                pred = action
                
            gt = inp_batch["action"][:, 0, :]  # 첫 번째 타임스텝만 사용
            
            # 디버깅: 첫 샘플의 예측값과 GT 출력
            if sample_count == 0:
                print(f"First prediction shape: {pred.shape}")
                print(f"First GT shape: {gt.shape}")
                
                # 예측값 통계
                pred_stats = {
                    'min': pred.min().item(),
                    'max': pred.max().item(),
                    'mean': pred.mean().item(),
                    'std': pred.std().item() if pred.numel() > 1 else 0,
                    'zeros': torch.sum(pred == 0).item(),
                    'total_elements': pred.numel()
                }
                print(f"Prediction stats: {pred_stats}")
                if pred_stats['zeros'] == pred_stats['total_elements']:
                    print("WARNING: Prediction is all zeros!")
                
                # 첫 10개 값 출력 (지나치게 길지 않도록)
                first_gt_vals = gt[0, :10].cpu().detach().numpy()
                first_pred_vals = pred[0, :10].cpu().detach().numpy()
                print(f"First 10 GT values: {first_gt_vals}")
                print(f"First 10 pred values: {first_pred_vals}")
            
            # 차원 맞추기
            min_dim = min(pred.size(1), gt.size(1))
            squared_diff = torch.square(pred[:, :min_dim] - gt[:, :min_dim])
            batch_error = torch.mean(squared_diff).item()
            
            # 디버깅: 첫 비제로 에러 기록
            if batch_error > 0 and not has_nonzero_error:
                has_nonzero_error = True
                first_nonzero_idx = sample_count
                print(f"First non-zero error at sample {sample_count}: {batch_error:.6f}")
                # 이 샘플의 처음 10개 값 출력
                current_gt = gt[0, :10].cpu().detach().numpy()
                current_pred = pred[0, :10].cpu().detach().numpy()
                print(f"This sample GT: {current_gt}")
                print(f"This sample pred: {current_pred}")
            
            all_errors.append(batch_error)
            
        except Exception as e:
            print(f"Error processing sample {sample_count}: {e}")
            import traceback
            traceback.print_exc()
            continue
            
        sample_count += 1
        # 샘플 10개마다 진행 상황 표시
        if sample_count % 10 == 0:
            print(f"Processed {sample_count}/{sample_size} samples")
    
    # 결과 분석 출력
    print("\n--- Evaluation Results Analysis ---")
    print(f"Total samples processed: {sample_count}")
    print(f"Samples with errors: {len(all_errors)}")
    
    if all_errors:
        # 에러 통계
        error_stats = {
            'min': min(all_errors),
            'max': max(all_errors),
            'mean': sum(all_errors) / len(all_errors),
            'zeros': sum(1 for e in all_errors if e == 0),
            'total': len(all_errors)
        }
        print(f"Error statistics: {error_stats}")
        print(f"Zero errors: {error_stats['zeros']}/{error_stats['total']} ({error_stats['zeros']/error_stats['total']*100:.1f}%)")
        
        # 첫 번째 샘플과 비제로 샘플 요약
        if first_gt_vals is not None and first_pred_vals is not None:
            print("\nFirst sample summary:")
            print(f"GT: {first_gt_vals}")
            print(f"Pred: {first_pred_vals}")
        
        if has_nonzero_error:
            print(f"\nFirst non-zero error was at sample {first_nonzero_idx}")
        else:
            print("\nALL ERRORS WERE ZERO - THIS IS SUSPICIOUS!")
            print("Check if the predictions exactly match ground truth or if there's a problem with the evaluation.")
        
        mse = sum(all_errors) / len(all_errors)
        return mse
    else:
        print("No samples evaluated")
        return None


def main():
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train Diffusion policy with selected action type')
    parser.add_argument('--action_type', type=str, default='joint', 
                        choices=['joint', 'eef_pose', 'delta_q'],
                        help='Action type to use for training: joint, eef_pose, or delta_q')
    parser.add_argument('--load_ckpt', action='store_true',
                        help='Whether to load from checkpoint')
    parser.add_argument('--num_epochs', type=int, default=5000,
                        help='Number of epochs to train')
    args = parser.parse_args()
    
    # Configuration
    REPO_NAME = 'omy_pnp'
    ROOT = "./demo_data_3"  # Path to demonstration data
    CKPT_DIR = "./ckpt/diffusion_y_v3"  # Path to save checkpoints
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    TRAINING_STEPS = args.num_epochs
    LOG_FREQ = 100
    
    # Set action type from command line argument
    ACTION_TYPE = args.action_type
    print(f"\n=== Training Diffusion Policy with action type: {ACTION_TYPE} on device: {DEVICE} ===\n")
    
    try:
        # 정책 생성 또는 로드
        policy, dataset_metadata, action_type_ckpt_dir = create_or_load_policy(
            CKPT_DIR, action_type=ACTION_TYPE, load_ckpt=args.load_ckpt, root_dir=ROOT)
        
        # Prepare dataset
        dataset = prepare_data(dataset_metadata, policy.config, ACTION_TYPE, ROOT)
        print(f"Dataset loaded with {dataset.num_episodes} episodes")
        
        # Train the policy
        losses = train_policy(
            policy=policy, 
            dataset=dataset, 
            ckpt_dir=action_type_ckpt_dir, 
            device=DEVICE, 
            training_steps=TRAINING_STEPS, 
            log_freq=LOG_FREQ,
            eval_freq=100  # Evaluate every 100 steps
        )
        
        # We don't need additional evaluation/plotting since it's done in train_policy now
        # The training_loss.png is also redundant since we now have metrics plots
        
    except Exception as e:
        print(f"Error during training or evaluation: {e}")
        import traceback
        traceback.print_exc()
        
    print("Script completed.")


if __name__ == "__main__":
    main() 