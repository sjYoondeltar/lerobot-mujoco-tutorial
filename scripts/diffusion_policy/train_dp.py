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
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.common.datasets.factory import resolve_delta_timestamps


def dataset_to_policy_features(features: dict[str, dict]) -> dict[str, PolicyFeature]:
    # TODO(aliberts): Implement "type" in dataset features and simplify this
    policy_features = {}
    for key, ft in features.items():
        shape = ft["shape"]
        if ft["dtype"] in ["image", "video"]:
            type = FeatureType.VISUAL
            if len(shape) != 3:
                raise ValueError(f"Number of dimensions of {key} != 3 (shape={shape})")

            names = ft["names"]
            # Backward compatibility for "channel" which is an error introduced in LeRobotDataset v2.0 for ported datasets.
            if names[2] in ["channel", "channels"]:  # (h, w, c) -> (c, h, w)
                shape = (shape[2], shape[0], shape[1])
        elif key == "observation.environment_state":
            type = FeatureType.ENV
        elif key.startswith("observation"):
            type = FeatureType.STATE
        elif key == "action":
            type = FeatureType.ACTION
        elif key.startswith("action"):
            type = FeatureType.ACTION
        else:
            continue

        policy_features[key] = PolicyFeature(
            type=type,
            shape=shape,
        )

    return policy_features


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


def create_or_load_policy(ckpt_dir: str, action_type: str = 'joint', load_ckpt: bool = False) -> Tuple[DiffusionPolicy, LeRobotDatasetMetadata, str]:
    """
    Create a new policy or load an existing one.
    
    Args:
        ckpt_dir: Directory to save/load checkpoints
        action_type: Type of action to train with ('joint', 'ee_pose', or 'delta_q')
        load_ckpt: Whether to load an existing checkpoint
        
    Returns:
        policy: DiffusionPolicy instance
        dataset_metadata: Dataset metadata
        action_type_ckpt_dir: Checkpoint directory for the specific action type
    """
    # Load dataset metadata
    dataset_metadata = LeRobotDatasetMetadata("omy_pnp", root='./demo_data')
    features = dataset_to_policy_features(dataset_metadata.features)
    
    # 디버깅: 사용 가능한 특성 출력
    print(f"Available features: {list(features.keys())}")
    print("Feature types:")
    for k, v in features.items():
        print(f"  - {k}: type={v.type if hasattr(v, 'type') else 'None'}, shape={v.shape if hasattr(v, 'shape') else 'None'}")
    
    # Filter output features based on selected action type
    if action_type == 'joint':
        output_features = {k: v for k, v in features.items() if k == "action.joint" and v.type is FeatureType.ACTION}
    elif action_type == 'ee_pose':
        output_features = {k: v for k, v in features.items() if k == "action.ee_pose" and v.type is FeatureType.ACTION}
    elif action_type == 'delta_q':
        output_features = {k: v for k, v in features.items() if k == "action.delta_q" and v.type is FeatureType.ACTION}
    else:
        raise ValueError(f"Unknown action type: {action_type}")
    
    # 선택된 output_features 확인
    print(f"Selected output features for {action_type}: {list(output_features.keys())}")
    
    # output_features가 비어있는지 확인
    if not output_features:
        print(f"WARNING: No output features found for action_type '{action_type}'")
        print("Dataset might not contain the requested action type.")
        print("Make sure you've collected data with the appropriate action types.")
        raise ValueError(f"No features found for action type: {action_type}")
    
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    # input_features.pop("observation.wrist_image")

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


def prepare_data(dataset_metadata: LeRobotDatasetMetadata, cfg: DiffusionConfig) -> LeRobotDataset:
    """
    Prepare dataset for training.
    
    Args:
        dataset_metadata: Metadata for the dataset
        cfg: Policy configuration
        
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
    
    # Create dataset
    dataset = LeRobotDataset(
        "omy_pnp", 
        delta_timestamps=delta_timestamps, 
        root='./demo_data', 
        image_transforms=transform
    )
    
    return dataset


def train_policy(
    policy: DiffusionPolicy, 
    dataset: LeRobotDataset, 
    ckpt_dir: str, 
    device: torch.device,
    training_steps: int = 5000,  # Increased from 3000 for better convergence
    log_freq: int = 100
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
    step = 0
    done = False
    
    print("Starting training...")
    while not done:
        for batch in dataloader:
            inp_batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            loss, _ = policy.forward(inp_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            
            if step % log_freq == 0:
                print(f"step: {step} loss: {loss.item():.3f}")
            
            step += 1
            if step >= training_steps:
                done = True
                break
    
    # Save the trained model
    print(f"Saving policy to {ckpt_dir}")
    policy.save_pretrained(ckpt_dir)
    
    return losses


def evaluate_policy(
    policy: DiffusionPolicy, 
    dataset: LeRobotDataset, 
    device: torch.device,
    episode_index: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the policy on an episode."""
    policy.eval()
    actions = []
    gt_actions = []
    
    # 에피소드 샘플러 설정
    episode_sampler = EpisodeSampler(dataset, episode_index)
    test_dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=1,
        shuffle=False,
        pin_memory=device.type != "cpu",
        sampler=episode_sampler,
    )
    
    policy.reset()
    print("Evaluating policy...")
    
    for batch in test_dataloader:
        inp_batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        action = policy.select_action(inp_batch)
        
        # 디버깅 정보 출력
        if len(actions) == 0:
            print(f"Prediction shape: {action.shape}")
            print(f"Ground truth shape: {inp_batch['action'].shape}")
        
        # 액션 저장 (첫 번째 타임스텝만)
        if len(action.shape) == 3:  # (batch, time, action_dim)
            actions.append(action[:, 0, :])  # 첫 번째 타임스텝만 사용
        else:
            actions.append(action)
            
        gt_actions.append(inp_batch["action"][:, 0, :])
    
    if actions:
        actions = torch.cat(actions, dim=0)
        gt_actions = torch.cat(gt_actions, dim=0)
        print(f"Final shapes - Pred: {actions.shape}, GT: {gt_actions.shape}")
        
        # 크기가 같은지 확인하고 평균 오차 계산
        min_dim = min(actions.size(1), gt_actions.size(1))
        error = torch.mean(torch.abs(actions[:, :min_dim] - gt_actions[:, :min_dim])).item()
        print(f"Mean action error: {error:.3f}")
        
        return gt_actions, actions
    else:
        print("No actions collected during evaluation")
        return None, None


def plot_results(gt_actions: torch.Tensor, pred_actions: torch.Tensor, save_dir: str):
    """Plot the evaluation results."""
    if gt_actions is None or pred_actions is None:
        print("No actions to plot")
        return
    
    # 디렉토리 생성
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    print(f"Saving plots to: {save_dir}")
    
    # 텐서를 numpy로 변환
    gt_np = gt_actions.cpu().detach().numpy()
    pred_np = pred_actions.cpu().detach().numpy()
    
    # 차원 조정 (3차원인 경우 2차원으로 변환)
    if len(pred_np.shape) == 3:
        print(f"Reshaping predictions from {pred_np.shape} to 2D")
        pred_np = pred_np[:, 0, :]  # 첫 번째 타임스텝만 사용
    
    print(f"Final shapes for plotting - Pred: {pred_np.shape}, GT: {gt_np.shape}")
    
    # 액션 차원 수 확인
    action_dim = gt_np.shape[1]
    fig, axs = plt.subplots(action_dim, 1, figsize=(10, 2*action_dim))
    
    # 각 액션 차원 별로 플롯
    for i in range(action_dim):
        if action_dim == 1:
            ax = axs
        else:
            ax = axs[i]
        
        ax.plot(pred_np[:, i], label="prediction")
        ax.plot(gt_np[:, i], label="ground truth")
        ax.set_title(f"Action Dimension {i}")
        ax.legend()
    
    plt.tight_layout()
    action_plot_path = os.path.join(save_dir, 'action_comparison.png')
    plt.savefig(action_plot_path)
    print(f"Saved action comparison plot to: {action_plot_path}")
    plt.close()
    
    # 에러 히트맵 플롯 (최소 차원까지만 비교)
    min_dim = min(pred_np.shape[1], gt_np.shape[1])
    error = np.abs(pred_np[:, :min_dim] - gt_np[:, :min_dim])
    
    plt.figure(figsize=(10, 6))
    plt.imshow(error.T, aspect='auto', cmap='hot')
    plt.colorbar(label='Absolute Error')
    plt.xlabel('Time Step')
    plt.ylabel('Action Dimension')
    plt.title('Error Heatmap')
    heatmap_path = os.path.join(save_dir, 'error_heatmap.png')
    plt.savefig(heatmap_path)
    print(f"Saved error heatmap to: {heatmap_path}")
    plt.close()
    
    # 에러 히스토그램 플롯
    plt.figure(figsize=(10, 6))
    plt.hist(error.flatten(), bins=50)
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    histogram_path = os.path.join(save_dir, 'error_histogram.png')
    plt.savefig(histogram_path)
    print(f"Saved error histogram to: {histogram_path}")
    plt.close()


def main():
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train Diffusion policy with selected action type')
    parser.add_argument('--action_type', type=str, default='joint', 
                        choices=['joint', 'ee_pose', 'delta_q'],
                        help='Action type to use for training: joint, ee_pose, or delta_q')
    args = parser.parse_args()
    
    # Configuration
    REPO_NAME = 'omy_pnp'
    ROOT = "./demo_data"  # Path to demonstration data
    CKPT_DIR = "./ckpt/diffusion_y"  # Path to save checkpoints
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    TRAINING_STEPS = 2000
    LOG_FREQ = 100
    
    # Set action type from command line argument
    ACTION_TYPE = args.action_type
    print(f"Training Diffusion Policy with action type: {ACTION_TYPE} on device: {DEVICE}")
    
    # 먼저 데이터셋 내용을 검증하고 필요한 데이터가 있는지 확인
    try:
        # 데이터셋 메타데이터 로드
        dataset_metadata = LeRobotDatasetMetadata(REPO_NAME, root=ROOT)
        print(f"Dataset contains the following features:")
        for k, v in dataset_metadata.features.items():
            print(f"  - {k}: {v}")
        
        # 필요한 액션 타입이 있는지 확인
        action_feature_name = f"action.{ACTION_TYPE}"
        if action_feature_name not in dataset_metadata.features:
            print(f"ERROR: The dataset does not contain the required feature '{action_feature_name}'")
            print(f"Available features: {list(dataset_metadata.features.keys())}")
            print("Please collect data with this feature or choose another action type.")
            return
        else:
            print(f"Found required feature '{action_feature_name}' in dataset")
    except Exception as e:
        print(f"Error while validating dataset: {e}")
        print("Please make sure the dataset exists at the correct location.")
        import traceback
        traceback.print_exc()
        return
    
    # Create or load policy
    try:
        policy, dataset_metadata, action_type_ckpt_dir = create_or_load_policy(
            CKPT_DIR, action_type=ACTION_TYPE, load_ckpt=False)
        
        # Prepare dataset
        dataset = prepare_data(dataset_metadata, policy.config)
        print(f"Dataset loaded with {dataset.num_episodes} episodes")
        
        # Train the policy
        losses = train_policy(policy, dataset, action_type_ckpt_dir, DEVICE, TRAINING_STEPS, LOG_FREQ)
        
        # Plot training loss
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title(f'Training Loss ({ACTION_TYPE})')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(action_type_ckpt_dir, 'training_loss.png'))
        
        # Evaluate the policy
        gt_actions, pred_actions = evaluate_policy(policy, dataset, DEVICE)
        
        # Plot evaluation results
        if gt_actions is not None and pred_actions is not None:
            plot_results(gt_actions, pred_actions, action_type_ckpt_dir)
    except Exception as e:
        print(f"Error during training or evaluation: {e}")
        import traceback
        traceback.print_exc()
        
    print("Script completed.")


if __name__ == "__main__":
    main() 