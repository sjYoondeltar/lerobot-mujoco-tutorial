#!/usr/bin/env python
# coding: utf-8

"""
Train Action-Chunking-Transformer (ACT)

This script trains an ACT model on the collected robot demonstration dataset.
It takes approximately 30-60 minutes to train the model.
The trained checkpoint will be saved in the './ckpt/act_y' folder.
"""

import os
import sys
import argparse

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import time
import torch
import matplotlib.pyplot as plt
from torchvision import transforms # Import transforms
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.common.policies.act.configuration_act import ACTConfig
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


def create_or_load_policy(ckpt_dir, action_type='joint', load_ckpt=False):
    """
    Create a new policy or load from checkpoint
    
    Args:
        ckpt_dir: Directory to save or load the checkpoint from
        action_type: Type of action to train with ('joint', 'ee_pose', or 'delta_q')
        load_ckpt: Whether to load from checkpoint
    """
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

    # 새로운 API 방식으로 설정
    cfg = ACTConfig(
        input_features=input_features, 
        output_features=output_features, 
        chunk_size=10, 
        n_action_steps=10
    )
    
    # Adjust the checkpoint directory to include the action type
    action_type_ckpt_dir = os.path.join(ckpt_dir, action_type)
    
    if load_ckpt and os.path.exists(action_type_ckpt_dir):
        print(f"Loading policy from {action_type_ckpt_dir}")
        policy = ACTPolicy.from_pretrained(action_type_ckpt_dir)
    else:
        print(f"Creating new policy for action type: {action_type}")
        try:
            policy = ACTPolicy(cfg, dataset_stats=dataset_metadata.stats)
            print("ACTPolicy successfully created")
        except Exception as e:
            print(f"Error creating ACTPolicy: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    return policy, dataset_metadata, action_type_ckpt_dir


def prepare_data(dataset_name, policy, dataset_metadata):
    """Prepare data for training using the new API"""
    # Policy의 config에서 delta_timestamps 해석
    delta_timestamps = resolve_delta_timestamps(policy.config, dataset_metadata)
    
    # Define image augmentations (excluding flips and rotations)
    image_augmentation_transforms = transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
        # Add 5 RandomErasing masks
        transforms.RandomErasing(p=1.0, scale=(0.01, 0.015), ratio=(0.95, 1.05), value=0),
        transforms.RandomErasing(p=1.0, scale=(0.01, 0.015), ratio=(0.95, 1.05), value=0),
        transforms.RandomErasing(p=1.0, scale=(0.01, 0.015), ratio=(0.95, 1.05), value=0),
        transforms.RandomErasing(p=1.0, scale=(0.01, 0.015), ratio=(0.95, 1.05), value=0),
        transforms.RandomErasing(p=1.0, scale=(0.01, 0.015), ratio=(0.95, 1.05), value=0),
    ])

    # 새 API 방식으로 데이터셋 생성, image_transforms 인자 사용
    dataset = LeRobotDataset(
        dataset_name, 
        delta_timestamps=delta_timestamps, 
        root='./demo_data',
        image_transforms=image_augmentation_transforms # Pass the defined transforms
    )
    
    # 훈련용 데이터로더 생성
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=128,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    
    return dataset, dataloader


def train_policy(policy, dataset, dataloader, ckpt_dir, num_epochs=3000):
    """Train the policy on the dataset"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.train()
    policy.to(device)
    
    # 옵티마이저 설정
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    
    os.makedirs(ckpt_dir, exist_ok=True)
    losses = []
    
    # 훈련 루프
    step = 0
    for epoch in range(num_epochs // len(dataloader) + 1):
        for batch in dataloader:
            if step >= num_epochs:
                break
                
            inp_batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) 
                         for k, v in batch.items()}
            loss, _ = policy.forward(inp_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            
            if step % 100 == 0:
                print(f"step: {step} loss: {loss.item():.4f}")
                
            step += 1
            if step >= num_epochs:
                break
    
    # 최종 모델 저장
    policy.save_pretrained(ckpt_dir)
    print(f"Training completed. Model saved to {ckpt_dir}")
    
    return losses


def evaluate_policy(policy, dataset, device, episode_index=0):
    """Evaluate the policy on the dataset"""
    policy.eval()
    actions = []
    gt_actions = []
    images = []
    
    # Create an episode sampler to sample frames from a specific episode
    episode_sampler = EpisodeSampler(dataset, episode_index)
    test_dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=1,
        shuffle=False,
        pin_memory=device.type != "cpu",
        sampler=episode_sampler,
    )
    
    # Reset policy state
    policy.reset()
    
    # Collect predictions
    print("Evaluating policy...")
    for batch in test_dataloader:
        inp_batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        action = policy.select_action(inp_batch)
        actions.append(action)
        gt_actions.append(inp_batch["action"][:, 0, :])
        images.append(inp_batch["observation.image"] if "observation.image" in inp_batch else None)
    
    # Concatenate results
    if actions:
        actions = torch.cat(actions, dim=0)
        gt_actions = torch.cat(gt_actions, dim=0)
        print(f"Mean action error: {torch.mean(torch.abs(actions[:, :gt_actions.size(1)] - gt_actions)).item():.3f}")
        return gt_actions, actions
    else:
        print("No actions collected during evaluation")
        return None, None


def plot_results(gt_actions: torch.Tensor, pred_actions: torch.Tensor, save_dir: str):
    """
    Plot the evaluation results and save them to the specified directory.
    
    Args:
        gt_actions: Ground truth actions
        pred_actions: Predicted actions
        save_dir: Directory to save plots
    """
    if gt_actions is None or pred_actions is None:
        print("No actions to plot")
        return
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving plots to: {save_dir}")
    
    # Convert to numpy
    gt_np = gt_actions.cpu().detach().numpy()
    pred_np = pred_actions.cpu().detach().numpy()
    
    # Plot each action dimension
    action_dim = gt_np.shape[1]
    fig, axs = plt.subplots(action_dim, 1, figsize=(10, 2*action_dim))
    
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
    plt.close()  # Close the figure to free memory
    
    # Plot error heatmap
    error = np.abs(pred_np[:, :gt_np.shape[1]] - gt_np)
    plt.figure(figsize=(10, 6))
    plt.imshow(error.T, aspect='auto', cmap='hot')
    plt.colorbar(label='Absolute Error')
    plt.xlabel('Time Step')
    plt.ylabel('Action Dimension')
    plt.title('Error Heatmap')
    heatmap_path = os.path.join(save_dir, 'error_heatmap.png')
    plt.savefig(heatmap_path)
    print(f"Saved error heatmap to: {heatmap_path}")
    plt.close()  # Close the figure to free memory
    
    # Plot error histogram
    plt.figure(figsize=(10, 6))
    plt.hist(error.flatten(), bins=50)
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    histogram_path = os.path.join(save_dir, 'error_histogram.png')
    plt.savefig(histogram_path)
    print(f"Saved error histogram to: {histogram_path}")
    plt.close()  # Close the figure to free memory


class EpisodeSampler(torch.utils.data.Sampler):
    """Sample frames from a specific episode"""
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train ACT policy with selected action type')
    parser.add_argument('--action_type', type=str, default='joint', 
                        choices=['joint', 'ee_pose', 'delta_q'],
                        help='Action type to use for training: joint, ee_pose, or delta_q')
    args = parser.parse_args()
    
    # Configuration
    REPO_NAME = 'omy_pnp'
    ROOT = "./demo_data"  # Path to demonstration data
    CKPT_DIR = "./ckpt/act_y"  # Path to save checkpoints
    
    # Set action type from command line argument
    ACTION_TYPE = args.action_type
    print(f"Training with action type: {ACTION_TYPE}")
    
    # 먼저 데이터셋 내용을 검증하고 필요한 데이터가 있는지 확인
    try:
        # 데이터셋 메타데이터 로드
        dataset_metadata = LeRobotDatasetMetadata(REPO_NAME, root=ROOT)
        print(f"Dataset contains the following features:")
        for k, v in dataset_metadata.features.items():
            print(f"  - {k}: {v}")
        
        # 필요한 액션 타입이 있는지 확인
        action_feature_name = f"action.{ACTION_TYPE}" if ACTION_TYPE != 'joint' else 'action.joint'
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
    
    # Try to load the dataset and create policy
    try:
        policy, dataset_metadata, action_type_ckpt_dir = create_or_load_policy(
            CKPT_DIR, action_type=ACTION_TYPE, load_ckpt=False)
        dataset, dataloader = prepare_data(REPO_NAME, policy, dataset_metadata)
        print(f"Dataset loaded with {dataset.num_episodes} episodes")
    except Exception as e:
        print(f"Failed to load dataset or create policy: {e}")
        print("Please make sure you have collected data or are using the correct path.")
        import traceback
        traceback.print_exc()
        return
    
    # Train the policy
    try:
        print("Starting training...")
        losses = train_policy(policy, dataset, dataloader, action_type_ckpt_dir)
        
        # Plot training loss and save in action-specific checkpoint directory
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title(f'Training Loss ({ACTION_TYPE})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(action_type_ckpt_dir, 'training_loss.png'))
        
        # Evaluate the policy
        print("Evaluating policy...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gt_actions, pred_actions = evaluate_policy(policy, dataset, device, episode_index=0)
        
        # Plot evaluation results and save in action-specific checkpoint directory
        if gt_actions is not None and pred_actions is not None:
            plot_results(gt_actions, pred_actions, action_type_ckpt_dir)
    except Exception as e:
        print(f"Error during training or evaluation: {e}")
        import traceback
        traceback.print_exc()
        
    print("Script completed.")


if __name__ == "__main__":
    main()
