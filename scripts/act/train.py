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
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.configs.types import FeatureType
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.datasets.factory import resolve_delta_timestamps


def create_or_load_policy(ckpt_dir, load_ckpt=False):
    """Create a new policy or load from checkpoint"""
    dataset_metadata = LeRobotDatasetMetadata("omy_pnp", root='./demo_data')
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    # input_features.pop("observation.wrist_image")

    # 새로운 API 방식으로 설정
    cfg = ACTConfig(
        input_features=input_features, 
        output_features=output_features, 
        chunk_size=10, 
        n_action_steps=10
    )
    
    if load_ckpt and os.path.exists(ckpt_dir):
        print(f"Loading policy from {ckpt_dir}")
        policy = ACTPolicy.from_pretrained(ckpt_dir)
    else:
        print("Creating new policy")
        policy = ACTPolicy(cfg, dataset_stats=dataset_metadata.stats)
    
    return policy, dataset_metadata


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
    # Configuration
    REPO_NAME = 'omy_pnp'
    ROOT = "./demo_data"  # Path to demonstration data
    CKPT_DIR = "./ckpt/act_y"  # Path to save checkpoints
    
    # Try to load the dataset
    try:
        policy, dataset_metadata = create_or_load_policy(CKPT_DIR, load_ckpt=False)
        dataset, dataloader = prepare_data(REPO_NAME, policy, dataset_metadata)
        print(f"Dataset loaded with {dataset.num_episodes} episodes")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Please make sure you have collected data or are using the correct path.")
        return
    
    # Train the policy
    print("Starting training...")
    losses = train_policy(policy, dataset, dataloader, CKPT_DIR)
    
    # Create directories if they don't exist
    if not os.path.exists(CKPT_DIR):
        os.makedirs(CKPT_DIR, exist_ok=True)
    
    # Plot training loss and save in CKPT_DIR
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(CKPT_DIR, 'training_loss.png'))
    # plt.show()
    
    # Evaluate the policy
    print("Evaluating policy...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gt_actions, pred_actions = evaluate_policy(policy, dataset, device, episode_index=0)
    
    # Plot evaluation results and save in CKPT_DIR
    if gt_actions is not None and pred_actions is not None:
        plot_results(gt_actions, pred_actions, CKPT_DIR)


if __name__ == "__main__":
    main()
