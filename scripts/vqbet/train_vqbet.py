#!/usr/bin/env python
# coding: utf-8

"""
Train Vector-Quantized Behavior Transformer (VQBeT)

This script trains a VQBeT model on the collected robot demonstration dataset.
It takes approximately 30-60 minutes to train the model.
The trained checkpoint will be saved in the './ckpt/vqbet_y' folder.
"""

import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import time
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from lerobot.common.policies.vqbet.modeling_vqbet import VQBeTPolicy
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.configs.types import FeatureType
from lerobot.common.policies.vqbet.configuration_vqbet import VQBeTConfig
from lerobot.common.datasets.factory import resolve_delta_timestamps


def create_or_load_policy(ckpt_dir, load_ckpt=False):
    """Create a new policy or load from checkpoint"""
    dataset_metadata = LeRobotDatasetMetadata("omy_pnp", root='./demo_data')
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    
    # For VQBeT, we need only one image input
    # Keep only 'observation.image' and remove other image inputs if they exist
    image_keys = [key for key in input_features.keys() if key.startswith("observation.") and key.endswith("image")]
    if len(image_keys) > 1:
        print(f"Found multiple image inputs: {image_keys}")
        print(f"Keeping only 'observation.image' and removing others.")
        for key in image_keys:
            if key != "observation.image":
                input_features.pop(key, None)
    
    # Print the features being used
    print(f"Input features: {list(input_features.keys())}")
    print(f"Output features: {list(output_features.keys())}")

    # Configure VQBeT model
    cfg = VQBeTConfig(
        input_features=input_features, 
        output_features=output_features,
        n_obs_steps=5,
        n_action_pred_token=3,
        action_chunk_size=5,
        vqvae_n_embed=1024,  # codebook size
        vqvae_embedding_dim=128,  # latent dimension
        vision_backbone="resnet18",
        spatial_softmax_num_keypoints=32,
        gpt_n_layer=8,
        gpt_n_head=8
    )
    
    if load_ckpt and os.path.exists(ckpt_dir):
        print(f"Loading policy from {ckpt_dir}")
        policy = VQBeTPolicy.from_pretrained(ckpt_dir)
    else:
        print("Creating new policy")
        policy = VQBeTPolicy(cfg, dataset_stats=dataset_metadata.stats)
    
    return policy, dataset_metadata


def prepare_data(dataset_name, policy, dataset_metadata):
    """Prepare data for training"""
    # Resolve delta timestamps for the policy configuration
    delta_timestamps = resolve_delta_timestamps(policy.config, dataset_metadata)
    
    # Define image augmentations
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

    # Create dataset with the defined transforms
    dataset = LeRobotDataset(
        dataset_name, 
        delta_timestamps=delta_timestamps, 
        root='./demo_data',
        image_transforms=image_augmentation_transforms
    )
    
    # Create data loader for training
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
    
    # Setup optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    
    os.makedirs(ckpt_dir, exist_ok=True)
    losses = []
    
    # Training loop
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
            
            # For tracking, get loss value
            loss_value = loss.item()
            losses.append(loss_value)
            
            if step % 100 == 0:
                print(f"step: {step} loss: {loss_value:.4f}")
                
            step += 1
            if step >= num_epochs:
                break
    
    # Save final model
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
        
        # Get action prediction from policy - this returns the raw, high-dimensional output
        with torch.no_grad():
            raw_action = policy.select_action(inp_batch)
            
            # For VQBeT, we need to use the unnormalize_outputs to convert to the actual action space
            # This converts from token/latent space to continuous action
            if hasattr(policy, 'unnormalize_outputs'):
                # The original action is stored in the batch with key 'action'
                # This helps us know the expected shape
                expected_action_shape = inp_batch["action"].shape
                
                # Try to use the policy's utility methods to convert
                if hasattr(policy, 'detokenize_action'):
                    # If there's a specific detokenize_action method
                    action = policy.detokenize_action(raw_action)
                else:
                    # If not, we need to pass through unnormalize_outputs
                    # Match the expected shape for the action output
                    action_dict = {"action": raw_action}
                    unnormalized = policy.unnormalize_outputs(action_dict)
                    action = unnormalized["action"]
                    
                    # We might need to reshape to match the expected action shape
                    if action.shape != expected_action_shape:
                        # Just take the first action if it's a sequence
                        action = action[:, 0, :]
            else:
                # Fallback - just use the raw output but warn the user
                action = raw_action
                print("Warning: Cannot properly convert VQBeT output to action format")
        
        actions.append(action)
        gt_actions.append(inp_batch["action"][:, 0, :])
        images.append(inp_batch["observation.image"] if "observation.image" in inp_batch else None)
    
    # Concatenate results
    if actions:
        actions = torch.cat(actions, dim=0)
        gt_actions = torch.cat(gt_actions, dim=0)
        
        # Print action shape information for debugging
        print(f"Predicted action shape: {actions.shape}")
        print(f"Ground truth action shape: {gt_actions.shape}")
        
        # Ensure dimensions match before calculating error
        if actions.shape[1] != gt_actions.shape[1]:
            print(f"Warning: Action dimensions don't match: pred={actions.shape[1]}, gt={gt_actions.shape[1]}")
            # Try to adjust dimensions if possible
            if hasattr(actions, "reshape") and actions.numel() // gt_actions.shape[0] == gt_actions.shape[1]:
                actions = actions.reshape(gt_actions.shape)
        
        # Calculate mean action error only if shapes match
        if actions.shape == gt_actions.shape:
            mae = torch.mean(torch.abs(actions - gt_actions)).item()
            print(f"Mean action error: {mae:.3f}")
        else:
            print("Cannot calculate error: action shapes don't match")
        
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
    CKPT_DIR = "./ckpt/vqbet_y"  # Path to save checkpoints
    
    # Use standard device setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
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
    if losses:
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(CKPT_DIR, 'training_loss.png'))
    
    # Evaluate the policy
    print("Evaluating policy...")
    # Move policy to device for evaluation
    policy.to(device)
    gt_actions, pred_actions = evaluate_policy(policy, dataset, device, episode_index=0)
        
    # Plot evaluation results and save in CKPT_DIR
    if gt_actions is not None and pred_actions is not None:
        # Check if shapes match before attempting to plot
        if gt_actions.shape == pred_actions.shape:
            plot_results(gt_actions, pred_actions, CKPT_DIR)
        else:
            print("Skipping plot_results because action shapes don't match")
            print(f"Ground truth shape: {gt_actions.shape}, Prediction shape: {pred_actions.shape}")


if __name__ == "__main__":
    main() 