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
        n_action_pred_token=5,
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
        batch_size=64,
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
    eval_metrics = []
    
    # Training loop
    step = 0
    current_epoch = 0
    for epoch in range(num_epochs // len(dataloader) + 1):
        current_epoch = epoch
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
                print(f"Step: {step}, Epoch: {current_epoch}, Loss: {loss_value:.4f}")
                
                # Run evaluation every 100 steps
                print(f"Running evaluation at step {step}...")
                # Switch to evaluation mode
                policy.eval()
                # Select a random episode for evaluation
                episode_index = np.random.randint(0, dataset.num_episodes)
                gt_actions, pred_actions = evaluate_policy(policy, dataset, device, episode_index)
                
                # Calculate and log evaluation metrics if available
                if gt_actions is not None and pred_actions is not None and gt_actions.shape == pred_actions.shape:
                    mae = torch.mean(torch.abs(pred_actions - gt_actions)).item()
                    eval_metrics.append((step, mae))
                    print(f"Evaluation MAE at step {step}: {mae:.4f}")
                    
                    # Optionally save plots at regular intervals
                    if step % 1000 == 0:
                        eval_dir = os.path.join(ckpt_dir, f"eval_step_{step}")
                        os.makedirs(eval_dir, exist_ok=True)
                        plot_results(gt_actions, pred_actions, eval_dir)
                
                # Switch back to training mode
                policy.train()
                
            step += 1
            if step >= num_epochs:
                break
        
        # 100 에포크마다 모델 저장 (에포크 번호 포함)
        if current_epoch % 100 == 0 and current_epoch > 0:
            epoch_ckpt_dir = os.path.join(ckpt_dir, f'epoch_{current_epoch}')
            os.makedirs(epoch_ckpt_dir, exist_ok=True)
            policy.save_pretrained(epoch_ckpt_dir)
            
            # 손실 그래프 저장
            plt.figure()
            plt.plot(losses)
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.title(f'Training Loss (Epoch {current_epoch})')
            plt.savefig(os.path.join(epoch_ckpt_dir, f'loss_epoch_{current_epoch}.png'))
            plt.close()
            
            print(f"Saved checkpoint at epoch {current_epoch}")
    
    # Final evaluation
    print("Running final evaluation...")
    policy.eval()
    gt_actions, pred_actions = evaluate_policy(policy, dataset, device, episode_index=0)
    if gt_actions is not None and pred_actions is not None and gt_actions.shape == pred_actions.shape:
        final_mae = torch.mean(torch.abs(pred_actions - gt_actions)).item()
        print(f"Final evaluation MAE: {final_mae:.4f}")
        eval_metrics.append((step, final_mae))
        
        # Save final evaluation plots
        final_eval_dir = os.path.join(ckpt_dir, "final_eval")
        os.makedirs(final_eval_dir, exist_ok=True)
        plot_results(gt_actions, pred_actions, final_eval_dir)
    
    # Save final model in 'final' directory
    final_ckpt_dir = os.path.join(ckpt_dir, 'final')
    os.makedirs(final_ckpt_dir, exist_ok=True)
    policy.save_pretrained(final_ckpt_dir)
    print(f"Training completed. Final model saved to {final_ckpt_dir}")
    
    # Save training and evaluation metrics in final directory
    if eval_metrics:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        
        plt.subplot(1, 2, 2)
        steps, maes = zip(*eval_metrics)
        plt.plot(steps, maes, 'r-')
        plt.title('Evaluation MAE')
        plt.xlabel('Step')
        plt.ylabel('Mean Absolute Error')
        
        plt.tight_layout()
        plt.savefig(os.path.join(final_ckpt_dir, 'training_metrics_final.png'))
        plt.close()
    
    return losses, eval_metrics


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
            action = policy.select_action(inp_batch)
        
        actions.append(action[:, 0, :])
        gt_actions.append(inp_batch["action"][:, 0, :])
        images.append(inp_batch["observation.image"] if "observation.image" in inp_batch else None)
    
    # Concatenate results
    if actions:
        actions = torch.cat(actions, dim=0)
        gt_actions = torch.cat(gt_actions, dim=0)
        
        # Ensure dimensions match before calculating error
        if actions.shape[1] != gt_actions.shape[1]:
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


def train_vqvae(policy, dataset, dataloader, ckpt_dir, num_epochs=1000):
    """Train only the VQ-VAE part of the model first"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.train()
    policy.to(device)
    
    # Setup optimizer that only updates VQ-VAE parameters
    # First identify VQ-VAE related parameters
    vqvae_params = []
    for name, param in policy.named_parameters():
        # Only select parameters from the encoder, decoder, and codebook
        if any(part in name for part in ['encoder', 'decoder', 'codebook', 'quantizer']):
            vqvae_params.append(param)
            param.requires_grad = True
        else:
            # Freeze other parameters (transformer/GPT parts)
            param.requires_grad = False
    
    # Setup optimizer for VQ-VAE params only
    optimizer = torch.optim.Adam(vqvae_params, lr=3e-4)
    
    os.makedirs(ckpt_dir, exist_ok=True)
    vqvae_ckpt_dir = os.path.join(ckpt_dir, "vqvae_only")
    os.makedirs(vqvae_ckpt_dir, exist_ok=True)
    
    losses = []
    recon_losses = []
    vq_losses = []
    
    # Training loop
    step = 0
    best_loss = float('inf')
    current_epoch = 0
    
    print("Starting VQ-VAE pre-training...")
    for epoch in range(num_epochs // len(dataloader) + 1):
        current_epoch = epoch
        for batch in dataloader:
            if step >= num_epochs:
                break
                
            inp_batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) 
                       for k, v in batch.items()}
            
            # Forward pass through VQ-VAE only
            # We need to modify this based on how the forward method works
            # Here we assume the forward method returns vq_loss and recon_loss as part of its output
            try:
                with torch.set_grad_enabled(True):
                    # Extract actions for VQ-VAE training
                    actions = inp_batch["action"]
                    
                    # Call the encoder, quantizer, and decoder parts directly if accessible
                    if hasattr(policy, 'vqvae_forward'):
                        loss, loss_dict = policy.vqvae_forward(actions)
                    else:
                        # Fallback: call regular forward but extract VQ-VAE losses
                        loss, loss_dict = policy.forward(inp_batch)
                        # We might need to extract just the VQ-VAE related losses
                        if isinstance(loss_dict, dict) and 'vq_loss' in loss_dict:
                            loss = loss_dict['vq_loss'] + loss_dict.get('reconstruction_loss', 0)
            except Exception as e:
                print(f"Error during VQ-VAE forward pass: {e}")
                continue
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # For tracking, get loss values
            loss_value = loss.item()
            losses.append(loss_value)
            
            # Extract component losses if available
            if isinstance(loss_dict, dict):
                if 'reconstruction_loss' in loss_dict:
                    recon_losses.append(loss_dict['reconstruction_loss'].item())
                if 'vq_loss' in loss_dict:
                    vq_losses.append(loss_dict['vq_loss'].item())
            
            if step % 100 == 0:
                recon_loss_str = f", recon_loss: {recon_losses[-1]:.4f}" if recon_losses else ""
                vq_loss_str = f", vq_loss: {vq_losses[-1]:.4f}" if vq_losses else ""
                print(f"VQ-VAE Step: {step}, Epoch: {current_epoch}, Loss: {loss_value:.4f}{recon_loss_str}{vq_loss_str}")
            
            step += 1
            if step >= num_epochs:
                break
        
        # 100 에포크마다 모델 저장 (에포크 번호 포함)
        if current_epoch % 100 == 0 and current_epoch > 0:
            epoch_vqvae_dir = os.path.join(vqvae_ckpt_dir, f'epoch_{current_epoch}')
            os.makedirs(epoch_vqvae_dir, exist_ok=True)
            
            # Save checkpoint with epoch number
            vqvae_epoch_path = os.path.join(epoch_vqvae_dir, f"model_epoch_{current_epoch}.pt")
            torch.save({
                'epoch': current_epoch,
                'step': step,
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_value,
            }, vqvae_epoch_path)
            
            # Save loss plot
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 3, 1)
            plt.plot(losses)
            plt.title(f'Total VQ-VAE Loss (Epoch {current_epoch})')
            plt.xlabel('Step')
            plt.ylabel('Loss')
            
            if recon_losses:
                plt.subplot(1, 3, 2)
                plt.plot(recon_losses)
                plt.title('Reconstruction Loss')
                plt.xlabel('Step')
            
            if vq_losses:
                plt.subplot(1, 3, 3)
                plt.plot(vq_losses)
                plt.title('VQ Loss')
                plt.xlabel('Step')
            
            plt.tight_layout()
            plt.savefig(os.path.join(epoch_vqvae_dir, f'vqvae_losses_epoch_{current_epoch}.png'))
            plt.close()
            
            print(f"Saved VQ-VAE checkpoint at epoch {current_epoch}")
    
    # Save final VQ-VAE model
    final_vqvae_dir = os.path.join(vqvae_ckpt_dir, "final")
    os.makedirs(final_vqvae_dir, exist_ok=True)
    final_vqvae_path = os.path.join(final_vqvae_dir, "final_model.pt")
    torch.save({
        'epoch': current_epoch,
        'step': step,
        'model_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': losses[-1] if losses else float('inf'),
    }, final_vqvae_path)
    print(f"VQ-VAE training completed. Final model saved to {final_vqvae_path}")
    
    # Plot VQ-VAE training losses
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.title('Total VQ-VAE Loss (Final)')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    
    if recon_losses:
        plt.subplot(1, 3, 2)
        plt.plot(recon_losses)
        plt.title('Reconstruction Loss')
        plt.xlabel('Step')
    
    if vq_losses:
        plt.subplot(1, 3, 3)
        plt.plot(vq_losses)
        plt.title('VQ Loss')
        plt.xlabel('Step')
    
    plt.tight_layout()
    plt.savefig(os.path.join(final_vqvae_dir, 'vqvae_training_losses_final.png'))
    plt.close()
    
    # Unfreeze all parameters for subsequent training
    for param in policy.parameters():
        param.requires_grad = True
    
    return policy, losses


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
    
    # First, train VQ-VAE component separately
    print("Step 1: Pre-training VQ-VAE component...")
    policy, vqvae_losses = train_vqvae(policy, dataset, dataloader, CKPT_DIR, num_epochs=1000)
    
    # Then train the full VQBeT model (with pre-trained VQ-VAE)
    print("\nStep 2: Training full VQBeT model with pre-trained VQ-VAE...")
    losses, eval_metrics = train_policy(policy, dataset, dataloader, CKPT_DIR, num_epochs=2000)
    
    # Create directories if they don't exist
    if not os.path.exists(CKPT_DIR):
        os.makedirs(CKPT_DIR, exist_ok=True)
    
    # Training metrics are already saved in train_policy function
    # But we can add additional visualization or logging here if needed
    if eval_metrics:
        print(f"Final evaluation metrics: MAE = {eval_metrics[-1][1]:.4f}")
        
        # Save evaluation metrics to a CSV file for later analysis
        import csv
        metrics_file = os.path.join(CKPT_DIR, 'eval_metrics.csv')
        with open(metrics_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Step', 'MAE'])
            for step, mae in eval_metrics:
                writer.writerow([step, mae])
        print(f"Saved evaluation metrics to {metrics_file}")
    
    # Evaluate the policy (final detailed evaluation)
    print("Running final detailed evaluation...")
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