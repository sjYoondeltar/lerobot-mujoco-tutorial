#!/usr/bin/env python
# coding: utf-8

"""
Train Temporal Difference Model Predictive Control (TD-MPC)

This script trains a TD-MPC model on the collected robot demonstration dataset.
The trained checkpoint will be saved in the './ckpt/tdmpc_y' folder.
"""

import os
import sys
import argparse

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) # Adjust path if script is moved

import numpy as np
import time
import torch
import matplotlib.pyplot as plt
from torchvision import transforms # Import transforms

# Import TD-MPC specific components from LeRobot
from lerobot.common.policies.tdmpc.modeling_tdmpc import TDMPCPolicy
from lerobot.common.policies.tdmpc.configuration_tdmpc import TDMPCConfig

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.datasets.factory import resolve_delta_timestamps

# Assuming EpisodeSampler might still be useful for evaluation structure
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


def create_or_load_policy(ckpt_dir, action_type='joint', load_ckpt=False, root_dir='./demo_data_4'):
    """
    Create a new TD-MPC policy or load from checkpoint

    Args:
        ckpt_dir: Directory to save or load the checkpoint from
        action_type: Type of action to train with ('joint', 'eef_pose', or 'delta_q')
        load_ckpt: Whether to load from checkpoint
        root_dir: Root directory for dataset
    """
    dataset_root = os.path.join(root_dir, action_type)
    dataset_metadata = LeRobotDatasetMetadata("omy_pnp", root=dataset_root) # Assuming dataset name is consistent
    features = dataset_to_policy_features(dataset_metadata.features)

    print(f"Available features: {list(features.keys())}")
    print("Feature types:")
    for k, v in features.items():
        print(f"  - {k}: type={v.type if hasattr(v, 'type') else 'None'}, shape={v.shape if hasattr(v, 'shape') else 'None'}")

    output_features = {k: v for k, v in features.items() if k == "action" and v.type is FeatureType.ACTION}
    if not output_features:
        print(f"WARNING: No output features found for action_type '{action_type}'")
        raise ValueError(f"No features found for action type: {action_type}")

    # Filter input features: Keep only non-action features
    input_features = {key: ft for key, ft in features.items() if ft.type is not FeatureType.ACTION}

    # TD-MPC currently supports only one image. Select one, excluding wrist_image if others are available.
    visual_features = {k: v for k, v in input_features.items() if v.type is FeatureType.VISUAL}
    non_visual_inputs = {k: v for k, v in input_features.items() if v.type is not FeatureType.VISUAL}
    
    exclude_key = 'observation.wrist_image'
    filtered_visual_features = {k: v for k, v in visual_features.items() if k != exclude_key}
    
    selected_visual_feature = None
    if len(filtered_visual_features) > 0:
        # Use the first available image key that is not the excluded key
        selected_key = next(iter(filtered_visual_features))
        selected_visual_feature = {selected_key: filtered_visual_features[selected_key]}
        print(f"Visual features found (excluding '{exclude_key}' if present). Using: {selected_key}")
        input_features = {**non_visual_inputs, **selected_visual_feature}
    elif exclude_key in visual_features: 
        # Only the excluded key was found, use it as a last resort
        selected_key = exclude_key
        selected_visual_feature = {selected_key: visual_features[selected_key]}
        print(f"Only '{exclude_key}' found. Using it as fallback.")
        input_features = {**non_visual_inputs, **selected_visual_feature}
    else:
        # No visual features found at all
        print("WARNING: No visual features found in input features. Proceeding without image data.")
        input_features = non_visual_inputs # Use only non-visual inputs

    print("Final input features for TD-MPC:")
    for k, v in input_features.items():
        print(f"  - {k}: type={v.type if hasattr(v, 'type') else 'None'}, shape={v.shape if hasattr(v, 'shape') else 'None'}")

    # Configure TD-MPC (adjust parameters as needed for TDMPCConfig)
    # Example parameters - replace with actual required TD-MPC config fields
    cfg = TDMPCConfig(
        input_features=input_features,
        output_features=output_features,
        horizon=12,  # Example TD-MPC specific parameter
        # Add other necessary TDMPCConfig parameters here...
    )

    action_type_ckpt_dir = os.path.join(ckpt_dir, action_type)

    if load_ckpt and os.path.exists(action_type_ckpt_dir):
        print(f"Loading policy from {action_type_ckpt_dir}")
        # Use TDMPCPolicy.from_pretrained
        policy = TDMPCPolicy.from_pretrained(action_type_ckpt_dir)
    else:
        print(f"Creating new TD-MPC policy for action type: {action_type}")
        try:
            # Use TDMPCPolicy constructor
            policy = TDMPCPolicy(cfg, dataset_stats=dataset_metadata.stats)
            print("TDMPCPolicy successfully created")
        except Exception as e:
            print(f"Error creating TDMPCPolicy: {e}")
            import traceback
            traceback.print_exc()
            raise

    return policy, dataset_metadata, action_type_ckpt_dir


def prepare_data(dataset_name, policy, dataset_metadata, action_type, root_dir='./demo_data_4'):
    """
    Prepare data for training using the LeRobotDataset API (similar to ACT)

    Args:
        dataset_name: Name of the dataset
        policy: Policy for which to prepare data
        dataset_metadata: Metadata for the dataset
        action_type: Type of action to train with
        root_dir: Root directory for dataset
    """
    delta_timestamps = resolve_delta_timestamps(policy.config, dataset_metadata)

    # Define image augmentations (reuse from ACT script or adjust for TD-MPC)
    image_augmentation_transforms = transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
        transforms.RandomErasing(p=1.0, scale=(0.01, 0.015), ratio=(0.95, 1.05), value=0),
        transforms.RandomErasing(p=1.0, scale=(0.01, 0.015), ratio=(0.95, 1.05), value=0),
        transforms.RandomErasing(p=1.0, scale=(0.01, 0.015), ratio=(0.95, 1.05), value=0),
        transforms.RandomErasing(p=1.0, scale=(0.01, 0.015), ratio=(0.95, 1.05), value=0),
        transforms.RandomErasing(p=1.0, scale=(0.01, 0.015), ratio=(0.95, 1.05), value=0),
    ])

    dataset_root = os.path.join(root_dir, action_type)

    # Create dataset (ensure it loads necessary fields for TD-MPC, e.g., rewards if needed)
    dataset = LeRobotDataset(
        dataset_name, # Use the provided dataset name
        delta_timestamps=delta_timestamps,
        root=dataset_root,
        image_transforms=image_augmentation_transforms
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=128, # Adjust batch size as needed for TD-MPC memory
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    return dataset, dataloader


def train_policy(policy, dataset, dataloader, ckpt_dir, action_type, num_epochs=3000):
    """ Train the TD-MPC policy """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.train()
    policy.to(device)

    # Optimizer (adjust learning rate or optimizer type if needed for TD-MPC)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4) # Or use policy.configure_optimizers() if available

    os.makedirs(ckpt_dir, exist_ok=True)
    losses = []

    step = 0
    current_epoch = 0
    # Calculate steps per epoch once
    steps_per_epoch = len(dataloader)
    total_epochs = (num_epochs + steps_per_epoch - 1) // steps_per_epoch # Calculate total epochs needed

    print(f"Total steps: {num_epochs}, Steps per epoch: {steps_per_epoch}, Total epochs: {total_epochs}")

    for epoch in range(total_epochs):
        current_epoch = epoch
        for batch_idx, batch in enumerate(dataloader):
            if step >= num_epochs:
                print(f"Reached target steps {num_epochs}. Stopping training.")
                break

            start_time = time.time()

            # Prepare batch data
            inp_batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                         for k, v in batch.items()}

            # TD-MPC specific training step
            # LeRobot policies typically return outputs including loss when called directly
            outputs = policy(inp_batch)
            
            # Check if the outputs contain the loss
            if "loss" not in outputs:
                raise KeyError("The policy output dictionary does not contain a 'loss' key. Check policy implementation.")
                
            total_loss = outputs["loss"]
            
            # For logging purposes, extract other potential loss components if available
            # This part depends on what TDMPCPolicy returns. Adapt as necessary.
            if isinstance(total_loss, torch.Tensor):
                loss_value = total_loss.item()
                # Try to get detailed losses if the policy returns more than just the total loss
                # Example: loss_dict_str = {k: f"{v.item():.4f}" for k, v in outputs.items() if 'loss' in k and isinstance(v, torch.Tensor)}
                # If only total loss is available:
                loss_dict_str = {"total_loss": f"{loss_value:.4f}"} 
            else:
                 raise TypeError(f"Expected 'loss' in policy outputs to be a Tensor, but got {type(total_loss)}")

            # Backpropagation and optimization
            optimizer.zero_grad()
            total_loss.backward()
            # Optional: Gradient clipping (common in RL)
            # torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            losses.append(loss_value)
            end_time = time.time()
            step_time = end_time - start_time

            # Logging
            if step % 100 == 0:
                 print(f"Step {step}/{num_epochs}, Epoch {current_epoch}/{total_epochs-1}, Batch {batch_idx}/{steps_per_epoch-1}, "
                       f"Loss: {loss_value:.4f}, Losses: {loss_dict_str}, Step Time: {step_time:.2f}s")

            # Save checkpoint periodically
            if step % 1000 == 0 and step > 0: # Save less frequently than ACT? Adjust as needed
                step_ckpt_dir = os.path.join(ckpt_dir, f'step_{step}')
                os.makedirs(step_ckpt_dir, exist_ok=True)
                policy.save_pretrained(step_ckpt_dir)

                # Save loss plot
                plt.figure()
                plt.plot(losses)
                plt.xlabel('Steps')
                plt.ylabel('Loss')
                plt.title(f'TD-MPC Training Loss for {action_type} (Step {step})')
                plt.savefig(os.path.join(step_ckpt_dir, f'loss_step_{step}.png'))
                plt.close()

                print(f"Saved checkpoint at step {step} to {step_ckpt_dir}")

            step += 1

        if step >= num_epochs:
            break # Exit outer loop if target steps reached

    # Final save
    final_ckpt_dir = os.path.join(ckpt_dir, 'final')
    os.makedirs(final_ckpt_dir, exist_ok=True)
    policy.save_pretrained(final_ckpt_dir)

    # Final loss plot
    plt.figure()
    plt.plot(losses)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title(f'TD-MPC Training Loss for {action_type} (Final)')
    plt.savefig(os.path.join(final_ckpt_dir, 'loss_final.png'))
    plt.close()

    print(f"Finished training. Final checkpoint saved to {final_ckpt_dir}")
    return losses


def evaluate_policy(policy, dataset, device, action_type, episode_index=0):
    """
    Evaluate TD-MPC policy on a specific episode (using dataset actions as reference)

    Note: This evaluates the policy's action prediction accuracy on offline data.
          A full evaluation would typically involve running the policy online in the env.
    """
    test_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32, # Process multiple steps at once if possible
        sampler=EpisodeSampler(dataset, episode_index),
        shuffle=False,
    )

    actions = []
    gt_actions = []
    # images = [] # Uncomment if needed for visualization

    policy.eval()
    policy.reset() # Reset any internal state if necessary

    print(f"Evaluating policy on episode {episode_index}...")
    with torch.no_grad(): # Disable gradients during evaluation
        for batch in test_dataloader:
            inp_batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

            # Use select_action for TD-MPC (performs planning internally)
            action = policy.select_action(inp_batch)
            actions.append(action)

            # Ground truth action for comparison
            # Assuming ground truth is the first action in the sequence if dataset provides sequences
            current_gt_action = inp_batch["action"][:, 0, :] if inp_batch["action"].ndim == 3 else inp_batch["action"]
            gt_actions.append(current_gt_action)
            # images.append(inp_batch.get("observation.image")) # Get images if present

    # Concatenate results
    if actions:
        actions = torch.cat(actions, dim=0)
        gt_actions = torch.cat(gt_actions, dim=0)

        # Ensure consistent length for comparison
        min_len = min(actions.shape[0], gt_actions.shape[0])
        actions = actions[:min_len]
        gt_actions = gt_actions[:min_len]

        # Ensure action dimensions match (predicted vs ground truth)
        pred_dim = actions.shape[-1]
        gt_dim = gt_actions.shape[-1]
        compare_dim = min(pred_dim, gt_dim)
        actions = actions[..., :compare_dim]
        gt_actions = gt_actions[..., :compare_dim]

        action_error = torch.mean(torch.abs(actions - gt_actions)).item()
        print(f"Mean action error (vs offline data): {action_error:.4f}")
        return gt_actions, actions
    else:
        print("No actions collected during evaluation")
        return None, None


def plot_results(gt_actions: torch.Tensor, pred_actions: torch.Tensor, save_dir: str):
    """
    Plot the evaluation results (predicted vs ground truth actions) and save them.
    (Reusing the function from train.py, assuming it's suitable)
    """
    if gt_actions is None or pred_actions is None:
        print("No actions to plot")
        return

    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving evaluation plots to: {save_dir}")

    gt_np = gt_actions.cpu().detach().numpy()
    pred_np = pred_actions.cpu().detach().numpy()

    # Ensure shapes match exactly before plotting
    min_samples = min(gt_np.shape[0], pred_np.shape[0])
    action_dim = gt_np.shape[1] # Assumes gt_actions and pred_actions now have same dim

    gt_np = gt_np[:min_samples]
    pred_np = pred_np[:min_samples]

    fig, axs = plt.subplots(action_dim, 1, figsize=(10, 2 * action_dim), sharex=True)
    if action_dim == 1:
        axs = [axs] # Make it iterable

    time_steps = np.arange(min_samples)

    for i in range(action_dim):
        axs[i].plot(time_steps, pred_np[:, i], label="Prediction (TD-MPC)")
        axs[i].plot(time_steps, gt_np[:, i], label="Ground Truth (Dataset)")
        axs[i].set_title(f"Action Dimension {i}")
        axs[i].legend()
        axs[i].grid(True)

    axs[-1].set_xlabel("Time Step")
    plt.tight_layout()
    action_plot_path = os.path.join(save_dir, 'action_comparison.png')
    plt.savefig(action_plot_path)
    print(f"Saved action comparison plot to: {action_plot_path}")
    plt.close(fig)

    # Plot error heatmap
    error = np.abs(pred_np - gt_np)
    plt.figure(figsize=(10, max(6, action_dim * 0.5))) # Adjust height based on dim
    plt.imshow(error.T, aspect='auto', cmap='viridis', interpolation='none') # Changed cmap
    plt.colorbar(label='Absolute Error')
    plt.xlabel('Time Step')
    plt.ylabel('Action Dimension')
    plt.title('Action Prediction Error Heatmap')
    heatmap_path = os.path.join(save_dir, 'error_heatmap.png')
    plt.savefig(heatmap_path)
    print(f"Saved error heatmap to: {heatmap_path}")
    plt.close()

    # Plot error histogram
    plt.figure(figsize=(10, 6))
    plt.hist(error.flatten(), bins=50, log=True) # Use log scale for better visibility
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency (log scale)')
    plt.title('Action Prediction Error Distribution')
    plt.grid(axis='y')
    histogram_path = os.path.join(save_dir, 'error_histogram.png')
    plt.savefig(histogram_path)
    print(f"Saved error histogram to: {histogram_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train the TD-MPC model.')
    # Reuse arguments from ACT, potentially add TD-MPC specific ones
    parser.add_argument('--action_type', type=str, choices=['joint', 'eef_pose', 'delta_q'], default='joint',
                        help='Type of action data to use')
    parser.add_argument('--load_ckpt', action='store_true', help='Load from the latest checkpoint')
    parser.add_argument('--num_epochs', type=int, default=50000, help='Number of training steps (not epochs)') # Changed to steps
    parser.add_argument('--data_root', type=str, default='./demo_data_4', help='Path to demonstration data directory')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt/tdmpc_y_v1', help='Path to save TD-MPC checkpoints') # Changed default dir
    parser.add_argument('--dataset_name', type=str, default='omy_pnp', help='Name of the dataset (e.g., huggingface repo id)')
    # Add TD-MPC specific arguments if needed (e.g., horizon, learning rates)
    # parser.add_argument('--horizon', type=int, default=12, help='TD-MPC planning horizon')

    args = parser.parse_args()

    print(f"\n=== Training TD-MPC model with action type: {args.action_type} ===")
    print(f"Arguments: {args}")

    try:
        # Create/Load Policy
        policy, dataset_metadata, action_type_ckpt_dir = create_or_load_policy(
            args.ckpt_dir, action_type=args.action_type, load_ckpt=args.load_ckpt, root_dir=args.data_root
        )

        # Prepare Data
        dataset, dataloader = prepare_data(args.dataset_name, policy, dataset_metadata, args.action_type, root_dir=args.data_root)

        # Train Policy
        print("Starting TD-MPC training...")
        train_policy(policy, dataset, dataloader, action_type_ckpt_dir, args.action_type, num_epochs=args.num_epochs)

        # Evaluate Policy (offline comparison)
        print("Starting evaluation (offline action comparison)...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load the final checkpoint for evaluation
        final_ckpt_path = os.path.join(action_type_ckpt_dir, 'final')
        if os.path.exists(final_ckpt_path):
             print(f"Loading final checkpoint from {final_ckpt_path} for evaluation.")
             eval_policy = TDMPCPolicy.from_pretrained(final_ckpt_path)
             eval_policy.to(device)
             eval_policy.eval() # Set to evaluation mode

             gt_actions, pred_actions = evaluate_policy(eval_policy, dataset, device, args.action_type, episode_index=0)

             # Plot evaluation results
             if gt_actions is not None and pred_actions is not None:
                 print("Plotting evaluation results...")
                 eval_save_dir = os.path.join(action_type_ckpt_dir, 'final', 'evaluation_results')
                 plot_results(gt_actions, pred_actions, eval_save_dir)
             else:
                 print("Evaluation did not produce results to plot.")
        else:
            print(f"Final checkpoint not found at {final_ckpt_path}. Skipping evaluation plotting.")

    except Exception as e:
        print(f"Error during script execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) # Exit with error code

    print("\n=== TD-MPC Training Script Finished ===\n")


if __name__ == "__main__":
    main() 