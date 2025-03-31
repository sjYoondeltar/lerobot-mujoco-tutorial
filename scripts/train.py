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
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.configs.types import FeatureType
from lerobot.common.policies.act.configuration_act import ACTConfig


def create_or_load_policy(ckpt_dir, load_ckpt=False):
    """Create a new policy or load from checkpoint"""
    dataset_metadata = LeRobotDatasetMetadata("omy_pnp", root='./demo_data')
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    input_features.pop("observation.wrist_image")

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
    
    return policy


def prepare_data(dataset, obs_key="observation.state", chunk_size=10):
    """Prepare data for training"""
    all_obs = []
    all_actions = []
    
    for episode_idx in range(dataset.num_episodes):
        episode = dataset.get_episode(episode_idx)
        obs = episode[obs_key]
        actions = episode["action"]
        
        all_obs.append(obs)
        all_actions.append(actions)
    
    return all_obs, all_actions


def train_policy(policy, all_obs, all_actions, ckpt_dir, num_epochs=20000):
    """Train the policy on the dataset"""
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Training loop
    losses = []
    
    start_time = time.time()
    for epoch in range(num_epochs):
        # Choose a random episode
        episode_idx = np.random.randint(len(all_obs))
        obs = torch.tensor(all_obs[episode_idx], dtype=torch.float32)
        actions = torch.tensor(all_actions[episode_idx], dtype=torch.float32)
        
        # Train on the episode
        loss = policy.train_step(obs, actions)
        losses.append(loss)
        
        # Save the model checkpoint periodically
        if (epoch + 1) % 1000 == 0:
            policy.save(ckpt_dir)
            
            # Calculate time per epoch
            time_per_epoch = (time.time() - start_time) / (epoch + 1)
            print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss:.8f}, "
                  f"Time per epoch: {time_per_epoch:.6f}s")
    
    # Save the final model
    policy.save(ckpt_dir)
    print(f"Training completed. Model saved to {ckpt_dir}")
    
    return losses


def evaluate_policy(policy, all_obs, all_actions):
    """Evaluate the policy on the dataset"""
    episode_idx = 0  # Evaluate on the first episode
    
    obs = torch.tensor(all_obs[episode_idx], dtype=torch.float32)
    gt_actions = torch.tensor(all_actions[episode_idx], dtype=torch.float32)
    
    # Get policy predictions
    with torch.no_grad():
        pred_actions = policy.predict_chunk(obs)
    
    # Convert to numpy for plotting
    gt_actions = gt_actions.numpy()
    pred_actions = pred_actions.numpy()
    
    return gt_actions, pred_actions


def plot_results(gt_actions, pred_actions, save_dir=None):
    """Plot ground truth vs predicted actions"""
    # Plot the joint angles
    plt.figure(figsize=(12, 8))
    
    for i in range(6):  # 6 joint angles
        plt.subplot(3, 2, i+1)
        plt.plot(gt_actions[:, i], label='Ground Truth')
        plt.plot(pred_actions[:, i], label='Prediction')
        plt.title(f'Joint {i+1}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('joint_predictions.png')
    plt.show()
    
    # Plot the gripper
    plt.figure(figsize=(6, 4))
    plt.plot(gt_actions[:, -1], label='Ground Truth')
    plt.plot(pred_actions[:, -1], label='Prediction')
    plt.title('Gripper')
    plt.legend()
    plt.savefig('gripper_prediction.png')
    plt.show()
    
    # Save the plots in save_dir if provided
    if save_dir:
        import os
        # Assuming you have multiple plots, save each one with a descriptive name
        plt.figure(figsize=(12, 8))
        # Example plot 1: Action comparison
        plt.subplot(2, 1, 1)
        plt.plot(gt_actions[:, 0], label='Ground Truth')
        plt.plot(pred_actions[:, 0], label='Prediction')
        plt.legend()
        plt.title('Action Comparison')
        
        # Example plot 2: Error distribution
        plt.subplot(2, 1, 2)
        plt.hist(np.abs(gt_actions - pred_actions).flatten(), bins=50)
        plt.title('Error Distribution')
        
        # Save the combined figure
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'evaluation_results.png'))
        
        # You might want individual plots for each action dimension
        action_dims = gt_actions.shape[1]
        for i in range(action_dims):
            plt.figure(figsize=(10, 6))
            plt.plot(gt_actions[:, i], label='Ground Truth')
            plt.plot(pred_actions[:, i], label='Prediction')
            plt.legend()
            plt.title(f'Action Dimension {i}')
            plt.savefig(os.path.join(save_dir, f'action_dim_{i}.png'))
            plt.close()


def main():
    # Configuration
    REPO_NAME = 'omy_pnp'
    ROOT = "./demo_data"  # Path to demonstration data
    CKPT_DIR = "./ckpt/act_y"  # Path to save checkpoints
    
    # Try to load the dataset
    try:
        dataset = LeRobotDataset(REPO_NAME, root=ROOT)
        print(f"Dataset loaded with {dataset.num_episodes} episodes")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Please make sure you have collected data or are using the correct path.")
        return
    
    # Create a policy
    policy = create_or_load_policy(CKPT_DIR, load_ckpt=False)
    
    # Prepare data for training
    all_obs, all_actions = prepare_data(dataset)
    print(f"Prepared {len(all_obs)} episodes for training")
    
    # Train the policy
    print("Starting training...")
    losses = train_policy(policy, all_obs, all_actions, CKPT_DIR)
    
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
    plt.show()
    
    # Evaluate the policy
    print("Evaluating policy...")
    gt_actions, pred_actions = evaluate_policy(policy, all_obs, all_actions)
    
    # Plot evaluation results and save in CKPT_DIR
    plot_results(gt_actions, pred_actions, save_dir=CKPT_DIR)


if __name__ == "__main__":
    main()

