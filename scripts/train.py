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
from lerobot.model.act import ACTPolicy
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def create_or_load_policy(ckpt_dir, load_ckpt=False):
    """Create a new policy or load from checkpoint"""
    policy = ACTPolicy(
        obs_dim=6,                # End-effector pose (x, y, z, roll, pitch, yaw)
        action_dim=7,             # 6 joint angles and 1 gripper
        chunk_size=10,            # Temporally abstract 10 actions
        hidden_dim=128,           # Hidden dimension of the transformer
        num_layers=2,             # Number of transformer layers
    )
    
    if load_ckpt and os.path.exists(ckpt_dir):
        policy.load(ckpt_dir)
        print(f"Loaded policy from {ckpt_dir}")
    
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


def plot_results(gt_actions, pred_actions):
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
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('training_loss.png')
    plt.show()
    
    # Evaluate the policy
    print("Evaluating policy...")
    gt_actions, pred_actions = evaluate_policy(policy, all_obs, all_actions)
    
    # Plot evaluation results
    plot_results(gt_actions, pred_actions)


if __name__ == "__main__":
    main()

