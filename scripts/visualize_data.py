#!/usr/bin/env python
# coding: utf-8

"""
Visualize Data

This script allows you to playback and visualize robot demonstration data.
It loads the collected dataset and replays the actions in a MuJoCo simulation.
Supports different action types from demo_data_3: joint, eef_pose, delta_q.
"""

import sys
import os
import time
import argparse

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


class EpisodeSampler(torch.utils.data.Sampler):
    """
    Sampler for a single episode
    """
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


def visualize_dataset(env, dataset, episode_idx=0, action_type=None):
    """Visualize a specific episode from the dataset using dataloader"""
    print(f"Visualizing action type: {action_type}, episode: {episode_idx}")
    
    # Create episode sampler and dataloader
    episode_sampler = EpisodeSampler(dataset, episode_idx)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=1,
        batch_size=1,
        sampler=episode_sampler,
    )
    
    # Get an iterator over the dataloader
    iter_dataloader = iter(dataloader)
    
    # Get the first frame to initialize the environment
    try:
        first_frame = next(iter_dataloader)
        # Reset the environment with the object at the initial position
        env.reset()
        env.set_obj_pose(first_frame['obj_init'][0,:3], first_frame['obj_init'][0,3:])
    except StopIteration:
        print(f"Episode {episode_idx} is empty")
        return True
    
    # Reset the iterator to include the first frame
    iter_dataloader = iter(dataloader)
    step = 0
    
    # Playback the episode
    for data in iter_dataloader:
        env.step_env()
        
        # Visualize only if the environment is still alive
        if not env.env.is_viewer_alive():
            return False
            
        if env.env.loop_every(HZ=20):
            # If first frame, set object pose
            if step == 0:
                env.set_obj_pose(data['obj_init'][0,:3], data['obj_init'][0,3:])
            
            # Get the action from dataset
            print(data['action'])
            action = data['action'].numpy()
            obs = env.step(action[0])
            
            # Visualize the images from dataset if available
            if 'observation.image' in data:
                env.rgb_agent = data['observation.image'][0].numpy()*255
                env.rgb_agent = env.rgb_agent.astype(np.uint8)
                env.rgb_agent = np.transpose(env.rgb_agent, (1,2,0))
            
            if 'observation.wrist_image' in data:
                env.rgb_ego = data['observation.wrist_image'][0].numpy()*255
                env.rgb_ego = env.rgb_ego.astype(np.uint8)
                env.rgb_ego = np.transpose(env.rgb_ego, (1,2,0))
                
            env.rgb_side = np.zeros((480, 640, 3), dtype=np.uint8)
            env.rgb_top = np.zeros((480, 640, 3), dtype=np.uint8)
            env.render()
            step += 1
    
    # Keep the viewer open briefly to show the completed episode
    timeout = time.time() + 0.5  # half second timeout between episodes
    while time.time() < timeout and env.env.is_viewer_alive():
        env.step_env()
    
    return env.env.is_viewer_alive()


def sequential_visualization(env, datasets, action_types):
    """
    Sequentially visualize all action types and episodes without user interaction.
    Automatically cycles through all action types and their episodes.
    """
    current_action_idx = 0
    
    while True:
        action_type = action_types[current_action_idx]
        dataset = datasets[action_type]
        
        # Visualize all episodes for the current action type
        for episode_idx in range(dataset.num_episodes):
            # Visualize the current episode
            viewer_alive = visualize_dataset(env, dataset, episode_idx, action_type)
            
            if not viewer_alive:
                return
                
        # Move to the next action type
        current_action_idx = (current_action_idx + 1) % len(action_types)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize robot demonstration data')
    parser.add_argument('--root', type=str, default='./demo_data_3', help='Path to demonstration data')
    parser.add_argument('--repo', type=str, default='omy_pnp', help='Repository name')
    parser.add_argument('--xml_path', type=str, default='./asset/example_scene_y.xml', help='Path to XML file')
    parser.add_argument('--action_type', type=str, choices=['joint', 'eef_pose', 'delta_q'], 
                        default='joint', help='Action type to visualize')
    args = parser.parse_args()
    
    # Action types to visualize
    all_action_types = ['joint', 'eef_pose', 'delta_q']
    action_types = all_action_types if args.action_type == 'all' else [args.action_type]
    
    # Load datasets for each action type
    datasets = {}
    for action_type in action_types:
        try:
            dataset = LeRobotDataset(args.repo, root=os.path.join(args.root, action_type))
            datasets[action_type] = dataset
            print(f"Loaded {action_type} dataset with {dataset.num_episodes} episodes")
        except Exception as e:
            print(f"Failed to load {action_type} dataset: {e}")
            print(f"Skipping {action_type} action type")
    
    if not datasets:
        print("No datasets could be loaded. Please check your data directory.")
        return
    
    # Import mujoco_env components here to avoid immediate import
    from mujoco_env.y_env import SimpleEnv
    
    # Initialize the environment
    if args.action_type == 'joint':
        env = SimpleEnv(args.xml_path, seed=0, state_type='joint_angle', action_type='joint_angle')
    elif args.action_type == 'eef_pose':
        env = SimpleEnv(args.xml_path, seed=0, state_type='joint_angle', action_type='eef_pose')
    elif args.action_type == 'delta_q':
        env = SimpleEnv(args.xml_path, seed=0, state_type='joint_angle', action_type='delta_joint_angle')
    else:
        raise ValueError(f"Invalid action type: {args.action_type}")
    
    # Start sequential visualization
    sequential_visualization(env, datasets, list(datasets.keys()))
    
    # Close the viewer
    env.env.close_viewer()


if __name__ == "__main__":
    main()
