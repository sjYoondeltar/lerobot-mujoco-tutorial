#!/usr/bin/env python
# coding: utf-8

"""
Deploy your Diffusion Policy

This script loads a trained Diffusion Policy model and deploys it in a MuJoCo simulation environment.
The model will control a robot to perform the pick and place task that it was trained on.
"""

import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
# Import mujoco_env components inside the functions

def load_policy(ckpt_dir):
    """Load a trained Diffusion Policy from checkpoint"""
    if not os.path.exists(ckpt_dir):
        print(f"Checkpoint directory {ckpt_dir} does not exist.")
        print("Please train the model first or download a pre-trained model.")
        return None
    
    # Load the diffusion policy from the checkpoint
    policy = DiffusionPolicy.from_pretrained(ckpt_dir)
    print(f"Diffusion Policy loaded from {ckpt_dir}")
    
    return policy


def deploy_policy(env, policy, max_steps=1000):
    """Deploy the policy in the environment"""
    # Reset the environment
    env.reset()
    policy.reset()  # Reset policy state (important for diffusion policy)
    
    # Get initial observation
    obs = env.get_ee_pose()
    
    # Prepare observation for the policy
    obs_dict = {
        "observation.state": torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    }
    
    step_count = 0
    done = False
    
    while env.env.is_viewer_alive() and step_count < max_steps and not done:
        env.step_env()
        
        if env.env.loop_every(HZ=20):
            # Get action from policy
            with torch.no_grad():
                action = policy.select_action(obs_dict).squeeze().cpu().numpy()
            
            # Apply action to environment
            env.step(action)
            
            # Update observation
            obs = env.get_ee_pose()
            obs_dict["observation.state"] = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            
            # Check if task is completed
            done = env.check_success()
            if done:
                print("Task completed successfully!")
            
            # Render the environment
            env.render()
            
            step_count += 1
    
    # Close the viewer
    env.env.close_viewer()
    
    return done


def main():
    # Configuration
    XML_PATH = './asset/example_scene_y.xml'
    CKPT_DIR = "./ckpt/diffusion_y"  # Path to saved Diffusion Policy checkpoints
    
    # Import mujoco_env components here to avoid immediate import
    from mujoco_env.y_env import SimpleEnv
    
    # Initialize the environment
    env = SimpleEnv(XML_PATH, seed=0, state_type='joint_angle')
    
    # Load policy
    policy = load_policy(CKPT_DIR)
    if policy is None:
        return
    
    # Deploy policy
    print("Deploying Diffusion Policy in simulation...")
    success = deploy_policy(env, policy)
    
    if not success:
        print("Task was not completed within the time limit.")


if __name__ == "__main__":
    main()
