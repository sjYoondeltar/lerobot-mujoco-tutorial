#!/usr/bin/env python
# coding: utf-8

"""
Deploy ACT or Diffusion Policy

This script loads a trained ACT or Diffusion policy model and deploys it in a MuJoCo 
simulation environment. The model will control a robot to perform the pick and 
place task that it was trained on. Select the policy type using the --policy_type argument.
"""

import os
import sys
import argparse

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from PIL import Image
import torchvision
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot.common.datasets.utils import dataset_to_policy_features, write_json, serialize_dict
# Import mujoco_env components inside the functions

# Global device setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_policy(policy_type, ckpt_dir):
    """Load a trained ACT or Diffusion Policy from checkpoint"""
    print(f"Attempting to load {policy_type.upper()} policy from: {ckpt_dir}")
    if not os.path.exists(ckpt_dir):
        print(f"Error: Checkpoint directory {ckpt_dir} does not exist.")
        print("Please train the model first or download/specify a valid path.")
        return None, None

    try:
        # Get dataset metadata and prepare features (common for both)
        print("Loading dataset metadata...")
        dataset_metadata = LeRobotDatasetMetadata("omy_pnp", root='./demo_data')
        features = dataset_to_policy_features(dataset_metadata.features)
        output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
        input_features = {key: ft for key, ft in features.items() if key not in output_features}
        
        policy = None
        cfg = None

        if policy_type == 'act':
            # input_features.pop("observation.wrist_image") # Specific adjustment if needed for ACT
            
            # Create ACT config
            print("Configuring ACT policy...")
            cfg = ACTConfig(
                input_features=input_features, 
                output_features=output_features, 
                chunk_size=10, 
                n_action_steps=1, 
                temporal_ensemble_coeff=0.9 # Example ACT specific param
            )
            
            # Load the ACT policy from the checkpoint with config and dataset stats
            policy = ACTPolicy.from_pretrained(
                ckpt_dir, 
                config=cfg, 
                dataset_stats=dataset_metadata.stats
            )

        elif policy_type == 'diffusion':
            # input_features.pop("observation.wrist_image") # Specific adjustment if needed for Diffusion
            
            # Configure Diffusion policy
            print("Configuring Diffusion policy...")
            cfg = DiffusionConfig(
                input_features=input_features, 
                output_features=output_features, 
                horizon=16,  # Example Diffusion specific param (match training)
                n_action_steps=8 # Example Diffusion specific param (match training)
            )

            # Load the trained policy with robust error handling
            print(f"Loading trained policy from checkpoint: {ckpt_dir}")
            try:
                policy = DiffusionPolicy.from_pretrained(
                    ckpt_dir,
                    config=cfg,
                    dataset_stats=dataset_metadata.stats,
                    local_files_only=True,  # Important for local loading
                    trust_remote_code=True # Depending on setup
                )
            except Exception as e:
                print(f"Error loading Diffusion policy via from_pretrained: {e}")
                print("\nTrying alternate loading method...")
                try:
                    policy = DiffusionPolicy(cfg, dataset_stats=dataset_metadata.stats)
                    weights_path = os.path.join(ckpt_dir, "model.safetensors")
                    if not os.path.exists(weights_path):
                         weights_path = os.path.join(ckpt_dir, "pytorch_model.bin") # Common alternative name
                         if not os.path.exists(weights_path):
                             weights_path = os.path.join(ckpt_dir, "diffusion_pytorch_model.bin") # Another alternative

                    if os.path.exists(weights_path):
                        print(f"Loading weights from {weights_path}")
                        if weights_path.endswith('.safetensors'):
                            from safetensors.torch import load_file
                            state_dict = load_file(weights_path, device=str(DEVICE))
                        else:
                            state_dict = torch.load(weights_path, map_location=DEVICE)
                        
                        policy.load_state_dict(state_dict)
                        print("Successfully loaded model weights via alternate method.")
                    else:
                        print(f"Error: Weights file not found in {ckpt_dir}. Tried model.safetensors, pytorch_model.bin, diffusion_pytorch_model.bin")
                        return None, None
                except Exception as e2:
                    print(f"Error with alternate loading method: {e2}")
                    return None, None
        else:
            print(f"Error: Unknown policy type '{policy_type}'. Choose 'act' or 'diffusion'.")
            return None, None

        # Move policy to the correct device and set to evaluation mode
        policy.to(DEVICE)
        policy.eval()  
        print(f"{policy_type.upper()} Policy loaded successfully from {ckpt_dir} to {DEVICE}")
        return policy, cfg

    except Exception as e:
        print(f"An unexpected error occurred during policy loading: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def deploy_policy(learning_env, policy, policy_type, max_steps=1000, control_hz=20):
    """Deploy the loaded policy in the environment"""
    print(f"Deploying {policy_type.upper()} policy...")
    
    # Ensure model is on the correct device (redundant check, but safe)
    model_device = next(policy.parameters()).device
    print(f"Policy is on device: {model_device}")
    if model_device != DEVICE:
        print(f"Warning: Policy device ({model_device}) differs from target device ({DEVICE}). Moving policy...")
        policy.to(DEVICE)
    
    # Environment and policy reset
    learning_env.reset(seed=0)
    policy.reset() 
    
    # Image transform
    img_transform = torchvision.transforms.ToTensor()
    
    step_count = 0
    done = False
    
    try:
        while learning_env.env.is_viewer_alive() and step_count < max_steps and not done:
            # Step simulation physics (might not be needed depending on env step)
            # learning_env.step_env() # Assuming learning_env.step handles physics stepping if needed

            if learning_env.env.loop_every(HZ=control_hz):
                # Get state and images
                state = learning_env.get_ee_pose()
                # print(f"Step: {step_count}, State: {state}") # Verbose logging
                image, wrist_image = learning_env.grab_image() 
                
                # Preprocess images
                image_pil = Image.fromarray(image).resize((256, 256))
                image_tensor = img_transform(image_pil).unsqueeze(0).to(DEVICE)
                
                wrist_image_pil = Image.fromarray(wrist_image).resize((256, 256))
                wrist_image_tensor = img_transform(wrist_image_pil).unsqueeze(0).to(DEVICE)
                
                # Prepare input data dictionary
                data = {
                    'observation.state': torch.tensor([state], dtype=torch.float32).to(DEVICE),
                    'observation.image': image_tensor,
                    'observation.wrist_image': wrist_image_tensor,
                    # 'task': ['Put mug cup on the plate'], # Optional, depends if model uses it
                    # 'timestamp': torch.tensor([step_count / control_hz], dtype=torch.float32).to(DEVICE) # Optional
                }
                
                # Predict action
                action = None
                with torch.no_grad():
                    action_output = policy.select_action(data)
                    
                    if policy_type == 'act':
                        # ACT output is typically [batch_size, action_dim] or similar direct action
                        action = action_output[0].cpu().numpy() 
                    elif policy_type == 'diffusion':
                        # Diffusion output is typically [batch, horizon, action_dim]
                        # We take the first action step of the predicted horizon
                        if isinstance(action_output, torch.Tensor) and action_output.ndim >= 2:
                             # Take first step [0] from the first batch element [0]
                            action = action_output[0, 0].cpu().numpy() 
                        else:
                             # Handle unexpected output format
                             print(f"Warning: Unexpected diffusion action output format: {type(action_output)}, shape: {action_output.shape if hasattr(action_output, 'shape') else 'N/A'}")
                             # Fallback or error handling needed here - e.g., use zeros
                             action_dim = policy.config.output_features["action.joint_angle"].shape[-1] # Infer action dim
                             action = np.zeros(action_dim)

                if action is None:
                     print("Error: Failed to determine action from policy output.")
                     # Handle error: maybe stop, or use a default action
                     action_dim = policy.config.output_features["action.joint_angle"].shape[-1] # Infer action dim
                     action = np.zeros(action_dim) # Example: use zero action

                # Ensure action is a flat numpy array
                action = np.asarray(action).flatten()
                # print(f"Step: {step_count}, Action: {action}") # Verbose logging

                # Apply action to environment
                obs, reward, terminated, truncated, info = learning_env.step(action)
                done = terminated or truncated # Use standard Gymnasium termination flags

                # Render
                learning_env.render()
                
                # Check success condition (if available and separate from termination)
                if hasattr(learning_env, 'check_success') and learning_env.check_success():
                    print(f"Task completed successfully at step {step_count}!")
                    # Optional: reset on success if you want loops
                    # policy.reset()
                    # learning_env.reset(seed=0) 
                    # step_count = 0 
                    break # End loop on success
                
                step_count += 1
                
        if not done and step_count >= max_steps:
            print("Task not completed: Reached maximum step limit.")
        elif done and not (hasattr(learning_env, 'check_success') and learning_env.check_success()):
             print(f"Task ended (terminated={terminated}, truncated={truncated}) but success condition not met.")
            
    except Exception as e:
        print(f"An error occurred during deployment: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close viewer if it exists and has the method
        if hasattr(learning_env, 'env') and hasattr(learning_env.env, 'close_viewer'):
            learning_env.env.close_viewer()
        elif hasattr(learning_env, 'close'): # Handle environments that might have a close method directly
             try:
                 learning_env.close()
             except Exception as close_e:
                 print(f"Error closing environment: {close_e}")

        print("Simulation finished.")
    
    # Return success based on the environment's success check, if available
    return hasattr(learning_env, 'check_success') and learning_env.check_success()


def main():
    parser = argparse.ArgumentParser(description="Deploy a trained policy (ACT or Diffusion) in a MuJoCo environment.")
    parser.add_argument(
        "--policy_type", 
        type=str, 
        required=True, 
        choices=['act', 'diffusion'], 
        help="Type of policy to load ('act' or 'diffusion')."
    )
    parser.add_argument(
        "--ckpt_dir", 
        type=str, 
        default=None, 
        help="Path to the policy checkpoint directory. If None, uses default path based on policy_type."
    )
    parser.add_argument(
        "--xml_path", 
        type=str, 
        default='./asset/example_scene_y.xml', 
        help="Path to the MuJoCo XML scene file."
    )
    parser.add_argument(
        "--max_steps", 
        type=int, 
        default=1000, 
        help="Maximum number of simulation steps."
    )
    parser.add_argument(
        "--control_hz", 
        type=int, 
        default=20, 
        help="Control frequency in Hz."
    )
    
    args = parser.parse_args()

    # Determine checkpoint directory
    ckpt_dir = args.ckpt_dir
    if ckpt_dir is None:
        ckpt_dir = f"ckpt/{args.policy_type}_y" # Default path structure
        print(f"--ckpt_dir not specified, using default: {ckpt_dir}")

    # Load policy
    policy, config = load_policy(args.policy_type, ckpt_dir)
    if policy is None:
        print("Failed to load policy. Exiting.")
        return

    # Import mujoco_env components here to avoid potential issues if imports fail early
    try:
        from mujoco_env.y_env import SimpleEnv
    except ImportError as e:
        print(f"Error importing SimpleEnv from mujoco_env.y_env: {e}")
        print("Please ensure 'mujoco_env' is installed and accessible.")
        return
        
    # Initialize the environment
    print(f"Initializing environment from: {args.xml_path}")
    try:
        pnp_env = SimpleEnv(args.xml_path, seed=0, action_type='joint_angle')
        print("Environment initialized.")
    except Exception as e:
         print(f"Error initializing environment: {e}")
         return
    
    # Deploy policy
    print(f"Deploying {args.policy_type.upper()} Policy in simulation...")
    success = deploy_policy(pnp_env, policy, args.policy_type, args.max_steps, args.control_hz)
    
    if success:
         print("Deployment finished: Success!")
    else:
         print("Deployment finished: Task not completed successfully.")


if __name__ == "__main__":
    main()

# Removed old In[] cells
