#!/usr/bin/env python
# coding: utf-8

"""
Deploy Temporal Difference Model Predictive Control (TD-MPC) Policy

This script loads a trained TD-MPC policy and executes it in the MuJoCo environment.
"""

import os
import sys
import argparse
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms # Import transforms

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import TD-MPC specific components and environment
from lerobot.common.policies.tdmpc.modeling_tdmpc import TDMPCPolicy
from lerobot.common.policies.tdmpc.configuration_tdmpc import TDMPCConfig
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.utils import dataset_to_policy_features

# Import the custom environment
try:
    from mujoco_env.y_env import SimpleEnv
    from mujoco_env.transforms import r2rpy # Assuming r2rpy might be needed depending on state_type
except ImportError:
    print("Error: Could not import SimpleEnv or transforms from mujoco_env.")
    print("Please ensure mujoco_env is in the Python path or installed correctly.")
    sys.exit(1)


def load_policy_for_deployment(ckpt_dir, action_type='joint', root_dir='./demo_data_4'):
    """
    Load a trained TD-MPC policy from checkpoint for deployment.

    Args:
        ckpt_dir: Base directory containing the policy checkpoints.
                  Expects a subdirectory named after action_type (e.g., ckpt_dir/joint/final).
        action_type: Type of action the policy was trained with ('joint', 'eef_pose', or 'delta_q').
        root_dir: Root directory of the dataset used for training (to load metadata/stats).

    Returns:
        Loaded policy object.
    """
    dataset_root = os.path.join(root_dir, action_type)
    if not os.path.exists(dataset_root):
         raise FileNotFoundError(f"Dataset directory not found at {dataset_root}. Needed for metadata.")

    print(f"Loading dataset metadata from: {dataset_root}")
    dataset_metadata = LeRobotDatasetMetadata("omy_pnp", root=dataset_root) # Assuming dataset name
    features = dataset_to_policy_features(dataset_metadata.features)

    # --- Feature Selection Logic (mirrors training script) ---
    print("Original features from metadata:", list(features.keys()))
    output_features = {k: v for k, v in features.items() if k == "action" and v.type is FeatureType.ACTION}
    if not output_features:
        raise ValueError(f"No ACTION feature found for action_type '{action_type}' in metadata.")

    input_features = {key: ft for key, ft in features.items() if ft.type is not FeatureType.ACTION}

    visual_features = {k: v for k, v in input_features.items() if v.type is FeatureType.VISUAL}
    non_visual_inputs = {k: v for k, v in input_features.items() if v.type is not FeatureType.VISUAL}

    exclude_key = 'observation.wrist_image' # Key to exclude if other images exist
    filtered_visual_features = {k: v for k, v in visual_features.items() if k != exclude_key}

    selected_visual_feature = None
    policy_input_features = {}

    if len(filtered_visual_features) > 0:
        selected_key = next(iter(filtered_visual_features))
        selected_visual_feature = {selected_key: filtered_visual_features[selected_key]}
        print(f"Selected visual feature for policy: {selected_key}")
        policy_input_features = {**non_visual_inputs, **selected_visual_feature}
    elif exclude_key in visual_features:
        selected_key = exclude_key
        selected_visual_feature = {selected_key: visual_features[selected_key]}
        print(f"Only '{exclude_key}' found. Using it as fallback.")
        policy_input_features = {**non_visual_inputs, **selected_visual_feature}
    else:
        print("WARNING: No visual features found. Policy will use only non-visual inputs.")
        policy_input_features = non_visual_inputs

    print("Final input features expected by the policy:")
    for k, v in policy_input_features.items():
        print(f"  - {k}: type={v.type if hasattr(v, 'type') else 'None'}, shape={v.shape if hasattr(v, 'shape') else 'None'}")
    # --- End Feature Selection ---

    # Construct the specific checkpoint path (assuming a 'final' subdirectory)
    policy_ckpt_path = os.path.join(ckpt_dir, action_type, 'final')
    if not os.path.isdir(policy_ckpt_path):
        # Fallback: try loading directly from action_type_ckpt_dir if 'final' doesn't exist
        policy_ckpt_path = os.path.join(ckpt_dir, action_type)
        if not os.path.isdir(policy_ckpt_path):
             raise FileNotFoundError(f"Checkpoint directory not found at {os.path.join(ckpt_dir, action_type, 'final')} or {policy_ckpt_path}")

    print(f"Loading TD-MPC policy from: {policy_ckpt_path}")
    # TDMPCPolicy.from_pretrained automatically loads the config
    try:
         policy = TDMPCPolicy.from_pretrained(policy_ckpt_path)
         print("Policy loaded successfully.")
         # Verify loaded config matches expectations (optional but recommended)
         # loaded_input_keys = set(policy.config.input_features.keys())
         # expected_input_keys = set(policy_input_features.keys())
         # if loaded_input_keys != expected_input_keys:
         #      print(f"Warning: Input features mismatch! Loaded: {loaded_input_keys}, Expected based on metadata: {expected_input_keys}")
    except Exception as e:
         print(f"Error loading policy from {policy_ckpt_path}: {e}")
         import traceback
         traceback.print_exc()
         raise

    return policy, policy_input_features # Return policy and the features it expects

def get_observation(env: SimpleEnv, policy_input_features: dict, device):
    """
    Get observation from the environment and format it for the policy.

    Args:
        env: The SimpleEnv instance.
        policy_input_features: Dictionary describing the features the policy expects.
        device: Torch device ('cuda' or 'cpu').

    Returns:
        A dictionary containing the formatted observation tensors on the specified device.
    """
    obs = {}
    # 1. Get raw data from environment
    # Example: Joint state (assuming 'observation.state' is used)
    if 'observation.state' in policy_input_features:
        # SimpleEnv's get_joint_state returns [j1..j6, gripper], policy might expect only joints
        # Adjust based on how 'observation.state' was defined during training data collection
        joint_state = env.get_joint_state()
        # Assuming the policy expects the 7-dim state [joints + gripper]
        obs['observation.state'] = torch.tensor(joint_state, dtype=torch.float32).unsqueeze(0) # Add batch dim

    # 2. Get image data if needed by policy
    image_keys = [k for k, v in policy_input_features.items() if v.type == FeatureType.VISUAL]
    if image_keys:
        # Assuming only one image key is in policy_input_features after filtering
        image_key = image_keys[0]
        rgb_agent, rgb_ego = env.grab_image() # Get both images

        if image_key == 'observation.image': # Corresponds to agent view in SimpleEnv
            image_data = rgb_agent
            print("Using observation.image (agent view)")
        elif image_key == 'observation.wrist_image': # Corresponds to egocentric view
            image_data = rgb_ego
            print("Using observation.wrist_image (ego view)")
        else:
            raise ValueError(f"Policy expects visual feature '{image_key}', but SimpleEnv mapping is unclear.")

        # Preprocess image (needs to match training transforms, especially ToTensor and Normalize if used)
        # Note: Training script uses ColorJitter, GaussianBlur, RandomErasing - these are augmentations
        # and typically NOT applied during deployment. We likely only need ToTensor and potentially normalization.
        # Let's assume normalization used dataset stats. We might need to load stats or use standard ones.
        # For now, just convert to tensor. Add normalization if policy requires it.
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            # Add Normalize here if the policy's visual encoder expects normalized images
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Example ImageNet stats
        ])
        image_tensor = preprocess(image_data).unsqueeze(0) # Add batch dim
        obs[image_key] = image_tensor
    else:
        print("Policy does not require visual input.")


    # 3. Add other features if necessary (e.g., proprioception like ee_pose)
    if 'observation.ee_pose' in policy_input_features:
         ee_pose = env.get_ee_pose() # Assuming SimpleEnv has this or similar method
         obs['observation.ee_pose'] = torch.tensor(ee_pose, dtype=torch.float32).unsqueeze(0)

    # Move all tensors to the target device
    obs_batch = {k: v.to(device) for k, v in obs.items()}
    return obs_batch


def main():
    parser = argparse.ArgumentParser(description='Deploy a trained TD-MPC policy.')
    parser.add_argument('--ckpt_dir', type=str, required=True, help='Path to the base checkpoint directory (containing action_type subdirs).')
    parser.add_argument('--action_type', type=str, choices=['joint', 'eef_pose', 'delta_q'], default='joint',
                        help='Action type the policy was trained with.')
    parser.add_argument('--data_root', type=str, default='./demo_data_4', help='Path to the dataset root (for metadata/stats).')
    parser.add_argument('--env_mode', type=str, choices=['easy', 'complex'], default='easy', help='Environment mode for object placement.')
    parser.add_argument('--xml_path', type=str, default='mujoco_env/assets/y.xml', help='Path to the MuJoCo XML file.')
    parser.add_argument('--max_steps', type=int, default=500, help='Maximum number of steps to run the policy.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for environment reset.')
    parser.add_argument('--render', action='store_true', help='Render the environment.')

    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Policy
    try:
        policy, policy_input_features = load_policy_for_deployment(args.ckpt_dir, args.action_type, args.data_root)
        policy.eval() # Set policy to evaluation mode
        policy.to(device)
    except Exception as e:
        print(f"Failed to load policy: {e}")
        sys.exit(1)

    # Initialize Environment
    print(f"Initializing environment with action_type='{args.action_type}', mode='{args.env_mode}'")
    env = SimpleEnv(xml_path=args.xml_path, action_type=args.action_type, mode=args.env_mode)
    env.reset(seed=args.seed)

    # Deployment Loop
    print("Starting deployment loop...")
    policy.reset() # Reset policy internal state if any (e.g., for recurrent layers)
    start_time = time.time()
    steps_taken = 0

    for step in range(args.max_steps):
        try:
            # 1. Get Observation
            obs_batch = get_observation(env, policy_input_features, device)

            # 2. Select Action
            with torch.no_grad():
                action = policy.select_action(obs_batch)

            # Convert action to numpy for the environment
            action_np = action.cpu().numpy().squeeze(0) # Remove batch dim

            # 3. Step Environment
            # env.step applies the action logic (IK, delta calculation, etc.)
            # and updates the internal target q.
            # It does NOT step the physics simulation.
            state_info = env.step(action_np) # state_info depends on env.state_type

            # env.step_env() applies the computed self.q to the simulator and steps physics
            env.step_env()

            steps_taken += 1

            # 4. Render
            if args.render:
                env.render(teleop=False) # Render without teleop overlays

            # 5. Check Success (optional)
            if env.check_success():
                print(f"Success condition met at step {step}!")
                break

            # Optional: Add a small delay if running too fast
            # time.sleep(0.01)

        except KeyboardInterrupt:
            print("\nDeployment interrupted by user.")
            break
        except Exception as e:
            print(f"\nError during deployment loop at step {step}: {e}")
            import traceback
            traceback.print_exc()
            break

    end_time = time.time()
    duration = end_time - start_time
    fps = steps_taken / duration if duration > 0 else float('inf')
    print(f"\nDeployment finished after {steps_taken} steps.")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Average FPS: {fps:.2f}")

    # Close viewer properly
    if hasattr(env.env, 'viewer') and env.env.viewer is not None:
        env.env.close_viewer()
    print("Environment viewer closed.")


if __name__ == "__main__":
    main() 