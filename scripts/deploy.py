#!/usr/bin/env python
# coding: utf-8

"""
Deploy ACT, Diffusion, or VQBeT Policy

This script loads a trained ACT, Diffusion, or VQBeT policy model and deploys it in a MuJoCo 
simulation environment. The model will control a robot to perform the pick and 
place task that it was trained on. Select the policy type using the --policy_type argument.

Available action types:
- 'joint': Deploy model trained with joint angles (action.joint)
- 'ee_pose': Deploy model trained with end-effector pose (action.ee_pose)
- 'delta_q': Deploy model trained with delta joint angles (action.delta_q)
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
from lerobot.common.policies.vqbet.configuration_vqbet import VQBeTConfig
from lerobot.common.policies.vqbet.modeling_vqbet import VQBeTPolicy
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot.common.datasets.utils import dataset_to_policy_features, write_json, serialize_dict
# Import mujoco_env components inside the functions

# Global device setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_policy(policy_type, ckpt_dir, action_type='joint'):
    """
    Load a trained ACT, Diffusion, or VQBeT Policy from checkpoint
    
    Args:
        policy_type: Type of policy ('act', 'diffusion', or 'vqbet')
        ckpt_dir: Path to the checkpoint directory
        action_type: Type of action ('joint', 'ee_pose', or 'delta_q')
    """
    # Adjust the checkpoint directory to include the action type for ACT policy
    if policy_type == 'act':
        action_type_ckpt_dir = os.path.join(ckpt_dir, action_type)
        if os.path.exists(action_type_ckpt_dir):
            ckpt_dir = action_type_ckpt_dir
            print(f"Using action-specific checkpoint: {ckpt_dir}")
        else:
            print(f"Warning: Action-specific checkpoint directory {action_type_ckpt_dir} not found.")
            print(f"Falling back to: {ckpt_dir}")
    
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
        
        # Filter output features based on selected action type if needed
        if policy_type == 'act':
            if action_type == 'joint':
                output_features = {k: v for k, v in features.items() if k == "action.joint" and v.type is FeatureType.ACTION}
            elif action_type == 'ee_pose':
                output_features = {k: v for k, v in features.items() if k == "action.ee_pose" and v.type is FeatureType.ACTION}
            elif action_type == 'delta_q':
                output_features = {k: v for k, v in features.items() if k == "action.delta_q" and v.type is FeatureType.ACTION}
            else:
                print(f"Unknown action type: {action_type}. Using all action features.")
                output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
            
            # 키 값 출력해서 디버깅
            print(f"Available features: {list(features.keys())}")
            print(f"Selected output features for {action_type}: {list(output_features.keys())}")
            
            # 출력 특성이 비어있는지 확인
            if not output_features:
                print(f"WARNING: No output features found for action_type '{action_type}'")
                print("Available feature types in dataset:")
                for k, v in features.items():
                    print(f"  - {k}: type={v.type}, shape={v.shape}")
                
                # 폴백 솔루션: 전체 액션 타입 사용
                print("Falling back to using all action features")
                output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
        else:
            # For non-ACT policies, use all action features
            output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
        
        input_features = {key: ft for key, ft in features.items() if key not in output_features}
        
        # Check if config.json exists in the checkpoint directory to determine if wrist_image is used
        config_json_path = os.path.join(ckpt_dir, "config.json")
        use_wrist_image = True  # Default to including wrist image
        
        if os.path.exists(config_json_path):
            try:
                import json
                with open(config_json_path, 'r') as f:
                    config_data = json.load(f)
                
                # Check if wrist_image was used during training
                if 'input_features' in config_data:
                    use_wrist_image = 'observation.wrist_image' in config_data['input_features']
                    print(f"Based on checkpoint config, wrist_image usage: {'Enabled' if use_wrist_image else 'Disabled'}")
                else:
                    print("Config file does not contain input_features information. Using all features.")
            except Exception as e:
                print(f"Error reading config.json: {e}. Using all features.")
        else:
            print(f"No config.json found in {ckpt_dir}. Using all features.")
        
        # Remove wrist_image from input features if not used during training
        if not use_wrist_image and 'observation.wrist_image' in input_features:
            print("Removing observation.wrist_image from input features to match training configuration")
            input_features.pop('observation.wrist_image')
        
        policy = None
        cfg = None

        if policy_type == 'act':
            # Create ACT config
            print(f"Configuring ACT policy with action type: {action_type}...")
            cfg = ACTConfig(
                input_features=input_features, 
                output_features=output_features, 
                chunk_size=10, 
                n_action_steps=1, 
                temporal_ensemble_coeff=0.9 # Example ACT specific param
            )
            
            # Print config details for debugging
            print(f"ACT Config - input features: {list(cfg.input_features.keys())}")
            print(f"ACT Config - output features: {list(cfg.output_features.keys())}")
            if not cfg.output_features:
                print("ERROR: No output features in config. Cannot continue.")
                return None, None
                
            # Make sure action_feature is set
            feature_key = next(iter(cfg.output_features.keys()), None)
            if feature_key:
                cfg.action_feature = cfg.output_features[feature_key]
                print(f"Setting action_feature to {feature_key} with shape {cfg.action_feature.shape}")
            else:
                print("ERROR: Cannot set action_feature - no output features available")
                return None, None
            
            # Load the ACT policy from the checkpoint with config and dataset stats
            try:
                print(f"Loading ACT policy from {ckpt_dir}...")
                policy = ACTPolicy.from_pretrained(
                    ckpt_dir, 
                    config=cfg, 
                    dataset_stats=dataset_metadata.stats
                )
                print("Successfully loaded ACT policy")
            except Exception as e:
                print(f"Error loading ACT policy: {e}")
                print("Trying to initialize a new policy and load the state dict manually...")
                try:
                    policy = ACTPolicy(cfg, dataset_stats=dataset_metadata.stats)
                    weights_path = os.path.join(ckpt_dir, "pytorch_model.bin")
                    if os.path.exists(weights_path):
                        state_dict = torch.load(weights_path, map_location=DEVICE)
                        policy.load_state_dict(state_dict, strict=False)
                        print("Successfully loaded model weights manually")
                    else:
                        print(f"No weights file found at {weights_path}")
                        return None, None
                except Exception as e2:
                    print(f"Failed to initialize and load policy manually: {e2}")
                    return None, None

        elif policy_type == 'diffusion':
            # Configure Diffusion policy
            print("Configuring Diffusion policy...")
            cfg = DiffusionConfig(
                input_features=input_features, 
                output_features=output_features, 
                horizon=16,  # Match training parameter
                n_action_steps=1 # Match training parameter
            )

            # Load the trained policy with robust error handling
            print(f"Loading trained policy from checkpoint: {ckpt_dir}")
            try:
                policy = DiffusionPolicy.from_pretrained(
                    ckpt_dir,
                    config=cfg,
                    dataset_stats=dataset_metadata.stats,
                    local_files_only=True,
                    trust_remote_code=True 
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
        
        elif policy_type == 'vqbet':
            # Configure VQBeT policy
            print("Configuring VQBeT policy...")
            
            # VQBeT only supports one image input
            image_keys = [key for key in input_features.keys() if key.startswith("observation.") and key.endswith("image")]
            if len(image_keys) > 1:
                print(f"Found multiple image inputs: {image_keys}")
                print(f"VQBeT only supports one image input. Keeping 'observation.image' and removing others.")
                for key in image_keys:
                    if key != "observation.image":
                        input_features.pop(key, None)
            
            # Configure VQBeT
            cfg = VQBeTConfig(
                input_features=input_features, 
                output_features=output_features,
                n_obs_steps=5,
                n_action_pred_token=5,  # Updated to match training config 
                action_chunk_size=5,
                vqvae_n_embed=1024,  # codebook size
                vqvae_embedding_dim=128,  # latent dimension
                vision_backbone="resnet18",
                spatial_softmax_num_keypoints=32,
                gpt_n_layer=8,
                gpt_n_head=8
            )
            
            # First, create a policy instance
            policy = VQBeTPolicy(cfg, dataset_stats=dataset_metadata.stats)
            
            # Check for VQ-VAE specific checkpoint
            vqvae_ckpt_dir = os.path.join(ckpt_dir, "vqvae_only")
            vqvae_final_path = os.path.join(vqvae_ckpt_dir, "final_model.pt")
            
            # Try to load VQ-VAE weights if available
            vqvae_loaded = False
            if os.path.exists(vqvae_final_path):
                try:
                    print(f"Found VQ-VAE checkpoint at {vqvae_final_path}. Loading...")
                    vqvae_ckpt = torch.load(vqvae_final_path, map_location=DEVICE)
                    
                    # Extract just VQ-VAE related weights
                    vqvae_state_dict = {}
                    full_state_dict = vqvae_ckpt['model_state_dict']
                    
                    # Filter for VQ-VAE related parameters only
                    for name, param in full_state_dict.items():
                        if any(part in name for part in ['encoder', 'decoder', 'codebook', 'quantizer', 'vqvae']):
                            vqvae_state_dict[name] = param
                    
                    # Load the VQ-VAE weights (partial state dict)
                    missing_keys, unexpected_keys = policy.load_state_dict(vqvae_state_dict, strict=False)
                    print(f"Loaded VQ-VAE weights. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
                    vqvae_loaded = True
                except Exception as e:
                    print(f"Error loading VQ-VAE checkpoint: {e}")
                    print("Will attempt to load full model instead.")
            
            # Now try to load the full model (keeping VQ-VAE weights if they were loaded)
            try:
                # First try standard loading method with from_pretrained
                if not vqvae_loaded:
                    print("Loading full VQBeT model...")
                    policy = VQBeTPolicy.from_pretrained(
                        ckpt_dir,
                        config=cfg,
                        dataset_stats=dataset_metadata.stats
                    )
                else:
                    # If VQ-VAE was loaded, only load the remaining parts
                    print("Loading full VQBeT model weights on top of VQ-VAE weights...")
                    # Try different possible weight file locations
                    weights_path = os.path.join(ckpt_dir, "model.safetensors")
                    if not os.path.exists(weights_path):
                        weights_path = os.path.join(ckpt_dir, "pytorch_model.bin")
                        
                    if os.path.exists(weights_path):
                        if weights_path.endswith('.safetensors'):
                            from safetensors.torch import load_file
                            state_dict = load_file(weights_path, device=str(DEVICE))
                        else:
                            state_dict = torch.load(weights_path, map_location=DEVICE)
                        
                        # Load with strict=False to avoid errors with already loaded VQ-VAE weights
                        missing_keys, unexpected_keys = policy.load_state_dict(state_dict, strict=False)
                        print(f"Loaded full model weights. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
                    else:
                        print(f"Warning: Full model weights not found at {weights_path}")
                        if vqvae_loaded:
                            print("Proceeding with only VQ-VAE weights loaded.")
                        else:
                            print("Error: No weights loaded for VQBeT model.")
                            return None, None
            except Exception as e:
                print(f"Error loading full VQBeT model: {e}")
                
                # If VQ-VAE is already loaded, we might still be able to proceed
                if not vqvae_loaded:
                    print("Trying alternate loading method...")
                    try:
                        weights_path = os.path.join(ckpt_dir, "model.safetensors")
                        if not os.path.exists(weights_path):
                            weights_path = os.path.join(ckpt_dir, "pytorch_model.bin")
                            
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
                            print(f"Error: Weights file not found in {ckpt_dir}.")
                            return None, None
                    except Exception as e2:
                        print(f"Error with alternate loading method: {e2}")
                        return None, None
                else:
                    print("Proceeding with only VQ-VAE weights loaded, as full model loading failed.")
        
        else:
            print(f"Error: Unknown policy type '{policy_type}'. Choose 'act', 'diffusion', or 'vqbet'.")
            return None, None

        # Move policy to the correct device and set to evaluation mode
        policy.to(DEVICE)
        policy.eval()  
        print(f"Successfully loaded {policy_type.upper()} policy with action type {action_type}")
        return policy, cfg

    except Exception as e:
        print(f"An error occurred while loading policy: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def deploy_policy(learning_env, policy, policy_type, action_type='joint', max_steps=1000, control_hz=20):
    """
    Deploy a trained policy in a learning environment
    
    Args:
        learning_env: MuJoCo environment
        policy: Trained policy model
        policy_type: Type of policy ('act', 'diffusion', 'vqbet')
        action_type: Type of action ('joint', 'ee_pose', 'delta_q')
        max_steps: Maximum number of steps to run
        control_hz: Control frequency in Hz
    """
    print(f"Starting deployment with {policy_type.upper()} policy using {action_type} action type")
    
    # Set the environment action type according to model's action type
    if hasattr(learning_env, 'action_type') and hasattr(learning_env, 'state_type'):
        # For SimpleEnv with action_type attribute, set it based on the loaded model
        if action_type == 'joint':
            learning_env.action_type = 'joint_angle'
            print("Using joint_angle action type for environment")
        elif action_type == 'ee_pose':
            learning_env.action_type = 'eef_pose'
            print("Using eef_pose action type for environment")
        elif action_type == 'delta_q':
            learning_env.action_type = 'delta_joint_angle'
            print("Using delta_joint_angle action type for environment")
        
        # Set state type to match action type for best results
        learning_env.state_type = 'ee_pose' if action_type == 'ee_pose' else 'joint_angle'
        print(f"Using {learning_env.state_type} state type for environment")
    
    # Setup image transformations
    img_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Check if the policy uses wrist image
    uses_wrist_image = 'observation.wrist_image' in policy.config.input_features if hasattr(policy, 'config') else False
    
    step_count = 0
    done = False
    
    try:
        print("Starting simulation...")
        # Initial reset
        learning_env.reset(seed=0)
        policy.reset()
        
        # Main simulation loop
        while not done and step_count < max_steps and learning_env.env.is_viewer_alive():
            learning_env.step_env()
            
            if learning_env.env.loop_every(HZ=control_hz):
                # Get state and images
                state = learning_env.get_ee_pose()
                # print(f"Step: {step_count}, State: {state}") # Verbose logging
                image, wrist_image = learning_env.grab_image() 
                
                # Preprocess main camera image
                image_pil = Image.fromarray(image).resize((256, 256))
                image_tensor = img_transform(image_pil).unsqueeze(0).to(DEVICE)
                
                # Prepare input data dictionary
                data = {
                    'observation.state': torch.tensor([state], dtype=torch.float32).to(DEVICE),
                    'observation.image': image_tensor,
                    'task': ['Put mug cup on the plate'], # Optional, depends if model uses it
                    'timestamp': torch.tensor([step_count / control_hz], dtype=torch.float32).to(DEVICE) # Optional
                }
                
                # Only include wrist camera image if the policy uses it
                if uses_wrist_image:
                    wrist_image_pil = Image.fromarray(wrist_image).resize((256, 256))
                    wrist_image_tensor = img_transform(wrist_image_pil).unsqueeze(0).to(DEVICE)
                    data['observation.wrist_image'] = wrist_image_tensor
                
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
                            action = action_output[0].cpu().numpy()
                        else:
                             # Handle unexpected output format
                             print(f"Warning: Unexpected diffusion action output format: {type(action_output)}, shape: {action_output.shape if hasattr(action_output, 'shape') else 'N/A'}")
                             # Fallback or error handling needed here - e.g., use zeros
                             action_dim = policy.config.output_features["action.joint_angle"].shape[-1] # Infer action dim
                             action = np.zeros(action_dim)
                    elif policy_type == 'vqbet':
                        # More robust handling for VQBeT outputs
                        if isinstance(action_output, torch.Tensor):
                            # Try to use policy's built-in methods first
                            if hasattr(policy, 'detokenize_action'):
                                print(f"Detokenizing action output: {action_output}")
                                action = policy.detokenize_action(action_output)
                                action = action[0].cpu().numpy()  # Take first action
                            else:
                                # Direct handling as fallback
                                print(f"Direct handling action output: {action_output}")
                                action = action_output.cpu().numpy()
                                if action.ndim > 1:
                                    action = action[0]  # Take first action
                        else:
                            # Handle unexpected output format
                            print(f"Warning: Unexpected VQBeT action output format: {type(action_output)}")
                            action_dim = 7  # Hardcoded for robot joint angles
                            action = np.zeros(action_dim)

                if action is None:
                     print("Error: Failed to determine action from policy output.")
                     # Handle error: maybe stop, or use a default action
                     action_dim = 7  # Standard dimension for robot actions
                     action = np.zeros(action_dim) # Example: use zero action

                # Ensure action is a flat numpy array
                action = np.asarray(action).flatten()
                # print(f"Step: {step_count}, Action: {action}") # Verbose logging

                # Apply action to environment
                _ = learning_env.step(action)

                # Render
                learning_env.render()
                
                # Check for success
                done = learning_env.check_success()
                if done:
                    print(f"Task completed successfully at step {step_count}!")
                    policy.reset()
                    learning_env.reset(seed=0)
                    step_count = 0
                    break
                
                step_count += 1
                
        if not done and step_count >= max_steps:
            print("Task not completed: Reached maximum step limit.")
            
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
    parser = argparse.ArgumentParser(description="Deploy a trained policy (ACT, Diffusion, or VQBeT) in a MuJoCo environment.")
    parser.add_argument(
        "--policy_type", 
        type=str, 
        required=True, 
        choices=['act', 'diffusion', 'vqbet'], 
        help="Type of policy to load ('act', 'diffusion', or 'vqbet')."
    )
    parser.add_argument(
        "--action_type", 
        type=str, 
        default='joint', 
        choices=['joint', 'ee_pose', 'delta_q'], 
        help="Type of action used in the model ('joint', 'ee_pose', or 'delta_q')."
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
    policy, config = load_policy(args.policy_type, ckpt_dir, args.action_type)
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
        # Initialize with appropriate action_type based on the model's action type
        if args.action_type == 'joint':
            env_action_type = 'joint_angle'
        elif args.action_type == 'ee_pose':
            env_action_type = 'eef_pose'
        elif args.action_type == 'delta_q':
            env_action_type = 'delta_joint_angle'
        else:
            env_action_type = 'joint_angle'  # Default

        pnp_env = SimpleEnv(args.xml_path, seed=0, action_type=env_action_type)
        print(f"Environment initialized with action_type: {env_action_type}")
    except Exception as e:
         print(f"Error initializing environment: {e}")
         return
    
    # Deploy policy
    print(f"Deploying {args.policy_type.upper()} Policy with {args.action_type} action type in simulation...")
    success = deploy_policy(pnp_env, policy, args.policy_type, args.action_type, args.max_steps, args.control_hz)
    
    if success:
         print("Deployment finished: Success!")
    else:
         print("Deployment finished: Task not completed successfully.")


if __name__ == "__main__":
    main()

# Removed old In[] cells
