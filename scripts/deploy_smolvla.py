#!/usr/bin/env python
"""
Deploy SmolVLA Policy

This script loads a trained SmolVLA policy and deploys it in a MuJoCo simulation environment.
The policy will perform the pick-and-place task with vision-language understanding.

Usage:
    python scripts/deploy_smolvla.py --ckpt_path ./ckpt/smolvla_omy/checkpoints/last/pretrained_model
    python scripts/deploy_smolvla.py --ckpt_path ./ckpt/smolvla_omy/checkpoints/last/pretrained_model --dataset_root ./demo_data_language
    python scripts/deploy_smolvla.py --hub_model Jeongeun/omy_pnp_smolvla --dataset_root ./omy_pnp_language
"""

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
import torchvision
from torchvision import transforms

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.configs.types import FeatureType
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.utils import dataset_to_policy_features
from mujoco_env.y_env2 import SimpleEnv2


def get_default_transform(image_size: int = 224):
    """
    Returns a torchvision transform that:
    Converts to a FloatTensor and scales pixel values [0,255] -> [0.0,1.0]
    """
    return transforms.Compose([
        transforms.ToTensor(),  # PIL [0–255] -> FloatTensor [0.0–1.0], shape C×H×W
    ])


def load_policy_and_metadata(ckpt_path=None, hub_model=None, dataset_root='./demo_data_language', device='cuda'):
    """
    Load SmolVLA policy and dataset metadata
    
    Args:
        ckpt_path: Path to local checkpoint
        hub_model: Hugging Face model name
        dataset_root: Root directory for dataset metadata
        device: Device to load the model on
        
    Returns:
        policy: Loaded SmolVLA policy
        dataset_metadata: Dataset metadata
    """
    # Load dataset metadata
    try:
        dataset_metadata = LeRobotDatasetMetadata("omy_pnp_language", root=dataset_root)
    except:
        # Try alternative path if the first one fails
        alt_root = './omy_pnp_language' if dataset_root == './demo_data_language' else './demo_data_language'
        print(f"Failed to load from {dataset_root}, trying {alt_root}")
        dataset_metadata = LeRobotDatasetMetadata("omy_pnp_language", root=alt_root)
    
    # Prepare features
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    
    # Create config
    cfg = SmolVLAConfig(
        input_features=input_features, 
        output_features=output_features, 
        chunk_size=5, 
        n_action_steps=5
    )
    delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)
    
    # Load policy
    if ckpt_path:
        print(f"Loading SmolVLA policy from local path: {ckpt_path}")
        policy = SmolVLAPolicy.from_pretrained(ckpt_path, dataset_stats=dataset_metadata.stats)
    elif hub_model:
        print(f"Loading SmolVLA policy from Hugging Face: {hub_model}")
        policy = SmolVLAPolicy.from_pretrained(hub_model, config=cfg, dataset_stats=dataset_metadata.stats)
    else:
        raise ValueError("Either ckpt_path or hub_model must be provided")
    
    policy.to(device)
    policy.eval()
    
    return policy, dataset_metadata


def deploy_policy(policy, xml_path='./asset/example_scene_y2.xml', device='cuda', max_episodes=10, control_hz=20):
    """
    Deploy the SmolVLA policy in the environment
    
    Args:
        policy: Loaded SmolVLA policy
        xml_path: Path to MuJoCo XML file
        device: Device the policy is on
        max_episodes: Maximum number of episodes to run
        control_hz: Control frequency in Hz
    """
    print(f"Deploying SmolVLA policy in MuJoCo environment...")
    print(f"XML path: {xml_path}")
    print(f"Device: {device}")
    print(f"Control frequency: {control_hz} Hz")
    
    # Initialize environment
    PnPEnv = SimpleEnv2(xml_path, action_type='joint_angle')
    IMG_TRANSFORM = get_default_transform()
    
    episode_count = 0
    
    try:
        while PnPEnv.env.is_viewer_alive() and episode_count < max_episodes:
            # Reset environment and policy for new episode
            step = 0
            PnPEnv.reset(seed=episode_count)
            policy.reset()
            print(f"\n=== Episode {episode_count + 1}/{max_episodes} ===")
            
            episode_success = False
            max_steps_per_episode = 1000  # Prevent infinite loops
            
            while PnPEnv.env.is_viewer_alive() and step < max_steps_per_episode:
                PnPEnv.step_env()
                
                if PnPEnv.env.loop_every(HZ=control_hz):
                    # Check if the task is completed
                    success = PnPEnv.check_success()
                    if success:
                        print(f'Success at step {step}!')
                        episode_success = True
                        break
                    
                    # Get the current state of the environment
                    state = PnPEnv.get_joint_state()[:6]
                    
                    # Get the current image from the environment
                    image, wrist_image = PnPEnv.grab_image()
                    
                    # Process main camera image
                    image = Image.fromarray(image)
                    image = image.resize((256, 256))
                    image = IMG_TRANSFORM(image)
                    
                    # Process wrist camera image
                    wrist_image = Image.fromarray(wrist_image)
                    wrist_image = wrist_image.resize((256, 256))
                    wrist_image = IMG_TRANSFORM(wrist_image)
                    
                    # Prepare input data
                    data = {
                        'observation.state': torch.tensor([state], dtype=torch.float32).to(device),
                        'observation.image': image.unsqueeze(0).to(device),
                        'observation.wrist_image': wrist_image.unsqueeze(0).to(device),
                        'task': [PnPEnv.instruction],
                    }
                    
                    # Select an action using the policy
                    with torch.no_grad():
                        action = policy.select_action(data)
                        action = action[0, :7].cpu().detach().numpy()
                    
                    # Take a step in the environment
                    _ = PnPEnv.step(action)
                    PnPEnv.render()
                    step += 1
            
            if episode_success:
                print(f"Episode {episode_count + 1} completed successfully in {step} steps!")
            else:
                print(f"Episode {episode_count + 1} did not complete successfully (max steps reached)")
            
            episode_count += 1
            
            # Small delay before next episode
            if episode_count < max_episodes:
                print("Press any key to continue to next episode, or close the viewer to exit...")
                # Wait a moment for user to see the result
                import time
                time.sleep(2)
    
    except KeyboardInterrupt:
        print("\nDeployment interrupted by user")
    except Exception as e:
        print(f"Error during deployment: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Deployment finished")


def main():
    parser = argparse.ArgumentParser(description='Deploy SmolVLA policy in MuJoCo environment')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='Path to local checkpoint (e.g., ./ckpt/smolvla_omy/checkpoints/last/pretrained_model)')
    parser.add_argument('--hub_model', type=str, default=None,
                        help='Hugging Face model name (e.g., Jeongeun/omy_pnp_smolvla)')
    parser.add_argument('--dataset_root', type=str, default='./demo_data_language',
                        help='Root directory for dataset metadata')
    parser.add_argument('--xml_path', type=str, default='./asset/example_scene_y2.xml',
                        help='Path to MuJoCo XML file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run the model on (cuda/cpu)')
    parser.add_argument('--max_episodes', type=int, default=10,
                        help='Maximum number of episodes to run')
    parser.add_argument('--control_hz', type=int, default=20,
                        help='Control frequency in Hz')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.ckpt_path and not args.hub_model:
        print("Error: Either --ckpt_path or --hub_model must be provided")
        parser.print_help()
        return
    
    if args.ckpt_path and not os.path.exists(args.ckpt_path):
        print(f"Error: Checkpoint path does not exist: {args.ckpt_path}")
        return
    
    if not os.path.exists(args.xml_path):
        print(f"Error: XML file does not exist: {args.xml_path}")
        return
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = 'cpu'
    
    print(f"\n=== SmolVLA Policy Deployment ===")
    print(f"Checkpoint path: {args.ckpt_path}")
    print(f"Hub model: {args.hub_model}")
    print(f"Dataset root: {args.dataset_root}")
    print(f"XML path: {args.xml_path}")
    print(f"Device: {args.device}")
    print(f"Max episodes: {args.max_episodes}")
    print(f"Control frequency: {args.control_hz} Hz")
    
    try:
        # Load policy and metadata
        policy, dataset_metadata = load_policy_and_metadata(
            ckpt_path=args.ckpt_path,
            hub_model=args.hub_model,
            dataset_root=args.dataset_root,
            device=args.device
        )
        
        print(f"Policy loaded successfully!")
        print(f"Dataset: {dataset_metadata.repo_id}")
        print(f"Number of episodes: {dataset_metadata.total_episodes}")
        print(f"Number of frames: {dataset_metadata.total_frames}")
        
        # Deploy policy
        deploy_policy(
            policy=policy,
            xml_path=args.xml_path,
            device=args.device,
            max_episodes=args.max_episodes,
            control_hz=args.control_hz
        )
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 