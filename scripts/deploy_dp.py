#!/usr/bin/env python3
# 4.1.deploy_dp.py - Deploy Trained Diffusion Policy

import torch
import numpy as np
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.factory import resolve_delta_timestamps
from mujoco_env.y_env import SimpleEnv

def main():
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load dataset metadata and prepare features
    print("Loading dataset metadata...")
    dataset_metadata = LeRobotDatasetMetadata("omy_pnp", root='./demo_data')
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    input_features.pop("observation.wrist_image")

    # Configure and load the diffusion policy
    print("Configuring diffusion policy...")
    cfg = DiffusionConfig(
        input_features=input_features, 
        output_features=output_features, 
        horizon=8,  # Must match the training configuration
        n_action_steps=8
    )

    # Get delta timestamps for action chunking
    delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)

    # Load the trained policy
    ckpt_dir = os.path.abspath('./ckpt/diffusion_y')
    print(f"Loading trained policy from checkpoint: {ckpt_dir}")
    
    if not os.path.exists(ckpt_dir):
        print(f"Error: Checkpoint directory {ckpt_dir} does not exist.")
        print("Please train the model first using 3.1.train_dp.ipynb")
        return
        
    try:
        # List files in checkpoint directory
        print("Files in checkpoint directory:")
        for file in os.listdir(ckpt_dir):
            print(f"  - {file}")
            
        # Use the loading mechanism directly from the model class with local files
        policy = DiffusionPolicy.from_pretrained(
            ckpt_dir,
            config=cfg,
            dataset_stats=dataset_metadata.stats,
            local_files_only=True,  # Important for local loading
            trust_remote_code=True
        )
        policy.to(device)
        policy.eval()  # Set to evaluation mode
    except Exception as e:
        print(f"Error loading policy: {e}")
        print("\nTrying alternate loading method...")
        
        # Alternative loading method
        try:
            # Create a new policy with the configuration
            policy = DiffusionPolicy(cfg, dataset_stats=dataset_metadata.stats)
            
            # Check for model files
            weights_path = os.path.join(ckpt_dir, "model.safetensors")
            if not os.path.exists(weights_path):
                weights_path = os.path.join(ckpt_dir, "diffusion_pytorch_model.bin")
                
            if os.path.exists(weights_path):
                print(f"Loading weights from {weights_path}")
                if weights_path.endswith('.safetensors'):
                    from safetensors.torch import load_file
                    state_dict = load_file(weights_path)
                else:
                    state_dict = torch.load(weights_path, map_location=device)
                
                policy.load_state_dict(state_dict)
                print("Successfully loaded model weights")
            else:
                print(f"Error: Weights file not found. Tried: model.safetensors, diffusion_pytorch_model.bin")
                return
                
            policy.to(device)
            policy.eval()
        except Exception as e2:
            print(f"Error with alternate loading method: {e2}")
            return
    
    # Initialize the environment
    print("Initializing environment...")
    xml_path = './asset/example_scene_y.xml'
    PnPEnv = SimpleEnv(xml_path, action_type='joint_angle')
    
    # Run policy
    print("Starting policy execution...")
    run_policy(policy, PnPEnv, device)

def run_policy(policy, PnPEnv, device):
    # Initialize variables
    step = 0
    PnPEnv.reset(seed=0)
    policy.reset()
    save_image = True
    img_transform = torchvision.transforms.ToTensor()
    action_trajectory = []  # Store actions for visualization

    print("Running policy. Press Ctrl+C to stop...")
    
    try:
        # 첫 번째 실행에서 출력 형태를 확인하기 위한 플래그
        check_shape = True
        
        while PnPEnv.env.is_viewer_alive():
            PnPEnv.step_env()
            if PnPEnv.env.loop_every(HZ=20):
                # Check for task completion
                success = PnPEnv.check_success()
                if success:
                    print('Success!')
                    # Reset environment and policy
                    policy.reset()
                    PnPEnv.reset(seed=0)
                    step = 0
                    save_image = False
                    action_trajectory = []
                    continue
                
                # Get current state and images
                state = PnPEnv.get_ee_pose()
                image, wrist_image = PnPEnv.grab_image()
                
                # Process the image
                image = Image.fromarray(image)
                image = image.resize((256, 256))
                image = img_transform(image)
                
                # Prepare input data for the policy
                data = {
                    'observation.state': torch.tensor([state], dtype=torch.float32).to(device),
                    'observation.image': image.unsqueeze(0).to(device),
                    'task': ['Put mug cup on the plate'],
                    'timestamp': torch.tensor([step/20], dtype=torch.float32).to(device)
                }
                
                # Get action from policy
                with torch.no_grad():  # Disable gradient computation for inference
                    action_traj = policy.select_action(data)
                    
                    # 모델 출력 형태 확인
                    if check_shape:
                        print(f"Model output type: {type(action_traj)}")
                        print(f"Model output shape: {action_traj.shape if hasattr(action_traj, 'shape') else 'No shape attribute'}")
                        if isinstance(action_traj, torch.Tensor) and len(action_traj.shape) > 0:
                            print(f"First dimension content shape: {action_traj[0].shape if len(action_traj.shape) > 1 else 'Not applicable'}")
                        if isinstance(action_traj, (list, tuple)):
                            print(f"First element type: {type(action_traj[0])}")
                            print(f"First element shape: {action_traj[0].shape if hasattr(action_traj[0], 'shape') else 'No shape attribute'}")
                        check_shape = False
                    
                    # 출력 형태에 따라 적절히 처리
                    if isinstance(action_traj, torch.Tensor):
                        if len(action_traj.shape) >= 3:  # [batch, horizon, action_dim]
                            action = action_traj[0, 0].cpu().detach().numpy()
                        elif len(action_traj.shape) == 2:  # [batch, action_dim]
                            action = action_traj[0].cpu().detach().numpy()
                        elif len(action_traj.shape) == 1:  # [action_dim]
                            action = action_traj.cpu().detach().numpy()
                        else:  # 스칼라
                            # 환경에 맞는 기본 액션 크기 생성 (예: 7차원)
                            action = np.zeros(7)  # 7은 예상되는 액션 차원
                            action[0] = action_traj.cpu().detach().numpy()  # 첫 번째 차원에 값 할당
                    elif isinstance(action_traj, (list, tuple)):
                        if isinstance(action_traj[0], torch.Tensor):
                            action = action_traj[0].cpu().detach().numpy()
                        else:
                            action = np.array(action_traj[0])
                    else:
                        print(f"Unexpected action type: {type(action_traj)}")
                        # 기본 액션 (제로 액션)
                        action = np.zeros(7)
                
                # 액션 유효성 확인
                if not isinstance(action, np.ndarray):
                    action = np.array(action)
                
                # 액션 차원이 스칼라인 경우 벡터로 확장
                if action.ndim == 0:
                    action = np.zeros(7)  # 7은 예상되는 액션 차원
                
                print(f"Action shape: {action.shape}, Action values: {action}")
                
                # Store action for visualization
                action_trajectory.append(action)
                
                # Execute action in environment
                try:
                    _ = PnPEnv.step(action)
                    PnPEnv.render()
                except Exception as e:
                    print(f"Error executing action: {e}")
                    print(f"Action: {action}, Type: {type(action)}, Shape: {action.shape if hasattr(action, 'shape') else 'Unknown'}")
                
                step += 1
                
                # Check for success after action
                if PnPEnv.check_success():
                    print('Task completed successfully!')
                    break
                
                # Optional: Add a step limit
                if step >= 1000:  # 50 seconds at 20Hz
                    print('Time limit exceeded')
                    break
    except KeyboardInterrupt:
        print("Execution interrupted by user")
    finally:
        # Visualize actions
        visualize_actions(action_trajectory)

def visualize_actions(action_trajectory):
    if not action_trajectory:
        print("No actions to visualize")
        return
        
    # 모든 action이 ndarray인지 확인하고 변환
    for i in range(len(action_trajectory)):
        if not isinstance(action_trajectory[i], np.ndarray):
            action_trajectory[i] = np.array(action_trajectory[i])
        # 차원이 0인 경우 확장
        if action_trajectory[i].ndim == 0:
            action_trajectory[i] = np.array([action_trajectory[i]])
            
    # 모든 액션의 차원을 통일
    max_dim = max(a.size for a in action_trajectory)
    for i in range(len(action_trajectory)):
        if action_trajectory[i].size < max_dim:
            # 작은 차원의 액션을 확장
            padded = np.zeros(max_dim)
            padded[:action_trajectory[i].size] = action_trajectory[i]
            action_trajectory[i] = padded
    
    # 이제 모든 액션이 동일한 차원을 가지므로 배열로 변환
    action_trajectory_array = np.array(action_trajectory)
    print(f"Visualization array shape: {action_trajectory_array.shape}")
    
    # 시각화
    plt.figure(figsize=(12, 8))
    for i in range(action_trajectory_array.shape[1]):
        plt.plot(action_trajectory_array[:, i], label=f'Joint {i+1}')
    
    plt.title('Diffusion Policy Joint Actions Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Joint Angle (rad)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Save the plot
    print("Saving action plot to ./ckpt/diffusion_y/deployment_actions.png")
    plt.savefig('./ckpt/diffusion_y/deployment_actions.png')
    plt.close()

if __name__ == "__main__":
    main()
