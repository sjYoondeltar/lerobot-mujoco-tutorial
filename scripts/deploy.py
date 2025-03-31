#!/usr/bin/env python
# coding: utf-8

"""
Deploy your Policy

This script loads a trained policy model and deploys it in a MuJoCo simulation environment.
The model will control a robot to perform the pick and place task that it was trained on.
"""

import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from lerobot.common.policies.act.modeling_act import ACTPolicy
# Import mujoco_env components inside the functions


def load_policy(ckpt_dir):
    """Load a trained policy from checkpoint"""
    if not os.path.exists(ckpt_dir):
        print(f"Checkpoint directory {ckpt_dir} does not exist.")
        print("Please train the model first or download a pre-trained model.")
        return None
    
    # Load the ACT policy from the checkpoint
    policy = ACTPolicy.from_pretrained(ckpt_dir)
    print(f"Policy loaded from {ckpt_dir}")
    
    return policy


def deploy_policy(env, policy, max_steps=1000):
    """Deploy the policy in the environment"""
    # 모델의 디바이스 확인
    device = next(policy.parameters()).device
    print(f"Model is on device: {device}")
    
    # 환경 초기화
    env.reset()
    policy.reset()  # 정책도 초기화
    policy.eval()
    
    # 이미지 변환 설정
    from torchvision import transforms
    from PIL import Image
    img_transform = transforms.ToTensor()
    
    step_count = 0
    done = False
    
    while env.env.is_viewer_alive() and step_count < max_steps and not done:
        env.step_env()
        
        if env.env.loop_every(HZ=20):
            # 상태 및 이미지 획득
            obs = env.get_ee_pose()
            image, wrist_image = env.grab_image()  # 두 이미지 반환
            
            # 이미지 전처리
            image = Image.fromarray(image)
            image = image.resize((256, 256))
            image = img_transform(image)
            
            wrist_image = Image.fromarray(wrist_image)
            wrist_image = wrist_image.resize((256, 256))
            wrist_image = img_transform(wrist_image)
            
            # 입력 데이터 준비
            data = {
                'observation.state': torch.tensor([obs]).to(device),
                'observation.image': image.unsqueeze(0).to(device),
                'observation.wrist_image': wrist_image.unsqueeze(0).to(device),
                'task': ['Put mug cup on the plate'],  # 태스크 설명 (필요시 변경)
                'timestamp': torch.tensor([step_count/20]).to(device)  # 타임스탬프 추가
            }
            
            # 액션 예측
            with torch.no_grad():
                action = policy.select_action(data)
                action = action[0].cpu().detach().numpy()
            
            # 환경에 액션 적용
            _ = env.step(action)
            
            # 성공 확인
            done = env.check_success()
            if done:
                print("Task completed successfully!")
            
            env.render()
            step_count += 1
    
    env.env.close_viewer()
    
    return done


def main():
    # Configuration
    XML_PATH = './asset/example_scene_y.xml'
    CKPT_DIR = "./ckpt/act_y"  # ACT 모델 체크포인트 경로
    
    # Import mujoco_env components here to avoid immediate import
    from mujoco_env.y_env import SimpleEnv
    
    # Initialize the environment
    env = SimpleEnv(XML_PATH, seed=0, state_type='joint_angle')
    
    # Load policy
    policy = load_policy(CKPT_DIR)
    if policy is None:
        return
    
    # Deploy policy
    print("Deploying policy in simulation...")
    success = deploy_policy(env, policy)
    
    if not success:
        print("Task was not completed within the time limit.")


if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




