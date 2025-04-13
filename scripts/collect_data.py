#!/usr/bin/env python
# coding: utf-8

"""
Collect Demonstration from Keyboard

Collect demonstration data for the given environment.
The task is to pick a mug and place it on the plate. The environment recognizes the success if the mug is on the plate,
the gripper opened, and the end-effector positioned above the mug.

Controls:
- WASD for the xy plane
- RF for the z-axis
- QE for tilt
- ARROWs for the rest of the rotations
- SPACEBAR to change gripper state
- Z key to reset environment and discard current episode data

For overlayed images:
- Top Right: Agent View 
- Bottom Right: Egocentric View
- Top Left: Left Side View
- Bottom Left: Top View
"""

import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import numpy as np
import shutil
from PIL import Image
# Import the SimpleEnv inside the function to avoid immediate import
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def create_dataset(repo_name, root, action_type=None):
    """Create or load a dataset for robot demonstrations
    
    Args:
        repo_name: Name of the repository
        root: Root directory to save the dataset
        action_type: Type of action to save ('joint', 'ee_pose', or 'delta_q')
    """
    # 액션 타입에 따라 저장 경로 분리
    if action_type:
        action_root = os.path.join(root, action_type)
    else:
        action_root = root
        
    create_new = True
    if os.path.exists(action_root):
        print(f"Directory {action_root} already exists.")
        ans = input(f"Do you want to delete it? (y/n) ")
        if ans == 'y':
            shutil.rmtree(action_root)
        else:
            create_new = False

    if create_new:
        # 액션 타입별로 다른 feature 설정
        features = {
            "observation.image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["state"],  # x, y, z, roll, pitch, yaw
            },
            "obj_init": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["obj_init"],  # just the initial position of the object
            },
        }
        
        # 액션 타입에 따라 액션 feature 추가
        if action_type == 'joint':
            features["action"] = {
                "dtype": "float32",
                "shape": (7,),
                "names": ["action"],  # 6 joint angles and 1 gripper
            }
        elif action_type == 'eef_pose':
            features["action"] = {
                "dtype": "float32",
                "shape": (7,),
                "names": ["action"],  # x, y, z, roll, pitch, yaw, gripper
            }
        elif action_type == 'delta_q':
            features["action"] = {
                "dtype": "float32",
                "shape": (7,),
                "names": ["action"],  # 6 delta joint angles and 1 gripper
            }
        else:
            # 모든 액션 타입을 저장하는 경우
            features.update({
                "action.joint": {
                    "dtype": "float32",
                    "shape": (7,),
                    "names": ["action_joint"],  # 6 joint angles and 1 gripper
                },
                "action.ee_pose": {
                    "dtype": "float32",
                    "shape": (7,),
                    "names": ["action_eef_pose"],  # x, y, z, roll, pitch, yaw, gripper
                },
                "action.delta_q": {
                    "dtype": "float32",
                    "shape": (7,),
                    "names": ["action_delta_q"],  # 6 delta joint angles and 1 gripper
                },
            })

        dataset = LeRobotDataset.create(
                    repo_id=f"{repo_name}_{action_type}" if action_type else repo_name,
                    root=action_root, 
                    robot_type="omy",
                    fps=20,  # 20 frames per second
                    features=features,
                    image_writer_threads=10,
                    image_writer_processes=5,
            )
    else:
        print("Load from previous dataset")
        dataset = LeRobotDataset(f"{repo_name}_{action_type}" if action_type else repo_name, root=action_root)
    
    return dataset


def collect_demonstrations(env, datasets, task_name, num_demos, seed):
    """Collect robot demonstrations using keyboard teleop
    
    Args:
        env: Environment to collect demonstrations from
        datasets: Dictionary of datasets for each action type or a single dataset
        task_name: Name of the task
        num_demos: Number of demonstrations to collect
        seed: Random seed for the environment
    """
    action = np.zeros(7)
    episode_id = 0
    record_flag = False  # Start recording when the robot starts moving
    
    # 데이터셋이 dictionary인지 확인
    is_multi_dataset = isinstance(datasets, dict)
    
    while env.env.is_viewer_alive() and episode_id < num_demos:
        env.step_env()
        if env.env.loop_every(HZ=5):
            # check if the episode is done
            done = env.check_success()
            if done: 
                # Save the episode data and reset the environment
                if is_multi_dataset:
                    for dataset in datasets.values():
                        dataset.save_episode()
                else:
                    datasets.save_episode()
                env.reset(seed=seed)
                episode_id += 1
            
            # Teleoperate the robot and get delta end-effector pose with gripper
            action, reset = env.teleop_robot()
            if not record_flag and sum(action) != 0:
                record_flag = True
                print("Start recording")
            
            if reset:
                # Reset the environment and clear the episode buffer
                # This can be done by pressing 'z' key
                env.reset(seed=seed)
                if is_multi_dataset:
                    for dataset in datasets.values():
                        dataset.clear_episode_buffer()
                else:
                    datasets.clear_episode_buffer()
                record_flag = False
            
            # Step the environment with the current action type
            joint_q = env.step(action)
            
            # Get the end-effector pose and delta joint angles
            ee_pose = env.get_ee_pose()
            
            # For delta_q, we need to temporarily set the state_type
            original_state_type = env.state_type
            env.state_type = 'delta_q'
            delta_q = env.get_delta_q()  # Get delta joint angles
            env.state_type = original_state_type  # Restore original state type
            
            # Get camera images
            agent_image, wrist_image = env.grab_image()
            
            # resize to 256x256
            agent_image = Image.fromarray(agent_image)
            wrist_image = Image.fromarray(wrist_image)
            agent_image = agent_image.resize((256, 256))
            wrist_image = wrist_image.resize((256, 256))
            agent_image = np.array(agent_image)
            wrist_image = np.array(wrist_image)
            
            if record_flag:
                # Get gripper state (last element of joint_q)
                gripper_state = joint_q[-1]
                
                # Create ee_pose with gripper state
                eef_pose_with_gripper = action
                
                # 공통 프레임 데이터
                common_frame_data = {
                    "observation.image": agent_image,
                    "observation.wrist_image": wrist_image,
                    "observation.state": ee_pose,
                    "obj_init": env.obj_init_pose,
                    "task": task_name,
                }
                
                if is_multi_dataset:
                    # 각 액션 타입별로 별도의 데이터셋에 추가
                    if 'joint' in datasets:
                        joint_frame = common_frame_data.copy()
                        joint_frame["action"] = joint_q
                        datasets['joint'].add_frame(joint_frame)
                    
                    if 'eef_pose' in datasets:
                        ee_pose_frame = common_frame_data.copy()
                        ee_pose_frame["action"] = eef_pose_with_gripper
                        datasets['eef_pose'].add_frame(ee_pose_frame)
                    
                    if 'delta_q' in datasets:
                        delta_q_frame = common_frame_data.copy()
                        delta_q_frame["action"] = delta_q
                        datasets['delta_q'].add_frame(delta_q_frame)
                else:
                    # 모든 액션 타입을 하나의 데이터셋에 추가
                    datasets.add_frame({
                            **common_frame_data,
                            "action.joint": joint_q,
                            "action.eef_pose": eef_pose_with_gripper,  # x, y, z, roll, pitch, yaw, gripper
                            "action.delta_q": delta_q,  # delta joint angles with gripper
                        }
                    )
            
            env.render()
    
    # Close the environment viewer
    env.env.close_viewer()
    
    # Clean up the images folder
    if is_multi_dataset:
        for action_type, dataset in datasets.items():
            shutil.rmtree(dataset.root / 'images')
    else:
        shutil.rmtree(datasets.root / 'images')


def main():
    # Configuration
    SEED = None  # Set to None to randomize object positions
    REPO_NAME = 'omy_pnp'
    NUM_DEMO = 25  # Number of demonstrations to collect
    ROOT = "./demo_data_4"  # The root directory to save the demonstrations
    TASK_NAME = 'Put mug cup on the plate'
    XML_PATH = './asset/example_scene_y.xml'
    
    # 모든 액션 타입에 대한 데이터셋 수집 여부 확인
    print("Do you want to collect separate datasets for each action type? (y/n)")
    separate_datasets = input().lower() == 'y'
    
    # Import the SimpleEnv here to avoid immediate import
    from mujoco_env.y_env import SimpleEnv
    
    # Define the environment
    env = SimpleEnv(XML_PATH, seed=SEED, state_type='joint_angle')
    
    if separate_datasets:
        # 액션 타입별로 별도의 데이터셋 생성
        action_types = ['joint', 'eef_pose', 'delta_q']
        datasets = {}
        
        for action_type in action_types:
            print(f"\nCreating dataset for action type: {action_type}")
            datasets[action_type] = create_dataset(REPO_NAME, ROOT, action_type)
        
        # 모든 데이터셋에 데모 수집
        collect_demonstrations(env, datasets, TASK_NAME, NUM_DEMO, SEED)
    else:
        # 기존 방식대로 단일 데이터셋 생성
        dataset = create_dataset(REPO_NAME, ROOT)
        
        # 단일 데이터셋에 데모 수집
        collect_demonstrations(env, dataset, TASK_NAME, NUM_DEMO, SEED)


if __name__ == "__main__":
    main()


# In[ ]:




