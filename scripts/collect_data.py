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


def create_dataset(repo_name, root):
    """Create or load a dataset for robot demonstrations"""
    create_new = True
    if os.path.exists(root):
        print(f"Directory {root} already exists.")
        ans = input("Do you want to delete it? (y/n) ")
        if ans == 'y':
            shutil.rmtree(root)
        else:
            create_new = False

    if create_new:
        dataset = LeRobotDataset.create(
                    repo_id=repo_name,
                    root=root, 
                    robot_type="omy",
                    fps=20,  # 20 frames per second
                    features={
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
                        "action.joint": {
                            "dtype": "float32",
                            "shape": (7,),
                            "names": ["action_joint"],  # 6 joint angles and 1 gripper
                        },
                        "action.ee_pose": {
                            "dtype": "float32",
                            "shape": (6,),
                            "names": ["action_ee_pose"],  # x, y, z, roll, pitch, yaw
                        },
                        "obj_init": {
                            "dtype": "float32",
                            "shape": (6,),
                            "names": ["obj_init"],  # just the initial position of the object
                        },
                    },
                    image_writer_threads=10,
                    image_writer_processes=5,
            )
    else:
        print("Load from previous dataset")
        dataset = LeRobotDataset(repo_name, root=root)
    
    return dataset


def collect_demonstrations(env, dataset, task_name, num_demos, seed):
    """Collect robot demonstrations using keyboard teleop"""
    action = np.zeros(7)
    episode_id = 0
    record_flag = False  # Start recording when the robot starts moving
    
    while env.env.is_viewer_alive() and episode_id < num_demos:
        env.step_env()
        if env.env.loop_every(HZ=20):
            # check if the episode is done
            done = env.check_success()
            if done: 
                # Save the episode data and reset the environment
                dataset.save_episode()
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
                dataset.clear_episode_buffer()
                record_flag = False
            
            # Step the environment
            joint_q = env.step(action)
            
            # Get the end-effector pose and images
            ee_pose = env.get_ee_pose()
            agent_image, wrist_image = env.grab_image()
            
            # resize to 256x256
            agent_image = Image.fromarray(agent_image)
            wrist_image = Image.fromarray(wrist_image)
            agent_image = agent_image.resize((256, 256))
            wrist_image = wrist_image.resize((256, 256))
            agent_image = np.array(agent_image)
            wrist_image = np.array(wrist_image)
            
            if record_flag:
                # Add the frame to the dataset
                dataset.add_frame({
                        "observation.image": agent_image,
                        "observation.wrist_image": wrist_image,
                        "observation.state": ee_pose, 
                        "action.joint": joint_q,
                        "action.ee_pose": ee_pose[:6],  # x, y, z, roll, pitch, yaw
                        "obj_init": env.obj_init_pose,
                        "task": task_name,
                    }
                )
            
            env.render()
    
    # Close the environment viewer
    env.env.close_viewer()
    
    # Clean up the images folder
    shutil.rmtree(dataset.root / 'images')


def main():
    # Configuration
    SEED = 0  # Set to None to randomize object positions
    REPO_NAME = 'omy_pnp'
    NUM_DEMO = 1  # Number of demonstrations to collect
    ROOT = "./demo_data"  # The root directory to save the demonstrations
    TASK_NAME = 'Put mug cup on the plate'
    XML_PATH = './asset/example_scene_y.xml'
    
    """
    Available action_type options in SimpleEnv:
    - 'eef_pose': Control using end-effector pose (x,y,z,roll,pitch,yaw)
    - 'joint_angle': Control using absolute joint angles
    - 'delta_joint_angle': Control using delta joint angles (changes to current angles)
    
    Available state_type options in SimpleEnv:
    - 'joint_angle': Get joint angles as state
    - 'ee_pose': Get end-effector pose as state
    - 'delta_q': Get delta joint angles (when using delta_joint_angle action type)
    """
    
    # Import the SimpleEnv here to avoid immediate import
    from mujoco_env.y_env import SimpleEnv
    
    # Define the environment
    env = SimpleEnv(XML_PATH, seed=SEED, state_type='joint_angle', action_type='eef_pose')
    
    # Create or load dataset
    dataset = create_dataset(REPO_NAME, ROOT)
    
    # Collect demonstrations
    collect_demonstrations(env, dataset, TASK_NAME, NUM_DEMO, SEED)


if __name__ == "__main__":
    main()


# In[ ]:




