#!/usr/bin/env python
# coding: utf-8

"""
Collect Language Demonstration from Keyboard

Collect demonstration data for language-conditioned robot tasks.
The task is to pick a mug (red or blue) and place it on the plate based on language instructions.
The environment recognizes the success if the mug is on the plate, the gripper opened, and the end-effector positioned above the mug.

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
"""

import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import numpy as np
import shutil
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def create_dataset(repo_name, root):
    """Create or load a dataset for language-conditioned robot demonstrations
    
    Args:
        repo_name: Name of the repository
        root: Root directory to save the dataset
    """
    create_new = True
    if os.path.exists(root):
        print(f"Directory {root} already exists.")
        ans = input(f"Do you want to delete it? (y/n) ")
        if ans == 'y':
            shutil.rmtree(root)
        else:
            create_new = False

    if create_new:
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
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["action"],  # 6 joint angles and 1 gripper
            },
            "obj_init": {
                "dtype": "float32",
                "shape": (9,),
                "names": ["obj_init"],  # initial position of red mug, blue mug, and plate
            },
        }

        dataset = LeRobotDataset.create(
                    repo_id=repo_name,
                    root=root, 
                    robot_type="omy",
                    fps=20,  # 20 frames per second
                    features=features,
                    image_writer_threads=10,
                    image_writer_processes=5,
            )
    else:
        print("Load from previous dataset")
        dataset = LeRobotDataset(repo_name, root=root)
    
    return dataset


def collect_demonstrations(env, dataset, num_demos):
    """Collect robot demonstrations using keyboard teleop
    
    Args:
        env: Environment to collect demonstrations from
        dataset: Dataset to save the demonstrations
        num_demos: Number of demonstrations to collect
    """
    action = np.zeros(7)
    episode_id = 0
    record_flag = False  # Start recording when the robot starts moving
    
    print("Start collecting language-conditioned demonstrations!")
    print(f"Target: {num_demos} demonstrations")
    print("\nControls:")
    print("- WASD: xy plane movement")
    print("- RF: z-axis movement") 
    print("- QE: tilt rotation")
    print("- Arrow keys: other rotations")
    print("- SPACEBAR: toggle gripper")
    print("- Z: reset environment\n")
    
    while env.env.is_viewer_alive() and episode_id < num_demos:
        env.step_env()
        if env.env.loop_every(HZ=20):
            # check if the episode is done
            done = env.check_success()
            if done: 
                print(f"Success! Episode {episode_id + 1} completed.")
                # Save the episode data and reset the environment
                dataset.save_episode()
                env.reset()
                episode_id += 1
                record_flag = False
                if episode_id < num_demos:
                    print(f"Starting episode {episode_id + 1}/{num_demos}")
                    print(f"Task: {env.instruction}")
            
            # Teleoperate the robot and get delta end-effector pose with gripper
            action, reset = env.teleop_robot()
            if not record_flag and sum(action) != 0:
                record_flag = True
                print("Start recording")
            
            if reset:
                # Reset the environment and clear the episode buffer
                # This can be done by pressing 'z' key
                env.reset()
                dataset.clear_episode_buffer()
                record_flag = False
                print("Environment reset. Episode data discarded.")
                print(f"Task: {env.instruction}")
            
            # Step the environment
            # Get the end-effector pose and images
            agent_image, wrist_image = env.grab_image()
            
            # resize to 256x256
            agent_image = Image.fromarray(agent_image)
            wrist_image = Image.fromarray(wrist_image)
            agent_image = agent_image.resize((256, 256))
            wrist_image = wrist_image.resize((256, 256))
            agent_image = np.array(agent_image)
            wrist_image = np.array(wrist_image)
            
            joint_q = env.step(action)
            action = env.q[:7]  # 6 joint angles and 1 gripper
            action = action.astype(np.float32)
            
            if record_flag:
                # Add the frame to the dataset
                dataset.add_frame({
                        "observation.image": agent_image,
                        "observation.wrist_image": wrist_image,
                        "observation.state": joint_q[:6], 
                        "action": action,
                        "obj_init": env.obj_init_pose,
                    }, task=env.instruction
                )
            
            env.render(teleop=True, idx=episode_id)
    
    # Close the environment viewer
    env.env.close_viewer()
    
    # Clean up the images folder
    shutil.rmtree(dataset.root / 'images')
    
    print(f"\nData collection completed! {episode_id} demonstrations saved.")


def main():
    # Command line argument parsing
    import argparse
    parser = argparse.ArgumentParser(description='Collect language-conditioned demonstration data for robot tasks.')
    parser.add_argument('--seed', type=int, default=0, 
                        help='Random seed for environment. Use 0 for fixed positions, None to randomize.')
    parser.add_argument('--repo_name', type=str, default='omy_pnp_language',
                        help='Name of the repository')
    parser.add_argument('--num_demo', type=int, default=20,
                        help='Number of demonstrations to collect')
    parser.add_argument('--root', type=str, default='./demo_data_language',
                        help='Root directory to save the demonstrations')
    parser.add_argument('--xml_path', type=str, default='asset/example_scene_y2.xml',
                        help='Path to the XML scene file')
    args = parser.parse_args()

    # Apply configuration
    SEED = args.seed
    REPO_NAME = args.repo_name
    NUM_DEMO = args.num_demo
    ROOT = args.root
    XML_PATH = args.xml_path
    
    print(f"Configuration:")
    print(f"- Seed: {SEED}")
    print(f"- Repository: {REPO_NAME}")
    print(f"- Demonstrations: {NUM_DEMO}")
    print(f"- Root directory: {ROOT}")
    print(f"- XML path: {XML_PATH}")
    print()
    
    # Import SimpleEnv2 here to avoid immediate import
    from mujoco_env.y_env2 import SimpleEnv2
    
    # Define the environment
    print("Initializing environment...")
    PnPEnv = SimpleEnv2(XML_PATH, seed=SEED, state_type='joint_angle')
    print(f"Initial task: {PnPEnv.instruction}")
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = create_dataset(REPO_NAME, ROOT)
    
    # Collect demonstrations
    collect_demonstrations(PnPEnv, dataset, NUM_DEMO)


if __name__ == "__main__":
    main() 