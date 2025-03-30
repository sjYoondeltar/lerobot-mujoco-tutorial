#!/usr/bin/env python
# coding: utf-8

"""
Visualize Data

This script allows you to playback and visualize robot demonstration data.
It loads the collected dataset and replays the actions in a MuJoCo simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from mujoco_env.y_env import SimpleEnv


def visualize_dataset(env, dataset, episode_idx=0):
    """Visualize a specific episode from the dataset"""
    # Get the episode data
    episode = dataset.get_episode(episode_idx)
    obj_init = episode["obj_init"][0]
    
    # Reset the environment with the object at the initial position
    env.reset(obj_init=obj_init)
    
    # Playback the episode
    for i in range(len(episode["action"])):
        env.step_env()
        
        # Visualize only if the environment is still alive
        if env.env.is_viewer_alive() and env.env.loop_every(HZ=20):
            # Set the joint angles from the dataset
            joint_q = episode["action"][i]
            env.set_joint_and_render(joint_q)
    
    # Close the viewer
    env.env.close_viewer()


def main():
    # Configuration
    REPO_NAME = 'omy_pnp'
    ROOT = "./demo_data"  # Path to demonstration data
    XML_PATH = './asset/example_scene_y.xml'
    
    # Try to load the dataset
    try:
        dataset = LeRobotDataset(REPO_NAME, root=ROOT)
        print(f"Dataset loaded with {dataset.num_episodes} episodes")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Please make sure you have collected data or are using the correct path.")
        return
    
    # Initialize the environment
    env = SimpleEnv(XML_PATH, seed=0, state_type='joint_angle')
    
    # Visualize the first episode (index 0)
    visualize_dataset(env, dataset, episode_idx=0)


if __name__ == "__main__":
    main()


# # Visualize your data
# 
# <img src="./media/data.gif" width="480" height="360">
# 
# Visualize your action based on the reconstructed simulation scene. 
# 
# The main simulation is replaying the action.
# 
# The overlayed images on the top right and bottom right are from the dataset. 

# In[ ]:


from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import numpy as np
from lerobot.common.datasets.utils import write_json, serialize_dict

dataset = LeRobotDataset('omy_pnp', root='./demo_data') # if youu want to use the example data provided, root = './demo_data_example' instead!


# ## Load Dataset

# In[2]:


import torch

class EpisodeSampler(torch.utils.data.Sampler):
    """
    Sampler for a single episode
    """
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


# In[8]:


# Select an episode index that you want to visualize
episode_index = 0

episode_sampler = EpisodeSampler(dataset, episode_index)
dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=1,
    batch_size=1,
    sampler=episode_sampler,
)


# ## Visualize your Dataset on Simulation

# In[9]:


from mujoco_env.y_env import SimpleEnv
xml_path = './asset/example_scene_y.xml'
PnPEnv = SimpleEnv(xml_path, action_type='joint_angle')


# In[10]:


step = 0
iter_dataloader = iter(dataloader)
PnPEnv.reset()

while PnPEnv.env.is_viewer_alive():
    PnPEnv.step_env()
    if PnPEnv.env.loop_every(HZ=20):
        # Get the action from dataset
        data = next(iter_dataloader)
        if step == 0:
            # Reset the object pose based on the dataset
            PnPEnv.set_obj_pose(data['obj_init'][0,:3], data['obj_init'][0,3:])
        # Get the action from dataset
        action = data['action'].numpy()
        obs = PnPEnv.step(action[0])

        # Visualize the image from dataset to rgb_overlay
        PnPEnv.rgb_agent = data['observation.image'][0].numpy()*255
        PnPEnv.rgb_ego = data['observation.wrist_image'][0].numpy()*255
        PnPEnv.rgb_agent = PnPEnv.rgb_agent.astype(np.uint8)
        PnPEnv.rgb_ego = PnPEnv.rgb_ego.astype(np.uint8)
        # 3 256 256 -> 256 256 3
        PnPEnv.rgb_agent = np.transpose(PnPEnv.rgb_agent, (1,2,0))
        PnPEnv.rgb_ego = np.transpose(PnPEnv.rgb_ego, (1,2,0))
        PnPEnv.rgb_side = np.zeros((480, 640, 3), dtype=np.uint8)
        PnPEnv.rgb_top = np.zeros((480, 640, 3), dtype=np.uint8)
        PnPEnv.render()
        step += 1

        if step == len(episode_sampler):
            # start from the beginning
            iter_dataloader = iter(dataloader)
            PnPEnv.reset()
            step = 0


# In[11]:


PnPEnv.env.close_viewer()


# ### [Optional] Save Stats.json for other versions

# In[7]:


stats = dataset.meta.stats
PATH = dataset.root / 'meta' / 'stats.json'
stats = serialize_dict(stats)

write_json(stats, PATH)


# 
