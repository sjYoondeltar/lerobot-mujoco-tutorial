# LeRobot Tutorial with MuJoCo
This repository contains minimal exampls for collecting demonstration data and training (or fine-tuning) vision language action models on custom dataset. 

## Installation
We have tested our enviornment on python 3.10. 

First, let's install lerobot package.
```
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e .
```

Install mujoco package dependencies.
```
cd ..
pip install -r requirements.txt
```
Make sure your mujoco version is **3.1.6**.

Unzip the asset
```
cd asset/objaverse
unzip plate_11.zip
```

## 1. Collection Demonstration Data

Run [1.collect_data.ipynb](1.collect_data.ipynb)

Collect demonstration data for given environment.
The task it to pick mug, and place it on the plate. The environment recognize the success if the mug is on the plate, gripper opened, and the end-effector postioned above the mug.

<video src="./media/teleop.mp4" width="480" height="360" controls></video>

Use WASD for xy plane, RF for z axis, QE for tilit, and ARROWs for the rest of rotations. 

SPACEBAR will change your gripper's state, and Z key will reset your environment with discarding the current episode data.

For overlayed images, 
- Top Right: Agent View 
- Bottom Right: Egocentric View
- Top Left: Left Side View
- Bottom Left: Top View

The dataset is contained as follows:
```
fps = 20,
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
        "names": ["state"], # x, y, z, roll, pitch, yaw
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["action"], # 6 joint angles and 1 gripper
    },
    "obj_init": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["obj_init"], # just the initial position of the object. Not used in training.
    },
},

```

This will make the dataset on './demo_data' folder, which will look like,
```
.
├── data
│   ├── chunk-000
│   │   ├── episode_000000.parquet
│   │   └── ...
├── meta
│   ├── episodes.jsonl
│   ├── info.json
│   ├── stats.json
│   └── tasks.jsonl
└── 
```

We have added [Example Data](./demo_data_example/) in the repository for convinience. 

## 2. Playback Your Data

Run [2.visualize_data.ipynb](2.visualize_data.ipynb)

<video src="./media/data.mp4" width="480" height="360" controls></video>

Visualize your action based on the recontructed simulation scene. 

The main simulation is replaying the action.

The overlayed images on the top right and bottom right are from the dataset. 

## 3. Train Action-Chunking-Transformer (ACT)

Run [3.train.ipynb](3.train.ipynb)

**This takes around 30~60 mins**.

Train ACT model on your custom dataset. In this example, we set chunk_size as 10. 

The trained checkpoint will be saved in './ckpt/act_y' folder.

To evaluate the policy on the dataset, you can calculate error between ground-truth action from the dataset.

<image src="./media/inference.png"  width="480" height="360">


## 4. Deploy your Policy

Run [4.deploy.ipynb](4.deploy.ipynb)

You can download checkpoint from [google drive](https://drive.google.com/drive/folders/1UqxqUgGPKU04DkpQqSWNgfYMhlvaiZsp?usp=sharing) if you don't have gpu to train your model.

<video src="./media/rollout.mp4" width="480" height="360" controls></video>

Deploy trained policy in simualtion.

## Acknowledgements
- The asset for the robotis-omy manipulater is from [robotis_mujoco_menagerie](https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie/tree/main).
- The [MuJoco Parser Class](./mujoco_env/mujoco_parser.py) is modified from [yet-another-mujoco-tutorial](https://github.com/sjchoi86/yet-another-mujoco-tutorial-v3). 
- We refer original tutorials from [lerobot examples](https://github.com/huggingface/lerobot/tree/main/examples).  