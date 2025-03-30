# LeRobot Tutorial with MuJoCo
This repository contains minimal examples for collecting demonstration data and training (or fine-tuning) vision language action models on custom datasets. 

## Installation
We have tested our environment on python 3.10. 

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

Run [1.collect_data.ipynb](notebooks/1.collect_data.ipynb)

Collect demonstration data for the given environment.
The task is to pick a mug and place it on the plate. The environment recognizes the success if the mug is on the plate, gthe ripper opened, and the end-effector positioned above the mug.

<img src="./media/teleop.gif" width="480" height="360">

Use WASD for the xy plane, RF for the z-axis, QE for tilt, and ARROWs for the rest of rthe otations. 

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

This will make the dataset on './demo_data' folder, which will look like this,
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

For convenience, we have added [Example Data](./demo_data_example/) to the repository. 

## 2. Playback Your Data

Run [2.visualize_data.ipynb](notebooks/2.visualize_data.ipynb)

<img src="./media/data.gif" width="480" height="360"></img>

Visualize your action based on the reconstructed simulation scene. 

The main simulation is replaying the action.

The overlayed images on the top right and bottom right are from the dataset. 

## 3. Train Action-Chunking-Transformer (ACT)

Run [3.train.ipynb](notebooks/3.train.ipynb)

**This takes around 30~60 mins**.

Train the ACT model on your custom dataset. In this example, we set chunk_size as 10. 

The trained checkpoint will be saved in './ckpt/act_y' folder.

To evaluate the policy on the dataset, you can calculate the error between ground-truth actions from the dataset.

<image src="./media/inference.png"  width="480" height="360">


## 4. Deploy your Policy

Run [4.deploy.ipynb](notebooks/4.deploy.ipynb)

You can download checkpoint from [google drive](https://drive.google.com/drive/folders/1UqxqUgGPKU04DkpQqSWNgfYMhlvaiZsp?usp=sharing) if you don't have gpu to train your model.

<img src="./media/rollout.gif" width="480" height="360" controls></img>

Deploy trained policy in simulation.

## Acknowledgements
- The asset for the robotis-omy manipulator is from [robotis_mujoco_menagerie](https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie/tree/main).
- The [MuJoco Parser Class](./mujoco_env/mujoco_parser.py) is modified from [yet-another-mujoco-tutorial](https://github.com/sjchoi86/yet-another-mujoco-tutorial-v3). 
- We refer to original tutorials from [lerobot examples](https://github.com/huggingface/lerobot/tree/main/examples).  
- The assets for plate and mug is from [Objaverse](https://objaverse.allenai.org/).