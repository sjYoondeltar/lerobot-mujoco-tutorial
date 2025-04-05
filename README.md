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
I do **not** recommend installing lerobot package with `pip install lerobot`. This causes errors. 

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

### Updates & Plans

:white_check_mark: Viewer Update.

:heavy_check_mark: Add different mugs, plates for different language instructions.

:heavy_check_mark: Add pi_0 training and inference. 


## 1. Collection Demonstration Data

Run [1.collect_data.ipynb](1.collect_data.ipynb) or use the command-line script:

```bash
# Run data collection script
python scripts/collect_data.py
```

Collect demonstration data for the given environment.
The task is to pick a mug and place it on the plate. The environment recognizes the success if the mug is on the plate, the gripper opened, and the end-effector positioned above the mug.

<img src="./media/teleop.gif" width="480" height="360">

Use WASD for the xy plane, RF for the z-axis, QE for tilt, and ARROWs for the rest of the rotations. 

SPACEBAR will change your gripper's state, and Z key will reset your environment with discarding the current episode data.

For overlayed images, 
- Top Right: Agent View 
- Bottom Right: Egocentric View
- Top Left: Left Side View
- Bottom Left: Top View

The dataset now collects multiple action types simultaneously:

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
        "names": ["state"],  # x, y, z, roll, pitch, yaw
    },
    "action.joint": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["action_joint"],  # 6 joint angles and 1 gripper
    },
    "action.ee_pose": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["action_ee_pose"],  # x, y, z, roll, pitch, yaw, gripper
    },
    "action.delta_q": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["action_delta_q"],  # 6 delta joint angles and 1 gripper
    },
    "obj_init": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["obj_init"],  # just the initial position of the object. Not used in training.
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

Run [2.visualize_data.ipynb](2.visualize_data.ipynb)

<img src="./media/data.gif" width="480" height="360"></img>

Visualize your action based on the reconstructed simulation scene. 

The main simulation is replaying the action.

The overlayed images on the top right and bottom right are from the dataset. 

## 3. Train Action-Chunking-Transformer (ACT)

Run [3.train.ipynb](3.train.ipynb) or use the command-line script:

```bash
# Train ACT with joint angles (default)
python scripts/act/train.py --action_type joint

# Train ACT with end-effector pose
python scripts/act/train.py --action_type ee_pose

# Train ACT with delta joint angles
python scripts/act/train.py --action_type delta_q
```

**This takes around 30~60 mins**.

Train the ACT model on your custom dataset. In this example, we set chunk_size as 10. 

You can choose from three action types for training:
- `joint`: Uses joint angles for control (default)
- `ee_pose`: Uses end-effector pose (x,y,z,roll,pitch,yaw)
- `delta_q`: Uses delta joint angles (changes from current position)

The trained checkpoint will be saved in './ckpt/act_y/{action_type}' folder.

To evaluate the policy on the dataset, you can calculate the error between ground-truth actions from the dataset.

<image src="./media/inference.png"  width="480" height="360">

<details>
    <summary>PicklingError: Can't pickle <function <lambda> at 0x131d1bd00>: attribute lookup <lambda> on __main__ failed</summary>
If you have a pickling error, 
        
```
PicklingError: Can't pickle <function <lambda> at 0x131d1bd00>: attribute lookup <lambda> on __main__ failed
```

Please set your num_workers as 0, like, 

```
dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=0, # 4
    batch_size=64,
    shuffle=True,
    pin_memory=device.type != "cpu",
    drop_last=True,
)
```
</details>

## 4. Deploy your Policy

Instead of using a notebook, you can now deploy your trained policy (either ACT or Diffusion) using the `scripts/deploy.py` script.

You can download pre-trained checkpoints from [google drive](https://drive.google.com/drive/folders/1UqxqUgGPKU04DkpQqSWNgfYMhlvaiZsp?usp=sharing) if you don't have a GPU to train your model.

Use the following command structure:

```bash
python scripts/deploy.py --policy_type <type> [--action_type <action_type>] [--ckpt_dir <path_to_checkpoint>] [--xml_path <path_to_xml>] [--max_steps <steps>] [--control_hz <hz>]
```

Arguments:
- `--policy_type`: Required. Choose either `act`, `diffusion` or `vqbet`.
- `--action_type`: Optional. Choose from `joint`, `ee_pose`, or `delta_q`. Defaults to `joint`.
- `--ckpt_dir`: Optional. Path to the checkpoint directory. Defaults to `ckpt/act_y` or `ckpt/diffusion_y` based on `--policy_type`.
- `--xml_path`: Optional. Path to the MuJoCo scene XML file. Defaults to `./asset/example_scene_y.xml`.
- `--max_steps`: Optional. Maximum simulation steps. Defaults to `1000`.
- `--control_hz`: Optional. Control frequency. Defaults to `20`.

**Examples:**

*   **Deploy ACT policy with joint angles (default):**
    ```bash
    python scripts/deploy.py --policy_type act --action_type joint
    ```
*   **Deploy ACT policy with end-effector pose:**
    ```bash
    python scripts/deploy.py --policy_type act --action_type ee_pose
    ```
*   **Deploy ACT policy with delta joint angles:**
    ```bash
    python scripts/deploy.py --policy_type act --action_type delta_q
    ```
*   **Deploy Diffusion policy (specifying a checkpoint path):**
    ```bash
    python scripts/deploy.py --policy_type diffusion --ckpt_dir ./my_diffusion_checkpoint
    ```

<img src="./media/rollout.gif" width="480" height="360" controls></img>

The script will load the specified policy and deploy it in the MuJoCo simulation environment. The environment's action type will be automatically set to match the trained model's action type.

## Acknowledgements
- The asset for the robotis-omy manipulator is from [robotis_mujoco_menagerie](https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie/tree/main).
- The [MuJoco Parser Class](./mujoco_env/mujoco_parser.py) is modified from [yet-another-mujoco-tutorial](https://github.com/sjchoi86/yet-another-mujoco-tutorial-v3). 
- We refer to original tutorials from [lerobot examples](https://github.com/huggingface/lerobot/tree/main/examples).  
- The assets for plate and mug is from [Objaverse](https://objaverse.allenai.org/).
