#!/usr/bin/env python
# coding: utf-8

"""
Train Action-Chunking-Transformer (ACT)

This script trains an ACT model on the collected robot demonstration dataset.
It takes approximately 30-60 minutes to train the model.
The trained checkpoint will be saved in the './ckpt/act_y' folder.
"""

import os
import sys
import argparse

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import time
import torch
import matplotlib.pyplot as plt
from torchvision import transforms # Import transforms
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.datasets.factory import resolve_delta_timestamps


def create_or_load_policy(ckpt_dir, action_type='joint', load_ckpt=False):
    """
    Create a new policy or load from checkpoint
    
    Args:
        ckpt_dir: Directory to save or load the checkpoint from
        action_type: Type of action to train with ('joint', 'ee_pose', or 'delta_q')
        load_ckpt: Whether to load from checkpoint
    """
    # 액션 타입에 따라 데이터셋 경로 설정
    dataset_root = os.path.join('./demo_data', action_type)
    dataset_metadata = LeRobotDatasetMetadata("omy_pnp", root=dataset_root)
    features = dataset_to_policy_features(dataset_metadata.features)
    
    # 디버깅: 사용 가능한 특성 출력
    print(f"Available features: {list(features.keys())}")
    print("Feature types:")
    for k, v in features.items():
        print(f"  - {k}: type={v.type if hasattr(v, 'type') else 'None'}, shape={v.shape if hasattr(v, 'shape') else 'None'}")
    
    # 액션 관련 feature 찾기
    output_features = {k: v for k, v in features.items() if k == "action" and v.type is FeatureType.ACTION}
    
    # output_features가 비어있는지 확인
    if not output_features:
        print(f"WARNING: No output features found for action_type '{action_type}'")
        print("Dataset might not contain the 'action' key.")
        print(f"Make sure you've collected data with action_type='{action_type}'.")
        raise ValueError(f"No features found for action type: {action_type}")
    
    # 입력 특성 설정
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    
    # 새로운 API 방식으로 설정
    cfg = ACTConfig(
        input_features=input_features, 
        output_features=output_features, 
        chunk_size=10, 
        n_action_steps=10
    )
    
    # Adjust the checkpoint directory to include the action type
    action_type_ckpt_dir = os.path.join(ckpt_dir, action_type)
    
    if load_ckpt and os.path.exists(action_type_ckpt_dir):
        print(f"Loading policy from {action_type_ckpt_dir}")
        policy = ACTPolicy.from_pretrained(action_type_ckpt_dir)
    else:
        print(f"Creating new policy for action type: {action_type}")
        try:
            policy = ACTPolicy(cfg, dataset_stats=dataset_metadata.stats)
            print("ACTPolicy successfully created")
        except Exception as e:
            print(f"Error creating ACTPolicy: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    return policy, dataset_metadata, action_type_ckpt_dir


def prepare_data(dataset_name, policy, dataset_metadata, action_type):
    """Prepare data for training using the new API"""
    # Policy의 config에서 delta_timestamps 해석
    delta_timestamps = resolve_delta_timestamps(policy.config, dataset_metadata)
    
    # Define image augmentations (excluding flips and rotations)
    image_augmentation_transforms = transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
        # Add 5 RandomErasing masks
        transforms.RandomErasing(p=1.0, scale=(0.01, 0.015), ratio=(0.95, 1.05), value=0),
        transforms.RandomErasing(p=1.0, scale=(0.01, 0.015), ratio=(0.95, 1.05), value=0),
        transforms.RandomErasing(p=1.0, scale=(0.01, 0.015), ratio=(0.95, 1.05), value=0),
        transforms.RandomErasing(p=1.0, scale=(0.01, 0.015), ratio=(0.95, 1.05), value=0),
        transforms.RandomErasing(p=1.0, scale=(0.01, 0.015), ratio=(0.95, 1.05), value=0),
    ])

    # 액션 타입에 따라 데이터셋 경로 설정
    dataset_root = os.path.join('./demo_data', action_type)
    
    # 새 API 방식으로 데이터셋 생성, image_transforms 인자 사용
    dataset = LeRobotDataset(
        "omy_pnp", 
        delta_timestamps=delta_timestamps, 
        root=dataset_root,
        image_transforms=image_augmentation_transforms # Pass the defined transforms
    )
    
    # 훈련용 데이터로더 생성
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=128,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    
    return dataset, dataloader


def train_policy(policy, dataset, dataloader, ckpt_dir, action_type, num_epochs=3000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.train()
    policy.to(device)
    
    # 옵티마이저 설정
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    
    os.makedirs(ckpt_dir, exist_ok=True)
    losses = []
    
    # 훈련 루프
    step = 0
    current_epoch = 0
    for epoch in range(num_epochs // len(dataloader) + 1):
        current_epoch = epoch
        for batch in dataloader:
            if step >= num_epochs:
                break
            
            # 배치 데이터 준비 (이제 'action' 키를 직접 사용)
            inp_batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) 
                        for k, v in batch.items()}
            
            # 예측 및 손실 계산
            loss, _  = policy(inp_batch)
            
            # 손실 역전파 및 가중치 업데이트
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 손실 기록
            losses.append(loss.item())
            
            # 로그 출력
            if step % 100 == 0:
                print(f"Step {step}, Epoch {current_epoch}, Loss: {loss.item():.4f}")
            
            # 100 스텝마다 모델 저장 (스텝 번호 포함)
            if step % 100 == 0 and step > 0:
                step_ckpt_dir = os.path.join(ckpt_dir, f'step_{step}')
                os.makedirs(step_ckpt_dir, exist_ok=True)
                policy.save_pretrained(step_ckpt_dir)
                
                # 손실 그래프 저장
                plt.figure()
                plt.plot(losses)
                plt.xlabel('Steps')
                plt.ylabel('Loss')
                plt.title(f'Training Loss for {action_type} (Step {step})')
                plt.savefig(os.path.join(step_ckpt_dir, f'loss_step_{step}.png'))
                plt.close()
                
                print(f"Saved checkpoint at step {step}")
            
            step += 1
            if step >= num_epochs:
                break
    
    # 최종 모델 저장
    final_ckpt_dir = os.path.join(ckpt_dir, 'final')
    os.makedirs(final_ckpt_dir, exist_ok=True)
    policy.save_pretrained(final_ckpt_dir)
    
    # 최종 손실 그래프 저장
    plt.figure()
    plt.plot(losses)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title(f'Training Loss for {action_type} (Final)')
    plt.savefig(os.path.join(final_ckpt_dir, 'loss_final.png'))
    plt.close()
    
    return losses


def evaluate_policy(policy, dataset, device, action_type, episode_index=0):
    """
    Evaluate policy on a specific episode
    
    Args:
        policy: Policy to evaluate
        dataset: Dataset to evaluate on
        device: Device to run evaluation on
        action_type: Type of action to evaluate with
        episode_index: Index of episode to evaluate on
    
    Returns:
        Ground truth actions and predicted actions
    """
    # Create a dataloader for the specific episode
    test_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        sampler=EpisodeSampler(dataset, episode_index),
        shuffle=False,
    )
    
    actions = []
    gt_actions = []
    images = []
    
    policy.eval()
    policy.reset()
    
    # Collect predictions
    print("Evaluating policy...")
    for batch in test_dataloader:
        # 이제 'action' 키를 직접 사용
        inp_batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            
        action = policy.select_action(inp_batch)
        actions.append(action)
        gt_actions.append(inp_batch["action"][:, 0, :])
        images.append(inp_batch["observation.image"] if "observation.image" in inp_batch else None)
    
    # Concatenate results
    if actions:
        actions = torch.cat(actions, dim=0)
        gt_actions = torch.cat(gt_actions, dim=0)
        print(f"Mean action error: {torch.mean(torch.abs(actions[:, :gt_actions.size(1)] - gt_actions)).item():.3f}")
        return gt_actions, actions
    else:
        print("No actions collected during evaluation")
        return None, None


def plot_results(gt_actions: torch.Tensor, pred_actions: torch.Tensor, save_dir: str):
    """
    Plot the evaluation results and save them to the specified directory.
    
    Args:
        gt_actions: Ground truth actions
        pred_actions: Predicted actions
        save_dir: Directory to save plots
    """
    if gt_actions is None or pred_actions is None:
        print("No actions to plot")
        return
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving plots to: {save_dir}")
    
    # Convert to numpy
    gt_np = gt_actions.cpu().detach().numpy()
    pred_np = pred_actions.cpu().detach().numpy()
    
    # Plot each action dimension
    action_dim = gt_np.shape[1]
    fig, axs = plt.subplots(action_dim, 1, figsize=(10, 2*action_dim))
    
    for i in range(action_dim):
        if action_dim == 1:
            ax = axs
        else:
            ax = axs[i]
        
        ax.plot(pred_np[:, i], label="prediction")
        ax.plot(gt_np[:, i], label="ground truth")
        ax.set_title(f"Action Dimension {i}")
        ax.legend()
    
    plt.tight_layout()
    action_plot_path = os.path.join(save_dir, 'action_comparison.png')
    plt.savefig(action_plot_path)
    print(f"Saved action comparison plot to: {action_plot_path}")
    plt.close()  # Close the figure to free memory
    
    # Plot error heatmap
    error = np.abs(pred_np[:, :gt_np.shape[1]] - gt_np)
    plt.figure(figsize=(10, 6))
    plt.imshow(error.T, aspect='auto', cmap='hot')
    plt.colorbar(label='Absolute Error')
    plt.xlabel('Time Step')
    plt.ylabel('Action Dimension')
    plt.title('Error Heatmap')
    heatmap_path = os.path.join(save_dir, 'error_heatmap.png')
    plt.savefig(heatmap_path)
    print(f"Saved error heatmap to: {heatmap_path}")
    plt.close()  # Close the figure to free memory
    
    # Plot error histogram
    plt.figure(figsize=(10, 6))
    plt.hist(error.flatten(), bins=50)
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    histogram_path = os.path.join(save_dir, 'error_histogram.png')
    plt.savefig(histogram_path)
    print(f"Saved error histogram to: {histogram_path}")
    plt.close()  # Close the figure to free memory


class EpisodeSampler(torch.utils.data.Sampler):
    """Sample frames from a specific episode"""
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


def main():
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description='Train the ACT model.')
    parser.add_argument('--action_type', type=str, choices=['joint', 'ee_pose', 'delta_q'], default='joint',
                        help='Type of action to train with')
    parser.add_argument('--load_ckpt', action='store_true', help='Whether to load from checkpoint')
    parser.add_argument('--num_epochs', type=int, default=3000, help='Number of epochs to train')
    args = parser.parse_args()

    # Use global variable to share action type
    global ACTION_TYPE
    ACTION_TYPE = args.action_type
    
    print(f"\n=== Training ACT model with action type: {ACTION_TYPE} ===\n")
    
    try:
        # 체크포인트 디렉토리 설정
        ckpt_dir = './ckpt/act_y'
        
        # 정책 생성 또는 로드
        policy, dataset_metadata, action_type_ckpt_dir = create_or_load_policy(
            ckpt_dir, action_type=ACTION_TYPE, load_ckpt=args.load_ckpt
        )
        
        # 데이터 준비
        dataset, dataloader = prepare_data('omy_pnp', policy, dataset_metadata, ACTION_TYPE)
        
        # 정책 훈련
        print("Training policy...")
        losses = train_policy(policy, dataset, dataloader, action_type_ckpt_dir, ACTION_TYPE, num_epochs=args.num_epochs)
        
        # 정책 평가
        print("Evaluating policy...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gt_actions, pred_actions = evaluate_policy(policy, dataset, device, ACTION_TYPE, episode_index=0)
        
        # 평가 결과 시각화 및 저장
        if gt_actions is not None and pred_actions is not None:
            print("Plotting evaluation results...")
            plot_results(gt_actions, pred_actions, os.path.join(action_type_ckpt_dir, 'evaluation_results'))
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
