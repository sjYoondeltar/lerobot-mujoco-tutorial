#!/usr/bin/env python3
# coding: utf-8

"""
Simple downsampling script without LeRobot imports
"""

import os
import pandas as pd
import numpy as np
import json
import shutil
from tqdm import tqdm
import argparse


def add_noise_to_action(action_array, noise_scale=0.02):
    """Add small random noise to action array"""
    if isinstance(action_array, (list, np.ndarray)):
        action_array = np.array(action_array, dtype=np.float32)  # Ensure consistent dtype
        noise = np.random.normal(0, noise_scale, action_array.shape).astype(np.float32)
        return action_array + noise
    return action_array


def downsample_dataset(input_root, output_root, action_type='joint', 
                      downsample_factor=5, perturbation_ratio=0.3):
    """
    Downsample dataset by processing parquet files directly
    """
    
    input_path = os.path.join(input_root, action_type)
    output_path = os.path.join(output_root, action_type)
    
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    # Check if input exists
    if not os.path.exists(input_path):
        print(f"Error: Input path {input_path} does not exist")
        return
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Copy metadata
    input_meta = os.path.join(input_path, "meta")
    output_meta = os.path.join(output_path, "meta")
    
    if os.path.exists(input_meta):
        if os.path.exists(output_meta):
            shutil.rmtree(output_meta)
        shutil.copytree(input_meta, output_meta)
        print("✓ Copied metadata")
    
    # Process data files
    input_data_dir = os.path.join(input_path, "data", "chunk-000")
    output_data_dir = os.path.join(output_path, "data", "chunk-000")
    os.makedirs(output_data_dir, exist_ok=True)
    
    if not os.path.exists(input_data_dir):
        print(f"Error: Input data directory {input_data_dir} does not exist")
        return
    
    # Get parquet files
    parquet_files = [f for f in os.listdir(input_data_dir) if f.endswith('.parquet')]
    parquet_files.sort()
    
    print(f"Found {len(parquet_files)} episode files")
    
    total_original = 0
    total_downsampled = 0
    
    # Process each episode
    for parquet_file in tqdm(parquet_files, desc="Processing episodes"):
        input_file = os.path.join(input_data_dir, parquet_file)
        output_file = os.path.join(output_data_dir, parquet_file)
        
        try:
            # Load episode
            df = pd.read_parquet(input_file)
            original_length = len(df)
            total_original += original_length
            
            # Downsample: take every nth frame
            downsampled_indices = list(range(0, len(df), downsample_factor))
            df_downsampled = df.iloc[downsampled_indices].copy().reset_index(drop=True)
            
            # Add perturbations to actions
            if 'action' in df_downsampled.columns:
                actions = df_downsampled['action'].tolist()
                for idx in range(len(actions)):
                    if np.random.random() < perturbation_ratio:
                        actions[idx] = add_noise_to_action(actions[idx])
                df_downsampled = df_downsampled.copy()
                df_downsampled['action'] = actions
            
            downsampled_length = len(df_downsampled)
            total_downsampled += downsampled_length
            
            # Save downsampled episode
            df_downsampled.to_parquet(output_file, index=False)
            
            print(f"  {parquet_file}: {original_length} -> {downsampled_length} frames")
            
        except Exception as e:
            print(f"Error processing {parquet_file}: {e}")
            continue
    
    print(f"\n📊 Summary:")
    print(f"  Original frames: {total_original}")
    print(f"  Downsampled frames: {total_downsampled}")
    print(f"  Reduction ratio: {total_downsampled/total_original:.2f}")
    
    # Update metadata
    update_info_json(output_path, len(parquet_files), total_downsampled)
    
    print(f"✅ Created downsampled dataset at {output_path}")


def update_info_json(dataset_path, num_episodes, total_frames):
    """Update info.json with new statistics"""
    info_file = os.path.join(dataset_path, "meta", "info.json")
    
    if os.path.exists(info_file):
        try:
            with open(info_file, 'r') as f:
                info = json.load(f)
            
            info['fps'] = 1  # Now 1Hz
            info['total_episodes'] = num_episodes
            info['total_frames'] = total_frames
            
            with open(info_file, 'w') as f:
                json.dump(info, f, indent=2)
            
            print(f"✓ Updated info.json: {num_episodes} episodes, {total_frames} frames at 1Hz")
            
        except Exception as e:
            print(f"Warning: Could not update info.json: {e}")


def main():
    parser = argparse.ArgumentParser(description='Simple dataset downsampling')
    parser.add_argument('--input_root', default='./demo_data_cube', help='Input dataset path')
    parser.add_argument('--output_root', default='./demo_data_cube_1hz', help='Output dataset path')
    parser.add_argument('--action_type', default='joint', choices=['joint', 'eef_pose', 'delta_q'], 
                       help='Action type to process')
    parser.add_argument('--downsample_factor', type=int, default=5, help='Downsample factor (5 = 5Hz->1Hz)')
    parser.add_argument('--perturbation_ratio', type=float, default=0.3, help='Ratio of frames to perturb')
    
    args = parser.parse_args()
    
    print("🚀 Starting dataset downsampling...")
    print(f"  Input: {args.input_root}")
    print(f"  Output: {args.output_root}")
    print(f"  Action type: {args.action_type}")
    print(f"  Downsample factor: {args.downsample_factor}")
    print(f"  Perturbation ratio: {args.perturbation_ratio}")
    print()
    
    downsample_dataset(
        input_root=args.input_root,
        output_root=args.output_root,
        action_type=args.action_type,
        downsample_factor=args.downsample_factor,
        perturbation_ratio=args.perturbation_ratio
    )


if __name__ == "__main__":
    main()
