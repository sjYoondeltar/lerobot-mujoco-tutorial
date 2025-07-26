#!/usr/bin/env python3
"""
Direct merge of two LeRobot datasets without HF conversion
"""

import pandas as pd
import json
import os
import shutil
from pathlib import Path

def load_episodes_jsonl(file_path):
    """Load episodes.jsonl file"""
    episodes = []
    with open(file_path, 'r') as f:
        for line in f:
            episodes.append(json.loads(line.strip()))
    return episodes

def load_info_json(file_path):
    """Load info.json file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def direct_merge_datasets():
    """Direct merge without HF conversion"""
    
    print("=== Direct LeRobot Dataset Merge ===")
    
    # Paths
    omy_path = "omy_pnp_language"
    demo_path = "demo_data_language2" 
    output_path = "merged_omy_language_data"
    
    # Check input datasets
    if not os.path.exists(omy_path):
        print(f"Error: {omy_path} not found!")
        return
    if not os.path.exists(demo_path):
        print(f"Error: {demo_path} not found!")
        return
    
    # Remove existing output
    if os.path.exists(output_path):
        print(f"Removing existing {output_path}...")
        shutil.rmtree(output_path)
    
    # Create output structure
    os.makedirs(f"{output_path}/data/chunk-000", exist_ok=True)
    os.makedirs(f"{output_path}/meta", exist_ok=True)
    
    print("Step 1: Loading metadata...")
    
    # Load omy metadata
    omy_episodes = load_episodes_jsonl(f"{omy_path}/meta/episodes.jsonl")
    omy_info = load_info_json(f"{omy_path}/meta/info.json")
    
    # Load demo metadata  
    demo_episodes = load_episodes_jsonl(f"{demo_path}/meta/episodes.jsonl")
    demo_info = load_info_json(f"{demo_path}/meta/info.json")
    
    print(f"Omy dataset: {len(omy_episodes)} episodes")
    print(f"Demo dataset: {len(demo_episodes)} episodes")
    
    print("Step 2: Copying omy episodes (0-19)...")
    
    # Copy omy episodes directly
    for i, episode in enumerate(omy_episodes):
        src_file = f"{omy_path}/data/chunk-000/episode_{i:06d}.parquet"
        dst_file = f"{output_path}/data/chunk-000/episode_{i:06d}.parquet"
        
        if os.path.exists(src_file):
            shutil.copy2(src_file, dst_file)
            print(f"  Copied episode_{i:06d}.parquet")
        else:
            print(f"  Warning: {src_file} not found!")
    
    print("Step 3: Copying and updating demo episodes (20-39)...")
    
    # Copy demo episodes with updated indices
    for i, episode in enumerate(demo_episodes):
        new_idx = i + 20
        src_file = f"{demo_path}/data/chunk-000/episode_{i:06d}.parquet"
        dst_file = f"{output_path}/data/chunk-000/episode_{new_idx:06d}.parquet"
        
        if os.path.exists(src_file):
            # Load, update episode_index, and save
            df = pd.read_parquet(src_file)
            if 'episode_index' in df.columns:
                df['episode_index'] = new_idx
            df.to_parquet(dst_file, compression='snappy')
            print(f"  Updated episode_{i:06d}.parquet -> episode_{new_idx:06d}.parquet")
        else:
            print(f"  Warning: {src_file} not found!")
    
    print("Step 4: Creating merged metadata...")
    
    # Create merged episodes.jsonl
    merged_episodes = []
    
    # Add omy episodes (0-19)
    for i, episode in enumerate(omy_episodes):
        merged_episode = episode.copy()
        merged_episode['episode_index'] = i
        merged_episodes.append(merged_episode)
    
    # Add demo episodes (20-39)
    for i, episode in enumerate(demo_episodes):
        merged_episode = episode.copy()
        merged_episode['episode_index'] = i + 20
        merged_episodes.append(merged_episode)
    
    # Save episodes.jsonl
    with open(f"{output_path}/meta/episodes.jsonl", 'w') as f:
        for episode in merged_episodes:
            f.write(json.dumps(episode) + '\n')
    
    # Create merged info.json
    total_frames = omy_info['total_frames'] + demo_info['total_frames']
    
    merged_info = omy_info.copy()
    merged_info.update({
        'total_episodes': 40,
        'total_frames': total_frames,
        'total_tasks': 2,
        'total_videos': 0,
        'total_chunks': 1
    })
    
    with open(f"{output_path}/meta/info.json", 'w') as f:
        json.dump(merged_info, f, indent=2)
    
    # Merge tasks.jsonl
    omy_tasks = []
    with open(f"{omy_path}/meta/tasks.jsonl", 'r') as f:
        for line in f:
            omy_tasks.append(json.loads(line.strip()))
    
    demo_tasks = []
    with open(f"{demo_path}/meta/tasks.jsonl", 'r') as f:
        for line in f:
            demo_tasks.append(json.loads(line.strip()))
    
    # Combine unique tasks
    all_tasks = []
    task_names = set()
    
    for task in omy_tasks + demo_tasks:
        if task['task_index'] not in task_names:
            all_tasks.append(task)
            task_names.add(task['task_index'])
    
    with open(f"{output_path}/meta/tasks.jsonl", 'w') as f:
        for task in all_tasks:
            f.write(json.dumps(task) + '\n')
    
    print("Step 5: Creating episodes_stats.jsonl...")
    
    # Create episodes_stats by copying from source datasets
    episodes_stats = []
    
    # Add omy stats (0-19)
    if os.path.exists(f"{omy_path}/meta/episodes_stats.jsonl"):
        with open(f"{omy_path}/meta/episodes_stats.jsonl", 'r') as f:
            for i, line in enumerate(f):
                if line.strip():
                    stats = json.loads(line.strip())
                    stats['episode_index'] = i
                    episodes_stats.append(stats)
    
    # Add demo stats (20-39)
    if os.path.exists(f"{demo_path}/meta/episodes_stats.jsonl"):
        with open(f"{demo_path}/meta/episodes_stats.jsonl", 'r') as f:
            for i, line in enumerate(f):
                if line.strip():
                    stats = json.loads(line.strip())
                    stats['episode_index'] = i + 20
                    # Update internal episode_index in stats
                    if 'stats' in stats and isinstance(stats['stats'], dict):
                        for key, stat_data in stats['stats'].items():
                            if isinstance(stat_data, dict) and 'episode_index' in stat_data:
                                stat_data['episode_index'] = i + 20
                    episodes_stats.append(stats)
    
    # Save episodes_stats.jsonl
    with open(f"{output_path}/meta/episodes_stats.jsonl", 'w') as f:
        for stats in episodes_stats:
            f.write(json.dumps(stats) + '\n')
    
    print(f"\n=== Merge Completed ===")
    print(f"Output: {output_path}")
    print(f"Total episodes: 40")
    print(f"Total frames: {total_frames}")
    print(f"Episodes 0-19: from {omy_path}")
    print(f"Episodes 20-39: from {demo_path}")
    
    # Verify output
    episode_files = len([f for f in os.listdir(f"{output_path}/data/chunk-000") 
                        if f.startswith('episode_') and f.endswith('.parquet')])
    print(f"Episode files created: {episode_files}")
    
    return output_path

if __name__ == "__main__":
    direct_merge_datasets() 