#!/usr/bin/env python3
"""
Multi-GPU training launcher script for efficient memory usage
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def get_gpu_count():
    """Get number of available GPUs"""
    try:
        result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                              capture_output=True, text=True, check=True)
        return len(result.stdout.strip().split('\n'))
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 0


def launch_distributed_training(config_path, num_gpus=None, port=29500):
    """Launch distributed training using torchrun"""
    
    if num_gpus is None:
        num_gpus = get_gpu_count()
    
    if num_gpus <= 1:
        print("Warning: Only 1 GPU detected or specified. Running single-GPU training...")
        cmd = [sys.executable, "train_model_multigpu.py", "--config-path", config_path]
    else:
        print(f"Launching distributed training on {num_gpus} GPUs...")
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={num_gpus}",
            f"--master_port={port}",
            "--nnodes=1",
            "train_model_multigpu.py",
            "--config-path", config_path
        ]
    
    # Set environment variables for optimal performance
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(num_gpus)),
        "NCCL_DEBUG": "INFO",
        "NCCL_SOCKET_IFNAME": "^docker0,lo",
        "OMP_NUM_THREADS": "1",  # Important for multi-GPU
        "MKL_NUM_THREADS": "1",
    })
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Environment variables:")
    for key, value in env.items():
        if key.startswith(("CUDA", "NCCL", "OMP", "MKL")):
            print(f"  {key}={value}")
    
    try:
        result = subprocess.run(cmd, env=env, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU training launcher")
    parser.add_argument("--config-path", type=str, required=True,
                       help="Path to the training config file")
    parser.add_argument("--num-gpus", type=int, default=None,
                       help="Number of GPUs to use (default: auto-detect)")
    parser.add_argument("--port", type=int, default=29500,
                       help="Master port for distributed training")
    
    args = parser.parse_args()
    
    # Validate config file exists
    config_path = Path(args.config_path)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1
    
    # Launch training
    return launch_distributed_training(
        config_path=str(config_path),
        num_gpus=args.num_gpus,
        port=args.port
    )


if __name__ == "__main__":
    sys.exit(main()) 