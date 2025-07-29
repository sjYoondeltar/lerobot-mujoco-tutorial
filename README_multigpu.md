# Multi-GPU Training Guide

This guide provides efficient multi-GPU training setup using **DDP (Distributed Data Parallel)** for SmolVLA models.

## 🚀 Key Features

### Distributed Training
- **DDP (Distributed Data Parallel)**: Stable and reliable multi-GPU training
- **Model Synchronization**: Automatic parameter synchronization across GPUs
- **Gradient Accumulation**: Efficient gradient communication between processes
- **BFloat16 Compatibility**: Optimized for SmolVLA's BFloat16 parameters

### Performance Optimization
- **Automatic Model State Sync**: Ensures consistent initialization across processes
- **Gradient Bucketing**: Memory-efficient gradient communication
- **Distributed Sampling**: Each GPU processes different data batches
- **Optimized Data Loading**: Distributed data sampling with multiple workers

## 📋 System Requirements

- **GPU**: NVIDIA RTX A6000 x4 (44.5GB each)
- **CUDA**: 12.4+
- **PyTorch**: 2.6.0+ (DDP support)
- **NCCL**: For distributed communication
- **LeRobot**: Latest version with SmolVLA support

## 🔧 Configuration

### 1. Environment Variables (Auto-configured)
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

### 2. Training Configuration

**Recommended Settings for SmolVLA**:
```yaml
policy:
  type: smolvla
  use_amp: false  # Disabled for BFloat16 compatibility
  
batch_size: 256    # Total: 64 per GPU on 4 GPUs
num_workers: 16    # Total: 4 per GPU
seed: 42          # Ensures consistent model initialization

distributed:
  backend: nccl
  find_unused_parameters: false

memory:
  gradient_checkpointing: true
  pin_memory: true
  persistent_workers: true
```

## 🚀 How to Run

### Method 1: Auto GPU Detection (Recommended)
```bash
python launch_multigpu.py --config-path smolvla_omy_multigpu.yaml
```

### Method 2: Specify GPU Count
```bash
python launch_multigpu.py --config-path smolvla_omy_multigpu.yaml --num-gpus 4
```

### Method 3: Manual Launch (Advanced)
```bash
python -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_port=29500 \
    train_model_multigpu.py \
    --config_path smolvla_omy_multigpu.yaml
```

## 📊 Performance Comparison

| Setup | GPU Memory Usage | Training Speed | Batch Size | Notes |
|-------|------------------|----------------|------------|-------|
| Single GPU | ~40GB | 1x | 64 | Baseline |
| **DDP (4 GPU)** | **~40GB x4** | **~3.5x** | **256** | **Stable & Reliable** |

## 🔍 Monitoring

### Check GPU Memory Usage
```bash
watch -n 1 nvidia-smi
```

### Check Training Progress
```bash
# Check if processes are running
ps aux | grep train_model_multigpu

# Monitor training logs (if available)
tail -f ckpt/smolvla_omy_multigpu/logs/train.log
```

## 🐛 Troubleshooting

### Common Issues & Solutions

#### 1. **"Must flatten tensors with uniform dtype" Error**
- **Cause**: FSDP compatibility issue with BFloat16 parameters
- **Solution**: ✅ **Fixed** - Now uses DDP instead of FSDP

#### 2. **"_amp_foreach_non_finite_check_and_unscale_cuda not implemented for BFloat16"**
- **Cause**: AMP incompatibility with BFloat16
- **Solution**: ✅ **Fixed** - AMP automatically disabled for BFloat16 models

#### 3. **Parameter Shape Mismatch Between Processes**
- **Cause**: Inconsistent model initialization across processes
- **Solution**: ✅ **Fixed** - Added model state synchronization

#### 4. **Out of Memory Errors**
1. Reduce `batch_size` (256 → 128 → 64)
2. Reduce `num_workers` (16 → 8 → 4)
3. Ensure `gradient_checkpointing: true`

#### 5. **Communication Errors (NCCL)**
1. Set `NCCL_DEBUG=INFO` for debugging
2. Check firewall/network settings
3. Change `master_port` (29500 → 29501)

#### 6. **Training Hangs or Slow Speed**
1. Check all GPUs are being utilized (`nvidia-smi`)
2. Verify `num_workers` is not too high
3. Ensure `pin_memory: true`

## 💡 Tips & Best Practices

1. **Batch Size**: Total batch size is automatically split across GPUs
2. **Checkpoints**: Saved only on main process (GPU 0) to avoid conflicts
3. **Logging**: Output only from main process to prevent duplicate logs
4. **Seed**: Always set a fixed seed for reproducible model initialization
5. **AMP**: Automatically disabled for SmolVLA to ensure compatibility

## 🎯 Validated Configuration

**Tested and Working on RTX A6000 x4**:
```yaml
dataset:
  repo_id: omy_pnp_language
  root: ./merged_omy_language_data

policy:
  type: smolvla
  chunk_size: 5
  n_action_steps: 5
  use_amp: false  # Auto-disabled for BFloat16

batch_size: 256      # 64 per GPU
num_workers: 16      # 4 per GPU
seed: 42
steps: 20_000

distributed:
  backend: nccl
  find_unused_parameters: false

memory:
  gradient_checkpointing: true
  pin_memory: true
  persistent_workers: true
```

## ✅ **Status: Ready for Production**

This configuration has been tested and validated for **stable multi-GPU training** with SmolVLA models. All major compatibility issues have been resolved.

**Key Achievements**:
- ✅ DDP-based distributed training
- ✅ BFloat16 compatibility 
- ✅ Automatic model synchronization
- ✅ Optimized memory usage
- ✅ ~3.5x training speedup on 4 GPUs 