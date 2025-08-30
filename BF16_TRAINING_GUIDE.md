# BF16 Mixed Precision Training Guide

This guide explains how to use BF16 (bfloat16) mixed precision training for Tensor Core acceleration.

## What is BF16?

BF16 (bfloat16) is a 16-bit floating point format that provides:
- **Tensor Core acceleration** on modern GPUs (A100, H100, RTX 30/40 series)
- **Faster training** with similar accuracy to FP32
- **Reduced memory usage** allowing larger batch sizes or models
- **Better numerical stability** compared to FP16

## How to Enable BF16

### Option 1: Use the configuration flag
Set `use_bf16: true` in your config file:

```yaml
# In experiment/conf/config.yaml
use_bf16: true  # Enable BF16 for Tensor Core acceleration
```

### Option 2: Use the BF16 experiment config
```bash
cd experiment
python train_lm.py experiment=bf16_train
```

### Option 3: Override from command line
```bash
cd experiment
python train_lm.py use_bf16=true
```

## What Changes Were Made

1. **Autocast Context**: Forward pass and loss computation use `torch.autocast` with `dtype=torch.bfloat16`
2. **Gradient Scaler**: Added `torch.cuda.amp.GradScaler` for numerical stability
3. **Mixed Precision Backward**: Gradients are scaled during backward pass to prevent underflow
4. **Validation**: Validation also uses autocast for consistent precision

## Performance Benefits

- **Speed**: 1.5-2x faster training on Tensor Core GPUs
- **Memory**: ~2x reduction in activation memory usage
- **Batch Size**: Can often double your batch size
- **Accuracy**: Maintains similar convergence to FP32

## Hardware Requirements

- **GPU**: NVIDIA A100, H100, or RTX 30/40 series with Tensor Cores
- **CUDA**: Version 11.0 or later
- **PyTorch**: Version 1.10 or later

## Troubleshooting

### If training becomes unstable:
1. Check gradient clipping is enabled: `gradient_clip_norm: 1.0`
2. Reduce learning rate slightly
3. Monitor for NaN values in logs

### If you see no speedup:
1. Ensure your GPU supports Tensor Cores
2. Check that your model dimensions are multiples of 8 (optimal for Tensor Cores)
3. Verify CUDA and PyTorch versions

## Disabling BF16

To disable BF16 and return to FP32:
```bash
cd experiment
python train_lm.py use_bf16=false
# or
python train_lm.py experiment=fp32_train
```
