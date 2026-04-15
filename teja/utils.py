"""
Teja Utilities
==============
Shared utility functions used across all stages.
"""

import torch


def get_device():
    """
    Detect the best available device for training.
    Priority: CUDA GPU > CPU

    Returns:
        torch.device: The device to use for tensors and model.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🔆 Teja using GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
    else:
        device = torch.device('cpu')
        print("⚠️  Teja using CPU (no CUDA GPU detected)")
        print("    Training will be slower. Consider installing PyTorch with CUDA.")
    return device


def count_parameters(model):
    """
    Count and display the total number of trainable parameters in a model.

    Args:
        model: A PyTorch nn.Module

    Returns:
        int: Total number of trainable parameters
    """
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if total < 1_000:
        print(f"📊 Model parameters: {total}")
    elif total < 1_000_000:
        print(f"📊 Model parameters: {total:,} ({total/1e3:.1f}K)")
    else:
        print(f"📊 Model parameters: {total:,} ({total/1e6:.1f}M)")

    return total


def print_banner(stage_name, stage_num):
    """Print a nice banner for each stage."""
    print()
    print("=" * 60)
    print(f"  🔆 TEJA — Stage {stage_num}: {stage_name}")
    print(f"  Built from zero. Trained to shine.")
    print("=" * 60)
    print()
