"""Target backend runners."""

import torch
from typing import Optional
from ..base import TargetRunner


def get_target_runner(target: str) -> TargetRunner:
    """
    Get appropriate runner for target backend.
    
    Args:
        target: Backend name (cuda/mps/rocm/cpu)
    
    Returns:
        TargetRunner instance
    """
    target = target.lower()
    
    if target == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        from .cuda import CUDARunner
        return CUDARunner()
    
    elif target == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        from .mps import MPSRunner
        return MPSRunner()
    
    elif target == "rocm":
        if not torch.cuda.is_available():  # ROCm uses CUDA API
            raise RuntimeError("ROCm requested but not available")
        from .rocm import ROCmRunner
        return ROCmRunner()
    
    elif target == "cpu":
        from ..ref_cpu import CPURunner
        return CPURunner()
    
    else:
        raise ValueError(f"Unknown target: {target}")


def get_available_targets() -> list:
    """Get list of available backends."""
    targets = ["cpu"]
    
    if torch.cuda.is_available():
        import subprocess
        try:
            # Check if ROCm
            result = subprocess.run(
                ["rocm-smi"], capture_output=True, text=True, timeout=1
            )
            if result.returncode == 0:
                targets.append("rocm")
            else:
                targets.append("cuda")
        except:
            targets.append("cuda")
    
    if torch.backends.mps.is_available():
        targets.append("mps")
    
    return targets