"""Test ROCm backend support."""

import torch
import pytest
import subprocess
from phantom_core.exec.targets.rocm import ROCmRunner


def is_rocm_available():
    """Check if ROCm is available."""
    if not torch.cuda.is_available():
        return False
    
    try:
        result = subprocess.run(
            ["rocm-smi"], capture_output=True, timeout=1
        )
        return result.returncode == 0
    except:
        return False


@pytest.mark.skipif(not is_rocm_available(), reason="ROCm not available")
def test_rocm_runner():
    """Test ROCm runner execution."""
    import torch.nn as nn
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3)
            self.bn = nn.BatchNorm2d(16)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))
    
    model = SimpleModel()
    inputs = torch.randn(1, 3, 32, 32)
    
    runner = ROCmRunner()
    outputs = runner.run_ops([], model, inputs)
    
    assert len(outputs) > 0
    assert outputs[0].device.type == 'cuda'  # ROCm uses CUDA API


@pytest.mark.skipif(not is_rocm_available(), reason="ROCm not available")
def test_rocm_mixed_precision():
    """Test ROCm mixed precision support."""
    import torch.nn as nn
    
    model = nn.Linear(100, 50)
    inputs = torch.randn(4, 100)
    
    runner = ROCmRunner()
    
    # Test fp16
    outputs_fp16 = runner.run_ops([], model, inputs, precision='fp16')
    assert outputs_fp16[0].dtype in [torch.float16, torch.float32]
    
    # Test bf16 if supported
    if torch.cuda.is_bf16_supported():
        outputs_bf16 = runner.run_ops([], model, inputs, precision='bf16')
        assert outputs_bf16[0].dtype in [torch.bfloat16, torch.float32]
