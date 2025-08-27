import torch
import pytest
from phantom_core.exec.targets.mps import MPSRunner
from phantom_core import trace


@pytest.mark.skipif(not torch.backends.mps.is_available(), 
                    reason="MPS not available")
def test_mps_runner():
    """Test MPS runner execution."""
    import torch.nn as nn
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            return self.relu(self.linear(x))
    
    model = SimpleModel()
    inputs = torch.randn(2, 10)
    
    runner = MPSRunner()
    outputs = runner.run_ops([], model, inputs)
    
    assert len(outputs) > 0
    assert outputs[0].device.type == 'mps'


@pytest.mark.skipif(not torch.backends.mps.is_available(),
                    reason="MPS not available")
def test_mps_precision_limitations():
    """Test MPS precision handling."""
    import torch.nn as nn
    
    model = nn.Linear(10, 5)
    inputs = torch.randn(2, 10)
    
    runner = MPSRunner()
    
    # MPS should handle fp32 well
    outputs_fp32 = runner.run_ops([], model, inputs, precision='fp32')
    assert outputs_fp32[0].dtype == torch.float32
    
    # MPS has limited fp16 support - should warn
    with pytest.warns(None) as record:
        outputs_fp16 = runner.run_ops([], model, inputs, precision='fp16')
    
    # Should fallback or warn about limitations
    assert len(record) > 0 or outputs_fp16[0].dtype == torch.float32
