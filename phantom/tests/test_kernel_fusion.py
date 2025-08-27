"""Test kernel fusion detection."""

import pytest
from phantom_core.analysis.kernel_fusion import (
    detect_fusion_patterns,
    FusionPattern
)


def test_cuda_fusion_detection():
    """Test CUDA fusion pattern detection."""
    # Mock graph with fusable patterns
    class MockOp:
        def __init__(self, name, target):
            self.name = name
            self.target = target
            self.module_type = target
    
    graph = [
        MockOp("conv1", "Conv2d"),
        MockOp("bn1", "BatchNorm"),
        MockOp("relu1", "ReLU"),
        MockOp("linear1", "Linear"),
        MockOp("gelu1", "GELU"),
    ]
    
    fusion_info = detect_fusion_patterns(graph, "cuda")
    
    assert fusion_info['backend'] == 'cuda'
    assert fusion_info['total_opportunities'] > 0
    assert len(fusion_info['patterns']) > 0
    
    # Check for Conv-BN-ReLU pattern
    patterns = fusion_info['patterns']
    conv_bn_relu = [p for p in patterns if p.name == 'Conv-BN-ReLU']
    assert len(conv_bn_relu) > 0


def test_mps_fusion_detection():
    """Test MPS fusion pattern detection."""
    class MockOp:
        def __init__(self, name, target):
            self.name = name
            self.target = target
    
    graph = [
        MockOp("conv1", "Conv2d"),
        MockOp("relu1", "ReLU"),
        MockOp("matmul1", "matmul"),
        MockOp("add1", "add"),
    ]
    
    fusion_info = detect_fusion_patterns(graph, "mps")
    
    assert fusion_info['backend'] == 'mps'
    # MPS has different fusion patterns
    assert 'recommendations' in fusion_info