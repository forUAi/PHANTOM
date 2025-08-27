"""Test per-channel quantization analysis."""

import torch
import pytest
from phantom_core.analysis.per_channel import (
    analyze_per_channel,
    compare_quantization_schemes
)


def test_per_channel_analysis():
    """Test per-channel quantization analysis."""
    # Create tensor with varying channel statistics
    tensor = torch.randn(64, 128, 3, 3)  # Conv weight
    tensor[0] *= 10  # Make first channel very different
    tensor[1] *= 0.1  # Make second channel very small
    
    analysis = analyze_per_channel(tensor, bits=8, symmetric=False)
    
    assert analysis['num_channels'] == 64
    assert len(analysis['per_channel_stats']) == 64
    assert 'worst_channels' in analysis
    assert 'recommendations' in analysis
    
    # Check that different channels have different scales
    scales = [s.optimal_scale for s in analysis['per_channel_stats']]
    assert max(scales) / min(scales) > 10  # High variance expected


def test_quantization_scheme_comparison():
    """Test comparison of different quantization schemes."""
    tensor = torch.randn(32, 64, 3, 3)
    
    schemes = [
        {'bits': 8, 'symmetric': False, 'per_channel': True},
        {'bits': 8, 'symmetric': True, 'per_channel': True},
        {'bits': 4, 'symmetric': False, 'per_channel': True},
        {'bits': 8, 'symmetric': False, 'per_channel': False},
    ]
    
    comparison = compare_quantization_schemes(tensor, schemes)
    
    assert 'best_scheme' in comparison
    assert 'comparison' in comparison
    assert len(comparison['comparison']) == len(schemes)
    
    # Per-channel should generally be better than per-tensor
    best = comparison['best_scheme']
    assert best['per_channel'] == True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
    # PHANTOM v0.2 - Complete Production Implementation