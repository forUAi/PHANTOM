"""Per-channel quantization analysis."""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ChannelStats:
    """Statistics for a single channel."""
    index: int
    min_val: float
    max_val: float
    mean: float
    std: float
    zero_percent: float
    saturation_score: float
    optimal_scale: float
    optimal_zero_point: int


def analyze_per_channel(
    tensor: torch.Tensor,
    bits: int = 8,
    symmetric: bool = False
) -> Dict[str, any]:
    """
    Perform per-channel quantization analysis.
    
    Args:
        tensor: Input tensor (typically weights or activations)
        bits: Target quantization bits
        symmetric: Use symmetric quantization
    
    Returns:
        Dictionary with per-channel statistics
    """
    logger.debug(f"Analyzing tensor shape {tensor.shape} for {bits}-bit quantization")
    
    # Determine channel dimension
    if tensor.dim() == 4:  # Conv weights: [out_channels, in_channels, H, W]
        channel_dim = 0
    elif tensor.dim() == 2:  # Linear weights: [out_features, in_features]
        channel_dim = 0
    else:
        channel_dim = 1  # Default for other tensors
    
    num_channels = tensor.shape[channel_dim]
    channel_stats = []
    
    for ch_idx in range(num_channels):
        # Extract channel
        if channel_dim == 0:
            channel_data = tensor[ch_idx].flatten()
        elif channel_dim == 1:
            channel_data = tensor[:, ch_idx].flatten()
        else:
            channel_data = tensor.select(channel_dim, ch_idx).flatten()
        
        # Calculate statistics
        stats = ChannelStats(
            index=ch_idx,
            min_val=channel_data.min().item(),
            max_val=channel_data.max().item(),
            mean=channel_data.mean().item(),
            std=channel_data.std().item() if channel_data.numel() > 1 else 0.0,
            zero_percent=(channel_data == 0).float().mean().item() * 100,
            saturation_score=0.0,
            optimal_scale=1.0,
            optimal_zero_point=0
        )
        
        # Calculate optimal quantization parameters
        if symmetric:
            abs_max = max(abs(stats.min_val), abs(stats.max_val))
            stats.optimal_scale = abs_max / (2**(bits-1) - 1)
            stats.optimal_zero_point = 0
        else:
            stats.optimal_scale = (stats.max_val - stats.min_val) / (2**bits - 1)
            stats.optimal_zero_point = int(-stats.min_val / stats.optimal_scale)
        
        # Calculate saturation score
        if stats.optimal_scale > 0:
            quantized = torch.round(channel_data / stats.optimal_scale + stats.optimal_zero_point)
            qmin = 0 if not symmetric else -(2**(bits-1))
            qmax = 2**bits - 1 if not symmetric else 2**(bits-1) - 1
            clipped = torch.clamp(quantized, qmin, qmax)
            stats.saturation_score = (quantized != clipped).float().mean().item()
        
        channel_stats.append(stats)
    
    # Aggregate analysis
    analysis = {
        'per_channel_stats': channel_stats,
        'num_channels': num_channels,
        'bits': bits,
        'symmetric': symmetric,
        'worst_channels': _find_worst_channels(channel_stats),
        'scale_variance': np.var([s.optimal_scale for s in channel_stats]),
        'recommendations': _generate_channel_recommendations(channel_stats, bits)
    }
    
    return analysis


def _find_worst_channels(stats: List[ChannelStats], top_k: int = 5) -> List[int]:
    """Find channels with highest saturation."""
    sorted_channels = sorted(stats, key=lambda x: x.saturation_score, reverse=True)
    return [ch.index for ch in sorted_channels[:top_k]]


def _generate_channel_recommendations(
    stats: List[ChannelStats],
    bits: int
) -> List[str]:
    """Generate recommendations based on per-channel analysis."""
    recommendations = []
    
    # Check scale variance
    scales = [s.optimal_scale for s in stats]
    scale_variance = np.var(scales)
    
    if scale_variance > 0.1:
        recommendations.append(
            f"High scale variance ({scale_variance:.3f}) detected. "
            "Per-channel quantization strongly recommended."
        )
    
    # Check saturation
    high_saturation = [s for s in stats if s.saturation_score > 0.01]
    if high_saturation:
        recommendations.append(
            f"{len(high_saturation)} channels show >1% saturation at {bits}-bit. "
            "Consider increasing bit width or using learned quantization."
        )
    
    # Check dead channels
    dead_channels = [s for s in stats if s.zero_percent > 99]
    if dead_channels:
        recommendations.append(
            f"{len(dead_channels)} channels are nearly all zeros. "
            "Consider pruning or channel reduction."
        )
    
    return recommendations


def compare_quantization_schemes(
    tensor: torch.Tensor,
    schemes: List[Dict[str, any]]
) -> Dict[str, any]:
    """
    Compare different quantization schemes.
    
    Args:
        tensor: Input tensor
        schemes: List of scheme configurations
        
    Returns:
        Comparison results
    """
    results = []
    
    for scheme in schemes:
        bits = scheme.get('bits', 8)
        symmetric = scheme.get('symmetric', False)
        per_channel = scheme.get('per_channel', True)
        
        if per_channel:
            analysis = analyze_per_channel(tensor, bits, symmetric)
            mse = _calculate_quantization_error(tensor, analysis)
        else:
            # Per-tensor quantization
            min_val = tensor.min().item()
            max_val = tensor.max().item()
            
            if symmetric:
                scale = max(abs(min_val), abs(max_val)) / (2**(bits-1) - 1)
                zero_point = 0
            else:
                scale = (max_val - min_val) / (2**bits - 1)
                zero_point = int(-min_val / scale)
            
            quantized = torch.round(tensor / scale + zero_point)
            dequantized = (quantized - zero_point) * scale
            mse = torch.mean((tensor - dequantized) ** 2).item()
        
        results.append({
            'scheme': scheme,
            'mse': mse,
            'psnr': -10 * np.log10(mse) if mse > 0 else float('inf')
        })
    
    # Sort by MSE
    results.sort(key=lambda x: x['mse'])
    
    return {
        'best_scheme': results[0]['scheme'],
        'comparison': results
    }


def _calculate_quantization_error(
    tensor: torch.Tensor,
    analysis: Dict
) -> float:
    """Calculate MSE for per-channel quantization."""
    total_error = 0.0
    total_elements = 0
    
    for stats in analysis['per_channel_stats']:
        # Simplified calculation - real implementation would apply
        # per-channel quantization properly
        channel_size = tensor.shape[1:].numel() if tensor.dim() > 1 else 1
        channel_error = (stats.optimal_scale ** 2) / 12  # Uniform quantization noise
        total_error += channel_error * channel_size
        total_elements += channel_size
    
    return total_error / total_elements if total_elements > 0 else 0.0