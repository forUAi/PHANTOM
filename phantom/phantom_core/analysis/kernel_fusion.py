"""Kernel fusion detection and analysis."""

import torch
from typing import List, Dict, Any, Optional
from ..utils.logging import get_logger

logger = get_logger(__name__)


class FusionPattern:
    """Represents a kernel fusion pattern."""
    
    def __init__(self, name: str, ops: List[str], backend: str):
        self.name = name
        self.ops = ops  # List of operation types that get fused
        self.backend = backend
        self.confidence = 0.0
        self.impact = "unknown"


def detect_fusion_patterns(graph: List, target: str) -> Dict[str, Any]:
    """
    Detect potential kernel fusion patterns in the graph.
    
    Args:
        graph: List of operations
        target: Target backend
    
    Returns:
        Dictionary with fusion information
    """
    logger.info(f"Detecting fusion patterns for {target}")
    
    patterns = []
    fusion_opportunities = 0
    
    # Backend-specific fusion patterns
    if target == "cuda":
        patterns.extend(_detect_cuda_fusions(graph))
    elif target == "mps":
        patterns.extend(_detect_mps_fusions(graph))
    elif target == "rocm":
        patterns.extend(_detect_rocm_fusions(graph))
    
    # Analyze impact
    for pattern in patterns:
        pattern.impact = _estimate_fusion_impact(pattern, graph)
    
    # Count opportunities
    fusion_opportunities = sum(1 for p in patterns if p.confidence > 0.5)
    
    return {
        'patterns': patterns,
        'total_opportunities': fusion_opportunities,
        'backend': target,
        'recommendations': _generate_fusion_recommendations(patterns, target)
    }


def _detect_cuda_fusions(graph: List) -> List[FusionPattern]:
    """Detect CUDA-specific fusion patterns."""
    patterns = []
    
    # Common CUDA fusion patterns
    fusion_rules = [
        {
            'name': 'Conv-BN-ReLU',
            'sequence': ['Conv2d', 'BatchNorm', 'ReLU'],
            'confidence': 0.9
        },
        {
            'name': 'Linear-GELU',
            'sequence': ['Linear', 'GELU'],
            'confidence': 0.8
        },
        {
            'name': 'LayerNorm-Linear',
            'sequence': ['LayerNorm', 'Linear'],
            'confidence': 0.7
        },
        {
            'name': 'Matmul-Add',
            'sequence': ['matmul', 'add'],
            'confidence': 0.85
        }
    ]
    
    # Scan graph for patterns
    for i in range(len(graph) - 2):
        for rule in fusion_rules:
            if _matches_sequence(graph[i:], rule['sequence']):
                pattern = FusionPattern(
                    name=rule['name'],
                    ops=rule['sequence'],
                    backend='cuda'
                )
                pattern.confidence = rule['confidence']
                patterns.append(pattern)
                logger.debug(f"Found CUDA fusion pattern: {rule['name']}")
    
    return patterns


def _detect_mps_fusions(graph: List) -> List[FusionPattern]:
    """Detect MPS-specific fusion patterns."""
    patterns = []
    
    # MPS has different fusion capabilities
    fusion_rules = [
        {
            'name': 'Conv-Activation',
            'sequence': ['Conv2d', 'ReLU'],
            'confidence': 0.8
        },
        {
            'name': 'Matmul-Bias',
            'sequence': ['matmul', 'add'],
            'confidence': 0.7
        }
    ]
    
    for i in range(len(graph) - 1):
        for rule in fusion_rules:
            if _matches_sequence(graph[i:], rule['sequence']):
                pattern = FusionPattern(
                    name=rule['name'],
                    ops=rule['sequence'],
                    backend='mps'
                )
                pattern.confidence = rule['confidence']
                patterns.append(pattern)
                logger.debug(f"Found MPS fusion pattern: {rule['name']}")
    
    return patterns


def _detect_rocm_fusions(graph: List) -> List[FusionPattern]:
    """Detect ROCm-specific fusion patterns."""
    patterns = []
    
    # ROCm MIOpen fusion patterns
    fusion_rules = [
        {
            'name': 'Conv-BN-Activation',
            'sequence': ['Conv2d', 'BatchNorm', 'ReLU'],
            'confidence': 0.85
        },
        {
            'name': 'GEMM-Activation',
            'sequence': ['Linear', 'ReLU'],
            'confidence': 0.75
        }
    ]
    
    for i in range(len(graph) - 2):
        for rule in fusion_rules:
            if _matches_sequence(graph[i:], rule['sequence']):
                pattern = FusionPattern(
                    name=rule['name'],
                    ops=rule['sequence'],
                    backend='rocm'
                )
                pattern.confidence = rule['confidence']
                patterns.append(pattern)
                logger.debug(f"Found ROCm fusion pattern: {rule['name']}")
    
    return patterns


def _matches_sequence(graph_slice: List, sequence: List[str]) -> bool:
    """Check if graph slice matches operation sequence."""
    if len(graph_slice) < len(sequence):
        return False
    
    for i, expected_op in enumerate(sequence):
        if i >= len(graph_slice):
            return False
        
        actual_op = graph_slice[i]
        
        # Check operation type (flexible matching)
        if hasattr(actual_op, 'target'):
            op_name = str(actual_op.target)
        elif hasattr(actual_op, 'name'):
            op_name = actual_op.name
        elif hasattr(actual_op, 'module_type'):
            op_name = actual_op.module_type
        else:
            op_name = str(actual_op)
        
        if expected_op.lower() not in op_name.lower():
            return False
    
    return True


def _estimate_fusion_impact(pattern: FusionPattern, graph: List) -> str:
    """Estimate the performance impact of a fusion."""
    # Simple heuristic based on pattern type
    high_impact = ['Conv-BN-ReLU', 'Conv-BN-Activation', 'Matmul-Add']
    medium_impact = ['Linear-GELU', 'LayerNorm-Linear', 'GEMM-Activation']
    
    if pattern.name in high_impact:
        return "high"
    elif pattern.name in medium_impact:
        return "medium"
    else:
        return "low"


def _generate_fusion_recommendations(patterns: List[FusionPattern], target: str) -> List[str]:
    """Generate recommendations based on detected patterns."""
    recommendations = []
    
    high_impact_count = sum(1 for p in patterns if p.impact == "high")
    
    if high_impact_count > 0:
        recommendations.append(
            f"Found {high_impact_count} high-impact fusion opportunities on {target}. "
            "These may cause numerical differences compared to CPU execution."
        )
    
    if target == "cuda":
        recommendations.append(
            "Consider using torch.jit.script() or torch.compile() for better fusion."
        )
    elif target == "mps":
        recommendations.append(
            "MPS fusion is automatic but limited. Consider batch size tuning."
        )
    elif target == "rocm":
        recommendations.append(
            "ROCm MIOpen fusions can be tuned via environment variables."
        )
    
    # Check for fusion-breaking patterns
    for i in range(len(patterns) - 1):
        if patterns[i].confidence > 0.7 and patterns[i+1].confidence < 0.3:
            recommendations.append(
                f"Potential fusion break between {patterns[i].name} and next operation. "
                "Consider reordering operations if possible."
            )
    
    return recommendations