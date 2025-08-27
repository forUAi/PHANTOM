"""PHANTOM - Cross-accelerator deterministic debugger for AI models."""

__version__ = "0.2.0"

from .capture.fx_capture import trace_module
from .capture.hook_capture import HookCapture
from .report.html import Report
from .report.interactive import InteractiveReport
from .utils.logging import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)

def trace(model, inputs, target="cuda", tolerances=None, seed=1234, 
          capture_intermediates=False, use_hooks=False, interactive=False):
    """
    High-level API for tracing and comparing model execution.
    
    Args:
        model: PyTorch model to debug
        inputs: Input tensors
        target: Target backend (cuda/mps/rocm/cpu)
        tolerances: Dict with 'abs' and 'rel' tolerance values
        seed: Random seed for determinism
        capture_intermediates: Capture all intermediate activations
        use_hooks: Use hook-based capture for dynamic models
        interactive: Generate interactive Plotly report
    
    Returns:
        Report object with analysis results
    """
    from .utils.determinism import set_global_determinism
    from .exec.ref_cpu import CPURunner
    from .exec.targets import get_target_runner
    from .diff.metrics import exceeds_tolerance
    from .analysis.kernel_fusion import detect_fusion_patterns
    import torch
    
    logger.info(f"Starting PHANTOM trace v{__version__}")
    logger.info(f"Target: {target}, Tolerances: {tolerances}, Seed: {seed}")
    
    set_global_determinism(seed)
    
    if tolerances is None:
        tolerances = {"abs": 1e-3, "rel": 1e-3}
    
    # Choose capture method
    if use_hooks or not _can_use_fx(model):
        logger.info("Using hook-based capture")
        capture = HookCapture()
        graph, intermediates = capture.capture_with_hooks(model, inputs)
    else:
        logger.info("Using FX-based capture")
        graph = trace_module(model, inputs)
        intermediates = None
    
    # Execute on reference backend
    ref_runner = CPURunner()
    ref_outputs = ref_runner.run_ops(
        graph, model, inputs, 
        capture_intermediates=capture_intermediates
    )
    
    # Execute on target backend
    target_runner = get_target_runner(target)
    target_outputs = target_runner.run_ops(
        graph, model, inputs,
        capture_intermediates=capture_intermediates
    )
    
    # Detect kernel fusion patterns
    fusion_info = detect_fusion_patterns(graph, target) if graph else None
    
    # Create appropriate report type
    ReportClass = InteractiveReport if interactive else Report
    report = ReportClass()
    report.graph = graph
    report.ref_outputs = ref_outputs
    report.target_outputs = target_outputs
    report.tolerances = tolerances
    report.fusion_info = fusion_info
    report.intermediates = intermediates
    report.analyze()
    
    logger.info(f"Analysis complete. First bad op: {report.first_bad_op}")
    
    return report


def _can_use_fx(model):
    """Check if model is compatible with FX tracing."""
    import torch.nn as nn
    
    # Models with dynamic control flow can't use FX
    incompatible_modules = (
        nn.RNN, nn.LSTM, nn.GRU,  # RNNs have loops
        nn.MultiheadAttention,  # Has conditional logic
    )
    
    for module in model.modules():
        if isinstance(module, incompatible_modules):
            return False
    
    return True


__all__ = ["trace", "trace_module", "Report", "InteractiveReport", "__version__"]