"""Metal Performance Shaders (MPS) backend for Apple Silicon."""

import torch
from typing import List, Optional
from ..base import TargetRunner
from ...utils.logging import get_logger

logger = get_logger(__name__)


class MPSRunner(TargetRunner):
    """MPS executor for Apple Silicon GPUs."""
    
    def __init__(self):
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS not available on this system")
        
        if not torch.backends.mps.is_built():
            raise RuntimeError("PyTorch not compiled with MPS support")
    
    def run_ops(
        self,
        graph: List,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        stop_at: Optional[int] = None,
        precision: str = 'fp32',  # MPS doesn't support fp16 well yet
        slice_spec: Optional[str] = None,
        capture_intermediates: bool = False
    ) -> List[torch.Tensor]:
        """Execute model on MPS backend."""
        device = torch.device('mps')
        
        logger.debug(f"Running on MPS with precision={precision}")
        
        # MPS has limited dtype support
        if precision == 'fp16':
            logger.warning("MPS has limited fp16 support, using fp32 instead")
            precision = 'fp32'
        
        dtype_map = {
            'fp32': torch.float32,
            'fp16': torch.float16,  # Limited support
            'bf16': torch.bfloat16,  # Not supported
        }
        
        if precision == 'bf16':
            logger.warning("MPS doesn't support bfloat16, using fp32")
            dtype = torch.float32
        else:
            dtype = dtype_map.get(precision, torch.float32)
        
        # Move model and inputs to MPS
        model = model.to(device).eval()
        inputs = inputs.to(device).to(dtype)
        
        outputs = []
        intermediates = []
        
        # Execute with MPS optimizations
        with torch.no_grad():
            # MPS-specific optimizations
            with torch.mps.profiler.profile() as prof:
                if capture_intermediates and graph:
                    # Hook-based intermediate capture
                    hooks = []
                    def make_hook(name):
                        def hook(module, input, output):
                            intermediates.append({
                                'name': name,
                                'output': output.clone() if torch.is_tensor(output) else output
                            })
                        return hook
                    
                    for name, module in model.named_modules():
                        hooks.append(module.register_forward_hook(make_hook(name)))
                    
                    output = model(inputs)
                    
                    for hook in hooks:
                        hook.remove()
                else:
                    output = model(inputs)
            
            # Log MPS performance metrics
            if hasattr(prof, 'key_averages'):
                for evt in prof.key_averages():
                    if evt.self_mps_time_total > 0:
                        logger.debug(f"MPS kernel: {evt.key}, Time: {evt.self_mps_time_total:.3f}ms")
        
        if isinstance(output, tuple):
            output = output[0]
        
        # Apply slice if specified
        output = self._apply_slice(output, slice_spec)
        outputs.append(output)
        
        if capture_intermediates:
            return outputs, intermediates
        return outputs