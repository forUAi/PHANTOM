"""ROCm backend for AMD GPUs."""

import torch
import os
from typing import List, Optional
from ..base import TargetRunner
from ...utils.logging import get_logger

logger = get_logger(__name__)


class ROCmRunner(TargetRunner):
    """ROCm executor for AMD GPUs."""
    
    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("ROCm not available (uses CUDA API)")
        
        # Check if this is actually ROCm
        import subprocess
        try:
            result = subprocess.run(
                ["rocm-smi"], capture_output=True, text=True, timeout=1
            )
            if result.returncode != 0:
                raise RuntimeError("ROCm tools not found, might be NVIDIA CUDA")
        except:
            logger.warning("Could not verify ROCm installation")
    
    def run_ops(
        self,
        graph: List,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        stop_at: Optional[int] = None,
        precision: str = 'fp16',
        slice_spec: Optional[str] = None,
        capture_intermediates: bool = False
    ) -> List[torch.Tensor]:
        """Execute model on ROCm backend."""
        device = torch.device('cuda')  # ROCm uses CUDA API
        
        logger.debug(f"Running on ROCm with precision={precision}")
        
        # Set ROCm-specific environment variables
        os.environ['HIP_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
        os.environ['HIP_LAUNCH_BLOCKING'] = '1' if logger.level <= 10 else '0'  # Debug mode
        
        dtype_map = {
            'fp64': torch.float64,
            'fp32': torch.float32,
            'fp16': torch.float16,
            'bf16': torch.bfloat16
        }
        dtype = dtype_map.get(precision, torch.float16)
        
        # Move model and inputs to ROCm device
        model = model.to(device).eval()
        
        # Handle mixed precision for ROCm
        if dtype in [torch.float16, torch.bfloat16]:
            # ROCm-specific mixed precision handling
            try:
                import apex
                model = apex.amp.initialize(model, opt_level="O1")
                logger.debug("Using APEX for ROCm mixed precision")
            except ImportError:
                # Fallback to native PyTorch
                if dtype == torch.float16:
                    model = model.half()
                elif dtype == torch.bfloat16:
                    model = model.bfloat16()
        
        inputs = inputs.to(device).to(dtype)
        
        outputs = []
        intermediates = []
        
        # Execute with ROCm optimizations
        with torch.no_grad():
            # ROCm-specific kernel settings
            with torch.backends.cudnn.flags(
                enabled=True,
                benchmark=False,  # Disable for determinism
                deterministic=True
            ):
                if capture_intermediates and graph:
                    # Capture intermediates
                    for i, op in enumerate(graph):
                        if stop_at is not None and i >= stop_at:
                            break
                        
                        # Execute op and capture output
                        # Simplified for MVP - real implementation would execute graph ops
                        if i == 0:
                            output = model(inputs)
                        
                        intermediates.append({
                            'op_index': i,
                            'op_name': op.name if hasattr(op, 'name') else f"op_{i}",
                            'output': output.clone() if torch.is_tensor(output) else output
                        })
                else:
                    output = model(inputs)
        
        # Handle output format
        if isinstance(output, tuple):
            output = output[0]
        
        # Apply slice if specified
        output = self._apply_slice(output, slice_spec)
        outputs.append(output)
        
        # Log ROCm memory usage
        if torch.cuda.is_available():
            logger.debug(f"ROCm memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
            logger.debug(f"ROCm memory reserved: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
        
        if capture_intermediates:
            return outputs, intermediates
        return outputs