"""Hook-based capture for dynamic models."""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class HookOpInfo:
    """Information captured via hooks."""
    index: int
    name: str
    module_type: str
    input_shapes: List[Tuple]
    output_shapes: List[Tuple]
    input_dtypes: List[str]
    output_dtypes: List[str]
    extra_info: Dict[str, Any]


class HookCapture:
    """Capture execution via hooks for dynamic models."""
    
    def __init__(self):
        self.ops = []
        self.activations = []
        self.hooks = []
        self._op_index = 0
    
    def capture_with_hooks(
        self, 
        model: nn.Module, 
        inputs: torch.Tensor
    ) -> Tuple[List[HookOpInfo], List[Dict]]:
        """
        Capture model execution using hooks.
        
        Returns:
            Tuple of (ops_list, activations_dict)
        """
        logger.info("Starting hook-based capture")
        
        # Clear previous captures
        self.ops.clear()
        self.activations.clear()
        self._op_index = 0
        
        # Register hooks
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook = module.register_forward_hook(
                    self._make_hook(name, module)
                )
                self.hooks.append(hook)
        
        # Run model
        try:
            with torch.no_grad():
                model.eval()
                outputs = model(inputs)
        finally:
            # Clean up hooks
            for hook in self.hooks:
                hook.remove()
            self.hooks.clear()
        
        logger.info(f"Captured {len(self.ops)} operations")
        
        return self.ops, self.activations
    
    def _make_hook(self, name: str, module: nn.Module):
        """Create a hook function for a module."""
        def hook(module, inputs, outputs):
            # Process inputs
            input_shapes = []
            input_dtypes = []
            for inp in inputs:
                if torch.is_tensor(inp):
                    input_shapes.append(tuple(inp.shape))
                    input_dtypes.append(str(inp.dtype))
                else:
                    input_shapes.append(None)
                    input_dtypes.append(str(type(inp)))
            
            # Process outputs
            output_shapes = []
            output_dtypes = []
            if torch.is_tensor(outputs):
                output_shapes.append(tuple(outputs.shape))
                output_dtypes.append(str(outputs.dtype))
                out_to_save = outputs
            elif isinstance(outputs, (tuple, list)):
                for out in outputs:
                    if torch.is_tensor(out):
                        output_shapes.append(tuple(out.shape))
                        output_dtypes.append(str(out.dtype))
                    else:
                        output_shapes.append(None)
                        output_dtypes.append(str(type(out)))
                out_to_save = outputs[0] if outputs else None
            else:
                output_shapes.append(None)
                output_dtypes.append(str(type(outputs)))
                out_to_save = None
            
            # Create op info
            op_info = HookOpInfo(
                index=self._op_index,
                name=name,
                module_type=module.__class__.__name__,
                input_shapes=input_shapes,
                output_shapes=output_shapes,
                input_dtypes=input_dtypes,
                output_dtypes=output_dtypes,
                extra_info=self._extract_extra_info(module)
            )
            
            self.ops.append(op_info)
            
            # Save activation if tensor
            if out_to_save is not None and torch.is_tensor(out_to_save):
                self.activations.append({
                    'index': self._op_index,
                    'name': name,
                    'tensor': out_to_save.detach().clone()
                })
            
            self._op_index += 1
        
        return hook
    
    def _extract_extra_info(self, module: nn.Module) -> Dict[str, Any]:
        """Extract additional information from module."""
        info = {}
        
        # Layer-specific parameters
        if isinstance(module, nn.Conv2d):
            info['kernel_size'] = module.kernel_size
            info['stride'] = module.stride
            info['padding'] = module.padding
            info['groups'] = module.groups
        elif isinstance(module, nn.Linear):
            info['in_features'] = module.in_features
            info['out_features'] = module.out_features
        elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
            info['eps'] = module.eps
            info['momentum'] = getattr(module, 'momentum', None)
        elif isinstance(module, nn.Dropout):
            info['p'] = module.p
        
        return info