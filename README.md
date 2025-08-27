# 🚀 PHANTOM - Cross-Accelerator Deterministic Debugger for AI Models

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4+](https://img.shields.io/badge/pytorch-2.4+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

PHANTOM is a powerful debugging tool that helps ML engineers identify and fix numerical differences when running AI models across different hardware accelerators (CPU vs CUDA, fp64 vs fp16). It provides layer-by-layer comparison, automatic root cause analysis, and deterministic replay capabilities.

## 🎯 Why PHANTOM?

Ever wondered why your model produces different results on CPU vs GPU? Or why fp16 training diverges? PHANTOM answers these questions by:

- 📊 **Pinpointing the exact operation** where numerical drift begins
- 🔍 **Analyzing root causes** (precision loss, kernel fusion, quantization effects)
- 🔄 **Providing deterministic replay** to reproduce issues consistently
- 📈 **Generating beautiful reports** with heatmaps and detailed metrics

## ✨ Key Features

### 🎯 Precision Debugging
- **Dual Execution**: Run models on reference (CPU/fp64) and target (CUDA/fp16) backends simultaneously
- **Binary Search**: O(log N) algorithm to find the first operation causing drift
- **Multiple Metrics**: Absolute error, relative error, ULP distance, statistical analysis

### 🔬 Deep Analysis
- **Activation Statistics**: Min/max, mean/std, zero/inf/nan percentages
- **Quantization Analysis**: Saturation detection, range utilization
- **Difference Heatmaps**: Visual representation of numerical differences
- **Attribution Hints**: Likely causes (precision, fusion, epsilon values)

### 🔄 Reproducibility
- **Deterministic Execution**: Controls all RNG sources (Python, NumPy, PyTorch, CUDA)
- **Replay Artifacts**: JSON files to reproduce exact debugging sessions
- **Environment Capture**: Complete system configuration recording

### 📊 Reporting
- **HTML Reports**: Self-contained, shareable debugging reports
- **Interactive Visualizations**: Heatmaps, error distribution plots
- **Operation Timeline**: Graph execution with error propagation
- **One-Click Replay**: Command snippets for reproduction

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/phantom.git
cd phantom

# Install with pip (recommended)
pip install -e .

# Or install with dependencies
pip install -e ".[dev,vision]"
```

### Basic Usage

```bash
# Debug a ResNet model
phantom trace examples/resnet_demo.py --target=cuda --abs_tol=1e-3 --rel_tol=1e-3 --out reports/resnet.html

# Debug a transformer model
phantom trace examples/tinyllama_demo.py --target=cuda --abs_tol=1e-4 --rel_tol=1e-4 --out reports/llm.html

# View the report
open reports/resnet.html

# Replay a previous debugging session
phantom replay runs/2025-08-25T12-00-00.json
```

### Python API

```python
from phantom import trace

# Load your model and inputs
import torch
from torchvision.models import resnet50
model = resnet50(pretrained=True).eval()
inputs = torch.randn(1, 3, 224, 224)

# Trace and compare
report = trace(
    model, 
    inputs, 
    target="cuda",  # or "cpu", "mps", "rocm" 
    tolerances={"abs": 1e-3, "rel": 1e-3},
    seed=1234
)

# Save report
report.save_html("reports/my_model.html")

# Access metrics programmatically
print(f"Max absolute error: {report.metrics['max_abs_error']}")
print(f"First bad operation: {report.first_bad_op}")
```

## 📖 Documentation

### Creating Custom Examples

Your model module should expose two functions:

```python
# my_model_demo.py
import torch
import torch.nn as nn

def build_model():
    """Build and return your model in eval mode."""
    model = YourModel()
    return model.eval()

def sample_inputs():
    """Return sample inputs for the model."""
    return torch.randn(batch_size, *input_shape)
```

### Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--target` | Target backend (cuda/cpu/mps/rocm) | cuda |
| `--abs_tol` | Absolute error tolerance | 1e-3 |
| `--rel_tol` | Relative error tolerance | 1e-3 |
| `--precision` | Target precision (fp16/bf16/fp32) | fp16 |
| `--slice` | Memory-saving slice spec | None |
| `--seed` | Random seed for determinism | 1234 |
| `--out` | Output HTML report path | reports/run.html |

### Understanding Tolerances

PHANTOM uses two tolerance thresholds:

- **Absolute Tolerance**: Maximum acceptable absolute difference `|a - b|`
- **Relative Tolerance**: Maximum acceptable relative difference `|a - b| / |b|`

An operation is flagged as "bad" if **either** threshold is exceeded.

### Memory Optimization

For large models, use slice specifications to reduce memory usage:

```bash
# Only compare first 64x64 elements of each tensor
phantom trace model.py --slice=":,:64,:64"

# Sample every 4th element
phantom trace model.py --slice="::4,::4"
```

## 🔬 How It Works

### 1. Graph Capture
PHANTOM uses PyTorch FX to capture the computation graph:
```python
tracer = torch.fx.Tracer()
graph = tracer.trace(model)
```

### 2. Dual Execution
The model runs on two backends simultaneously:
- **Reference**: CPU with fp64 (maximum precision)
- **Target**: CUDA with fp16/bf16 (production setting)

### 3. Binary Search
When outputs differ, PHANTOM uses binary search to find the first problematic operation:
```
Initial: 100 ops, output differs
Step 1: Check op 50 → outputs match
Step 2: Check op 75 → outputs differ  
Step 3: Check op 62 → outputs match
Step 4: Check op 68 → outputs differ
...
Result: Op 66 is the first bad operation
```

### 4. Root Cause Analysis
For the identified operation, PHANTOM analyzes:
- Precision loss patterns
- Activation statistics
- Kernel fusion indicators
- Numerical stability metrics

## 📊 Example Report

<details>
<summary>Click to see example HTML report content</summary>

```
🚀 PHANTOM Debug Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Summary
Status: ❌ Drift Detected
First Bad Op: #42 - LayerNorm
Operation: aten::layer_norm
Location: transformer.encoder.layer_0.norm1

📈 Error Metrics
┌─────────────────────┬──────────┬───────────┐
│ Metric              │ Value    │ Tolerance │
├─────────────────────┼──────────┼───────────┤
│ Max Absolute Error  │ 0.00234  │ 0.00100   │
│ Max Relative Error  │ 0.00872  │ 0.00100   │
│ Mean Absolute Error │ 0.00052  │ -         │
│ ULP Distance        │ 18       │ -         │
└─────────────────────┴──────────┴───────────┘

🔥 Activation Analysis at Bad Op
┌─────────────────┬────────────┬────────────┐
│ Statistic       │ Reference  │ Target     │
├─────────────────┼────────────┼────────────┤
│ Min             │ -2.8451    │ -2.8398    │
│ Max             │  3.2156    │  3.2187    │
│ Mean            │  0.0012    │  0.0015    │
│ Std Dev         │  1.0023    │  1.0031    │
│ % Zero          │  0.12%     │  0.14%     │
│ % Inf           │  0.00%     │  0.00%     │
│ % NaN           │  0.00%     │  0.00%     │
└─────────────────┴────────────┴────────────┘

🎯 Likely Root Causes
1. LayerNorm epsilon (1e-6) too small for fp16
2. Accumulation in reduced precision
3. Kernel fusion altering computation order

🔄 Replay Command
phantom replay runs/2025-08-25T14-30-00.json
```

</details>

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_diff.py -v          # Difference metrics
pytest tests/test_fx_capture.py -v    # Graph capture
pytest tests/test_determinism.py -v   # Determinism

# Run with coverage
pytest tests/ --cov=phantom_core --cov-report=html
```

## 🛠️ Development

### Setting Up Development Environment

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run formatters
black phantom_core/ phantom_cli/
ruff check phantom_core/ phantom_cli/

# Type checking
mypy phantom_core/
```

### Project Structure

```
phantom/
├── phantom_cli/          # Command-line interface
│   └── main.py
├── phantom_core/         # Core functionality
│   ├── capture/         # Graph capture (FX)
│   ├── exec/            # Execution runners
│   │   ├── ref_cpu.py   # Reference executor
│   │   └── targets/     # Target backends
│   ├── diff/            # Difference metrics
│   ├── quant/           # Quantization analysis
│   ├── replay/          # Replay functionality
│   ├── report/          # Report generation
│   └── utils/           # Utilities
├── examples/            # Example models
├── tests/               # Test suite
└── docs/                # Documentation
```

### Adding New Backends

To add a new backend (e.g., ROCm):

1. Create `phantom_core/exec/targets/rocm.py`:
```python
from ..base import TargetRunner

class ROCmRunner(TargetRunner):
    def run_ops(self, graph, model, inputs, **kwargs):
        # Implementation here
        pass
```

2. Register in CLI:
```python
if args.target == "rocm":
    target_runner = ROCmRunner()
```

## 🗺️ Roadmap

### v0.1 (Current) ✅
- [x] CPU vs CUDA comparison
- [x] Binary search for first bad op
- [x] HTML reports with heatmaps
- [x] Deterministic replay
- [x] Basic quantization analysis

### v0.2 (Q1 2025) 🚧
- [x] Metal (MPS) backend support
- [x] ROCm backend support
- [x] Interactive Plotly visualizations
- [x] Per-channel quantization stats
- [x] Kernel fusion detection

### v0.3 (Q2 2025) 📋
- [ ] Distributed training support
- [ ] Collective operation debugging
- [ ] Cross-rank drift detection
- [ expectations ] Gradient flow analysis
- [ ] Memory profiling integration

### v0.4 (Q3 2025) 🔮
- [ ] TensorRT debugging
- [ ] ONNX model support
- [ ] Custom operation hooks
- [ ] Performance regression detection
- [ ] CI/CD integration tools

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas We Need Help

- 🔧 Backend implementations (ROCm, TPU, etc.)
- 📊 Visualization improvements
- 🧪 Test coverage expansion
- 📖 Documentation and tutorials
- 🐛 Bug fixes and performance improvements

## 📚 Citations and References

If you use PHANTOM in your research, please cite:

```bibtex
@software{phantom2025,
  title = {PHANTOM: Cross-Accelerator Deterministic Debugger for AI Models},
  author = {Your Team},
  year = {2025},
  url = {https://github.com/forUAi/PHANTOM}
}
```

### Related Work
- [PyTorch FX Documentation](https://pytorch.org/docs/stable/fx.html)
- [Numerical Precision in Deep Learning](https://arxiv.org/abs/xxxx.xxxxx)
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740)

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyTorch team for FX framework
- NumFOCUS for numerical computing tools
- Contributors and early adopters

## 💬 Support

- **Documentation**: [https://phantom-debug.readthedocs.io](https://phantom-debug.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/yourusername/phantom/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/phantom/discussions)
- **Discord**: [Join our community](https://discord.gg/phantom-debug)

## 🔒 Security

For security vulnerabilities, please email security@phantom-debug.org instead of using the issue tracker.

---

<p align="center">
  Made with ❤️ by ML Systems Engineers
  <br>
  <a href="https://github.com/yourusername/phantom">Star us on GitHub</a> ⭐
</p>
