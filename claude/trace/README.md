# TensorRT Model Optimizer - Architecture & Trace Module Documentation

This directory contains comprehensive documentation of the TensorRT Model Optimizer codebase, with detailed focus on the trace module and how it integrates with the broader optimization ecosystem.

## Documentation Files

### 1. **STRUCTURE.md** (Recommended Starting Point)
Complete map of the repository structure:
- Top-level directory organization
- All 13 PyTorch submodules with line counts
- Detailed trace module file-by-file breakdown
- Quantization module architecture
- Plugin system overview
- Export and utility modules
- ONNX support structure
- Key statistics and patterns

**Use this for**: Getting overview of entire codebase, understanding module relationships

### 2. **SUMMARY.md** (Detailed Technical Reference)
In-depth technical summary addressing CLAUDE.md requirements:
- Overall codebase organization (214 Python files)
- Trace module deep dive with code examples
- Key classes: RobustTracer, GraphCollection, Symbol, SymMap, GraphDependencyProcessor
- How trace analysis works (graph extraction, symbol identification, dependency analysis)
- How trace modifies/doesn't modify user code
- Downstream use in quantization
- Complete API references
- FP8 and NVFP4 quantization details
- Kernel fusion explanations
- Full code examples (INT8, trace direct usage, NVFP4)
- File and line number mappings
- Architecture patterns used

**Use this for**: Understanding implementation details, finding specific code locations, seeing code examples

## Quick Navigation

### Finding Code

**Want to understand how tracing works?**
- STRUCTURE.md → "The Trace Module: Core Architecture" section
- SUMMARY.md → "2. Trace Module Deep Dive" section
- Code: `/modelopt/torch/trace/tracer.py` (lines 33-331)

**Want to understand quantization?**
- STRUCTURE.md → "Quantization Module: Detailed Structure" section
- SUMMARY.md → "5. Quantization Details: FP8 and NVFP4" section
- Code: `/modelopt/torch/quantization/model_quant.py` (lines 132-224)

**Want to see how trace and quantization work together?**
- SUMMARY.md → "3. Downstream Use in Optimization" section
- STRUCTURE.md → "Quantization and Trace Integration" section

**Want to understand symbols?**
- SUMMARY.md → "2. Trace Module Deep Dive" → Symbol Classes section
- Code: `/modelopt/torch/trace/symbols.py` (lines 29-244)

**Want to understand kernel fusion?**
- SUMMARY.md → "Example: Kernel Fusion" section
- STRUCTURE.md → "Backend Implementations" section
- Code: `/modelopt/torch/quantization/backends/`

### Key Files by Purpose

**API Entry Points**
- `modelopt/torch/quantization/model_quant.py` - quantize(), calibrate(), auto_quantize()
- `modelopt/torch/trace/tracer.py` - recursive_trace()
- `modelopt/torch/quantization/__init__.py` - Public API exports

**Graph Tracing**
- `modelopt/torch/trace/tracer.py` (331 lines) - FX graph tracing
- `modelopt/torch/trace/symbols.py` (545 lines) - Symbol definitions
- `modelopt/torch/trace/analyzer.py` (1,368 lines) - Dependency analysis
- `modelopt/torch/trace/modules/nn.py` (110 lines) - Symbol registration for PyTorch modules

**Quantization Core**
- `modelopt/torch/quantization/model_quant.py` (477 lines) - User-facing API
- `modelopt/torch/quantization/model_calib.py` (934 lines) - Calibration
- `modelopt/torch/quantization/tensor_quant.py` (857 lines) - Low-level quantization ops
- `modelopt/torch/quantization/nn/modules/tensor_quantizer.py` (1,293 lines) - Core quantizer class

**Optimized Kernels**
- `modelopt/torch/quantization/backends/fp8_per_tensor_gemm.py` (250 lines) - FP8 GEMM
- `modelopt/torch/quantization/backends/nvfp4_gemm.py` (280 lines) - NVFP4 GEMM
- `modelopt/torch/quantization/backends/gemm_registry.py` (210 lines) - Kernel registry

**Configuration**
- `modelopt/torch/quantization/config.py` (906 lines) - Config system
- `modelopt/torch/quantization/conversion.py` (386 lines) - Config to implementation

## Module Statistics

```
Total Files in modelopt/: 214 Python files, 61,440 lines

Key Modules:
- trace/              2,834 lines    (tracer + symbols + analyzer + modules)
- quantization/       5,289 lines    (core quantization + nn + backends + calib)
- export/             8,218 lines    (ONNX + TRT-LLM export)
- opt/                1,359 lines    (modes + search)
- onnx/quantization/  3,700+ lines   (ONNX-specific)
```

## Architecture Overview

```
User Model
    ↓
trace.recursive_trace()  → Extract FX graphs
    ↓
trace.analyze_symbols()  → Find parameter dependencies
    ↓
quantization.quantize()  → Apply quantization config
    ├→ Insert TensorQuantizer modules
    ├→ Configure per-layer settings
    └→ Select backend kernels (FP8, NVFP4)
    ↓
calibrate()              → Gather statistics
    ↓
export()                 → Deploy-ready model
```

## Key Concepts

### Trace Module
- **RobustTracer**: Augmented PyTorch FX tracer that handles unsupported ops gracefully
- **GraphCollection**: Container for all traced FX graphs
- **Symbol**: Abstraction for model parameters (e.g., hidden_size, kernel_size)
- **SymMap**: Registry mapping module types to their symbols
- **GraphDependencyProcessor**: Analyzes how parameters flow across layers

### Quantization Module
- **TensorQuantizer**: Core fake quantization operator
- **QuantModule**: Base class for quantized modules (QuantLinear, QuantConv, etc.)
- **QuantizeConfig**: Configuration system for per-layer quantization
- **Calibration**: Algorithm to compute quantization scales (amax values)
- **Backends**: Optimized CUDA/Triton kernels for efficient inference

### Supported Quantization Formats
- **INT8**: 8-bit integer (weight + activation)
- **INT4**: 4-bit integer (weights only, AWQ)
- **FP8**: 8-bit floating point (e4m3 or e5m2)
- **NVFP4**: 4-bit floating point (2025+, Blackwell+)
- **MX Formats**: Mixed precision support

## Trace Module Details

### Public API
```python
from modelopt.torch.trace import recursive_trace, analyze_symbols, Symbol, SymMap

# Main entry point
graphs = recursive_trace(model)  # Returns GraphCollection

# Access traced information
for module in graphs:
    fx_graph = graphs[module]           # Get FX graph
    if graphs.is_failed(module):        # Check if failed
        print(graphs.failure_msg(module))

# Analyze dependencies
analyzer = analyze_symbols(model, graphs)
```

### Key Features
1. **Robust Tracing**: Handles unsupported operations by treating as leaf modules
2. **Symbol Tracking**: Identifies parameters that affect tensor shapes
3. **Cross-Layer Analysis**: Finds which parameters must be synchronized
4. **Dependency Graph**: Builds graph of how parameters flow through model

### Use Cases
- Understand model structure and connectivity
- Find optimization opportunities (fusion, quantization)
- Enable intelligent quantization decisions
- Support framework-specific optimizations

## Quantization Module Details

### Public API
```python
from modelopt.torch.quantization import quantize, calibrate

# Quantization config
config = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": 8, "format": "fp8"},
        "*input_quantizer": {"num_bits": 8},
    },
    "algorithm": "max"
}

# Apply quantization
model = quantize(model, config, forward_loop=calib_dataloader)
```

### Supported Algorithms
- **max**: Per-channel max scaling (weight-only)
- **smoothquant**: Activation smoothing for INT8
- **awq_lite**: Activation-aware weight quantization
- **awq_full**: Full AWQ with optimization
- **awq_clip**: Clipping-based AWQ

## Examples

### Example 1: Trace a Model
```python
from modelopt.torch.trace import recursive_trace

model = load_model()
graphs = recursive_trace(model)

# Analyze structure
print("Model structure:")
for module in graphs:
    print(f"  {type(module).__name__}: {len(list(graphs[module].nodes))} ops")
```

### Example 2: Quantize to FP8
```python
from modelopt.torch.quantization import quantize

FP8_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": 8, "format": "fp8"},
        "*input_quantizer": {"num_bits": 8, "format": "fp8"},
    },
    "algorithm": "max"
}

model = quantize(model, FP8_CFG, forward_loop=calib_loader)
```

### Example 3: NVFP4 Quantization
```python
from modelopt.torch.quantization import quantize

NVFP4_CFG = {
    "quant_cfg": {
        "*.linear.weight_quantizer": {"num_bits": 4, "format": "nvfp4"},
        "*.linear.input_quantizer": {"num_bits": 8, "format": "fp8"},
        "default": {"num_bits": 8},
    },
    "algorithm": "max"
}

model = quantize(model, NVFP4_CFG, forward_loop=calib_loader)
```

## File Locations Quick Reference

```
modelopt/
├── torch/
│   ├── trace/
│   │   ├── tracer.py           ← FX graph tracing
│   │   ├── symbols.py          ← Symbol definitions
│   │   ├── analyzer.py         ← Dependency analysis
│   │   ├── modules/
│   │   │   ├── nn.py           ← PyTorch symbol registration
│   │   │   └── concat.py       ← Concat layer handling
│   │   └── plugins/
│   │       ├── transformers.py ← HF Transformers support
│   │       └── megatron.py     ← Megatron-LM support
│   │
│   ├── quantization/
│   │   ├── model_quant.py      ← quantize(), calibrate(), auto_quantize()
│   │   ├── tensor_quant.py     ← Low-level quantization ops
│   │   ├── config.py           ← Configuration system
│   │   ├── nn/modules/
│   │   │   ├── tensor_quantizer.py   ← Core quantizer class
│   │   │   ├── quant_linear.py       ← Quantized Linear
│   │   │   ├── quant_conv.py         ← Quantized Conv
│   │   │   └── [other modules]
│   │   ├── backends/
│   │   │   ├── fp8_per_tensor_gemm.py ← FP8 kernels
│   │   │   ├── nvfp4_gemm.py         ← NVFP4 kernels
│   │   │   └── gemm_registry.py      ← Kernel dispatch
│   │   ├── calib/
│   │   │   └── histogram.py    ← Calibration algorithm
│   │   └── plugins/
│   │       ├── huggingface.py  ← HF Transformers integration
│   │       └── megatron.py     ← Megatron-LM integration
│   │
│   ├── export/                 ← Model export (8,218 lines)
│   ├── opt/                    ← Optimization modes
│   ├── utils/                  ← Shared utilities
│   └── [other optimization modules]
│
├── onnx/                       ← ONNX quantization support
└── deploy/                     ← Deployment utilities
```

## Related Documentation

- CLAUDE.md: Original requirements
- STRUCTURE.md: Complete repository map
- SUMMARY.md: Detailed technical reference
- README.md (in repo root): Installation and usage

## For Learning

1. **Start here**: STRUCTURE.md overview
2. **Understand trace**: SUMMARY.md section 2
3. **Understand quantization**: SUMMARY.md section 5
4. **See examples**: SUMMARY.md section 7
5. **Find code**: Reference sections in both docs

## Notes

- All file paths are relative to `/home/jeromeku/tensorrt-model-optimizer/modelopt/`
- Line numbers refer to code locations for easy reference in editor
- Code examples are simplified for clarity
- Full implementations have additional error handling and optimization

---

**Generated**: October 26, 2025
**Repository**: TensorRT Model Optimizer
**Scope**: Complete architecture analysis with focus on trace module
