# Quick Reference: TensorRT Model Optimizer Architecture

## One-Page Summary

### The System
- **Purpose**: Optimize deep learning models for efficient GPU inference
- **Approach**: Graph tracing → Symbol analysis → Intelligent quantization
- **Size**: 214 Python files, 61,440 lines of code
- **Key Innovation**: Symbol-aware cross-layer dependency tracking

### Core Modules (5 key)

| Module | Lines | What It Does | Key Class |
|--------|-------|-------------|-----------|
| **trace** | 2,834 | Extract model graphs & find param dependencies | `RobustTracer` |
| **quantization** | 5,289 | Apply quantization (INT8, FP8, NVFP4) | `TensorQuantizer` |
| **export** | 8,218 | Deploy to ONNX/TRT-LLM | `UnifiedExporter` |
| **opt** | 1,359 | Dynamic mode & config search | `Searcher` |
| **onnx** | 3,700+ | ONNX model optimization | `OnnxQuantizer` |

---

## Trace Module at a Glance

### What It Does
1. **Extracts FX graphs** - Converts PyTorch model to functional representation
2. **Identifies symbols** - Maps parameters that affect tensor shapes
3. **Analyzes dependencies** - Finds which parameters must be synchronized
4. **Returns metadata** - Enables downstream optimization decisions

### Key Classes
```
RobustTracer          ← Wraps PyTorch's FX tracer
  └─ _FxTracerPlus   ← Handles unsupported ops
  
GraphCollection       ← Container for all traced graphs
  └─ stores FX graphs per module
  
Symbol               ← Represents model parameters
  └─ states: free, constant, dynamic, searchable
  
SymMap               ← Registry of symbols
  └─ maps module types → symbols
  
GraphDependencyProcessor ← Analyzes dependencies
  └─ builds parameter flow graph
```

### Main API
```python
from modelopt.torch.trace import recursive_trace

# Entry point
graphs = recursive_trace(model)

# Query
for module in graphs:
    graph = graphs[module]                    # FX graph
    failed = graphs.is_failed(module)         # Trace failed?
    msg = graphs.failure_msg(module)          # Why failed?
```

---

## Quantization Module at a Glance

### What It Does
1. **Takes model + config** - User specifies quantization settings
2. **Uses trace info** - Understands model structure
3. **Inserts quantizers** - Replaces modules with quantized versions
4. **Calibrates** - Computes quantization scales using data
5. **Exports** - Produces deployment-ready model

### Key Classes
```
TensorQuantizer      ← Core fake quantization operator
  └─ computes scale (amax)
  └─ applies fake quantization
  
QuantModule          ← Base for quantized modules
  └─ QuantLinear, QuantConv, QuantRNN, etc.
  
QuantizeConfig       ← Configuration dictionary
  └─ per-layer settings

Backends             ← Optimized CUDA/Triton kernels
  ├─ fp8_per_tensor_gemm.py   (FP8 GEMM)
  ├─ nvfp4_gemm.py            (NVFP4 GEMM)
  └─ gemm_registry.py         (dispatch)
```

### Main API
```python
from modelopt.torch.quantization import quantize

config = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": 8},
        "*input_quantizer": {"num_bits": 8},
    },
    "algorithm": "max"
}

model = quantize(model, config, forward_loop=calib_loader)
```

---

## Data Flow

```
PyTorch Model
    │
    ├─→ trace.recursive_trace()
    │   │ [extracts FX graphs]
    │   └─→ GraphCollection
    │
    ├─→ trace.analyze_symbols()
    │   │ [finds dependencies]
    │   └─→ Symbol dependency tree
    │
    ├─→ quantization.quantize()
    │   │ [applies config]
    │   │ [inserts QuantModules]
    │   └─→ Quantized model
    │
    ├─→ calibrate()
    │   │ [runs on data]
    │   │ [computes scales]
    │   └─→ Calibrated model
    │
    └─→ export()
        │ [converts format]
        └─→ Deployable model
```

---

## Supported Formats

| Format | Bits | Type | Use Case |
|--------|------|------|----------|
| INT8 | 8 | Integer | General purpose, CNN |
| INT4-AWQ | 4 | Integer | LLM weights |
| FP8 | 8 | Float | Activations, LLM |
| NVFP4 | 4 | Float | LLM (2025+) |
| MX | 2-8 | Mixed | Flexible |

---

## File Organization

```
modelopt/torch/
├── trace/                      (2,834 lines)
│   ├── tracer.py              (331) FX tracing
│   ├── symbols.py             (545) Symbol def
│   ├── analyzer.py           (1368) Dependency analysis
│   ├── modules/nn.py          (110) PyTorch symbols
│   └── plugins/               - Framework support
│
├── quantization/              (5,289 lines)
│   ├── model_quant.py         (477) Main API
│   ├── tensor_quant.py        (857) Low-level ops
│   ├── config.py              (906) Configuration
│   ├── nn/modules/
│   │   └── tensor_quantizer.py (1293) Core class
│   ├── backends/              - CUDA kernels
│   ├── calib/                 - Calibration
│   └── plugins/               - Framework support
│
├── export/                    (8,218 lines)
│   ├── unified_export_hf.py   (426) HF export
│   ├── unified_export_megatron (1381) Megatron
│   └── [other exporters]
│
├── opt/                       (1,359 lines)
│   ├── dynamic.py             - Mode resolution
│   └── searcher.py            - Auto-search
│
└── utils/                     - Helpers
```

---

## Symbol Concept (Key Innovation)

### What's a Symbol?
A parameter that affects tensor shapes and can be optimized across layers.

Example: In a Transformer, `hidden_size` is a symbol that:
- Is INCOMING to Linear layers (output shape depends on it)
- Is OUTGOING from Linear layers (input shape affects it)
- Must be consistent across attention heads
- Can be searched over for optimal dimensions

### Symbol States
- **free**: Not bound to anything, searchable
- **constant**: Fixed value, cannot change
- **dynamic**: Depends on parent symbol
- **searchable**: Free AND cross-layer

### Why Matters
Enables intelligent optimization:
1. Find which params must be synchronized
2. Identify fusion opportunities
3. Guide quantization decisions
4. Support auto-tuning

---

## Example: NVFP4 Quantization

```python
from modelopt.torch.quantization import quantize

# Configuration
NVFP4_CFG = {
    "quant_cfg": {
        # Linear weights: NVFP4 (4-bit FP)
        "*.linear.weight_quantizer": {
            "num_bits": 4,
            "format": "nvfp4"
        },
        # Linear activations: FP8 (8-bit FP)
        "*.linear.input_quantizer": {
            "num_bits": 8,
            "format": "fp8"
        },
        # LM head: default
        "*lm_head*": {"num_bits": 8},
    },
    "algorithm": "max"
}

# Apply
def calib_loop(model):
    for batch in calib_dataloader:
        model(batch.cuda())

model = quantize(model, NVFP4_CFG, forward_loop=calib_loop)

# Behind the scenes:
# 1. trace extracts model structure
# 2. analyzer finds layer dependencies
# 3. quantization applies NVFP4 to linear layers
# 4. backends use fused NVFP4 GEMM kernels
# 5. calibration computes scales
```

---

## Architecture Patterns

1. **Registry Pattern** - SymMap, QuantModuleRegistry, QuantizeModeRegistry
2. **Plugin Pattern** - Framework-specific code (HF, Megatron, APEX)
3. **Visitor Pattern** - NodeProcessor for graph analysis
4. **Strategy Pattern** - Different calibration algorithms
5. **Facade Pattern** - Simple high-level APIs hiding complexity

---

## Key Numbers

| Metric | Value |
|--------|-------|
| Total Python files | 214 |
| Total lines of code | 61,440 |
| Trace module | 2,834 lines |
| Quantization module | 5,289 lines |
| Export module | 8,218 lines |
| Largest single file | 2,149 lines |
| Supported quant formats | 6+ |
| Supported frameworks | 5+ (HF, Megatron, APEX, etc.) |

---

## Related Sections

- **STRUCTURE.md** - Full repository map
- **SUMMARY.md** - Detailed technical reference
- **README.md** - Navigation guide

---

## Quick Links

| Need | File | Lines |
|------|------|-------|
| Trace entry point | trace/tracer.py | 310-331 |
| Quantization entry | quantization/model_quant.py | 132-224 |
| Core quantizer | quantization/nn/modules/tensor_quantizer.py | 1-300 |
| FP8 kernels | quantization/backends/fp8_per_tensor_gemm.py | 1-250 |
| NVFP4 kernels | quantization/backends/nvfp4_gemm.py | 1-280 |
| Symbol definition | trace/symbols.py | 29-244 |
| Dependency analysis | trace/analyzer.py | 433-500 |

---

**Last Updated**: October 26, 2025
