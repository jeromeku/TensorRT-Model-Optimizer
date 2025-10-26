# TensorRT Model Optimizer Repository Structure

## Overview

This document provides a comprehensive map of the TensorRT Model Optimizer (ModelOpt) codebase, with a focus on the trace module and its role in the broader architecture. The repository contains approximately 214 Python files across multiple optimization domains.

## Repository Root Structure

```
/home/jeromeku/tensorrt-model-optimizer/
├── modelopt/                    # Main package directory
│   ├── torch/                   # PyTorch-specific optimizations (13 submodules)
│   ├── onnx/                    # ONNX-specific optimizations
│   └── deploy/                  # Deployment utilities
├── tests/                       # Test suite (unit, GPU, examples)
├── examples/                    # Usage examples for different techniques
├── docs/                        # Documentation source
├── setup.py / pyproject.toml    # Package configuration
└── CHANGELOG.rst                # Release notes
```

## Top-Level Directory Tree (3 levels)

```
modelopt/
├── torch/                                    # PyTorch model optimization
│   ├── _deploy/                              # Internal deployment runtime
│   │   ├── _runtime/
│   │   │   └── tensorrt/
│   │   └── utils/
│   ├── distill/                              # Knowledge distillation
│   │   └── plugins/                          # Plugin integrations
│   ├── export/                               # Model export functionality
│   │   └── plugins/
│   ├── nas/                                  # Neural architecture search
│   │   ├── hparams/
│   │   ├── modules/
│   │   └── plugins/
│   ├── opt/                                  # Optimization utilities
│   │   └── plugins/
│   ├── prune/                                # Model pruning
│   │   └── plugins/
│   ├── quantization/                         # Quantization (5,289 lines)
│   │   ├── backends/                         # GEMM & kernel implementations
│   │   ├── calib/                            # Calibration algorithms
│   │   ├── nn/
│   │   │   └── modules/                      # Quantized module replacements
│   │   ├── plugins/                          # Framework integrations
│   │   ├── qtensor/                          # Quantized tensor types
│   │   ├── src/                              # C++ extensions
│   │   └── triton/                           # Triton kernel implementations
│   ├── sparsity/                             # Sparsity optimization
│   │   └── plugins/
│   ├── speculative/                          # Speculative decoding
│   │   ├── eagle/
│   │   ├── medusa/
│   │   ├── mtp/
│   │   └── plugins/
│   ├── trace/                                # GRAPH TRACING MODULE ⭐
│   │   ├── modules/                          # Layer-specific symbol info
│   │   └── plugins/                          # Framework-specific tracing
│   └── utils/                                # Shared utilities
├── onnx/                                     # ONNX optimizations
│   ├── autocast/
│   └── quantization/
│       └── src/
└── deploy/
    └── llm/
```

---

## The Trace Module: Core Architecture

### Directory Contents

```
/modelopt/torch/trace/
├── __init__.py              # Public API exports
├── tracer.py               # (331 lines) FX graph tracing logic
├── symbols.py              # (545 lines) Symbol definitions & tracking
├── analyzer.py             # (1,368 lines) Dependency analysis
├── modules/
│   ├── __init__.py
│   ├── nn.py               # Native PyTorch layer symbol info
│   └── concat.py           # Concat-specific symbol handling
└── plugins/
    ├── __init__.py
    ├── megatron.py         # Megatron-LM framework plugin
    └── transformers.py     # HuggingFace Transformers plugin
```

### Trace Module Statistics

| File | Lines | Purpose |
|------|-------|---------|
| analyzer.py | 1,368 | Symbol dependency graph analysis, cross-layer constraint resolution |
| symbols.py | 545 | Symbol abstraction layer, state management, dependency trees |
| tracer.py | 331 | FX graph tracing wrapper, robust module tracing |
| modules/nn.py | 110 | Symbol info registration for standard nn modules |
| modules/concat.py | 480 | Complex concatenation layer symbol analysis |
| **TOTAL** | **2,834** | Complete layer-wise tracing infrastructure |

---

## Key Modules and Their Relationships

### Module Dependency Graph

```
User Code
    ↓
quantization.quantize()           ← Main user-facing API
    ↓
opt.apply_mode()                  ← Dynamic mode application
    ├→ opt/conversion.py          ← Convert between configs
    └→ opt/dynamic.py             ← Dynamic mode resolution
         ↓
    quantization.model_quant.quantize()
         ↓
    nn.modules.QuantModule        ← Base quantized modules
         ↓
    tensor_quant.py               ← Low-level quantization ops
         ├→ backends/             ← CUDA kernels (FP8, NVFP4)
         └→ triton/               ← Triton kernels

ANALYSIS & TRACING PATH:
    quantization.config.py         ← Define quant configs
         ↓
    trace.tracer.recursive_trace() ← Extract graph structure
         ↓
    trace.analyzer.analyze_symbols()← Find cross-layer dependencies
         ↓
    trace.symbols.SymMap          ← Track parameter relationships
         ↓
    opt/searcher.py               ← Search space optimization
```

---

## Quantization Module: Detailed Structure

The quantization module (5,289 lines) is the largest and most complex:

### Core Quantization Files

| File | Lines | Key Responsibility |
|------|-------|-------------------|
| **model_quant.py** | 477 | Public API: quantize(), calibrate(), auto_quantize() |
| **model_calib.py** | 934 | Calibration loop implementation, algorithm dispatch |
| **tensor_quant.py** | 857 | Low-level quantization ops (FP8, INT4, NVFP4, MX formats) |
| **config.py** | 906 | Quantization config definition & validation |
| **mode.py** | 428 | Mode registry system for dynamic configuration |
| **conversion.py** | 386 | Config to quantizer attribute conversion |
| **export_onnx.py** | 615 | ONNX export for quantized models |
| **compress.py** | 204 | Model compression utilities |
| **utils.py** | 367 | Helper functions for quantization |

### Quantization Neural Network Modules (nn/modules/)

```
nn/modules/
├── tensor_quantizer.py     (1,293 lines) ← Core TensorQuantizer class
├── quant_linear.py         (330 lines)   ← Quantized linear layers
├── quant_conv.py           (155 lines)   ← Quantized conv layers
├── quant_rnn.py            (527 lines)   ← Quantized RNN/LSTM
├── quant_batchnorm.py      (50 lines)    ← Quantized batch norm
├── quant_activations.py    (40 lines)    ← Quantized activations
├── quant_instancenorm.py   (65 lines)    ← Quantized instance norm
├── quant_pooling.py        (135 lines)   ← Quantized pooling
└── quant_module.py         (270 lines)   ← Base QuantModule class
```

The **TensorQuantizer** (1,293 lines) is the central quantization class that handles:
- Fake quantization forward pass
- Scale factor computation
- Quantization parameter tracking
- Export to various formats (FP8, INT8, INT4, NVFP4, MXfloat)

### Backend Implementations (quantization/backends/)

```
backends/
├── fp8_per_tensor_gemm.py   (250 lines)   ← FP8 GEMM kernels
├── nvfp4_gemm.py            (280 lines)   ← NVFP4 GEMM kernels
├── gemm_registry.py         (210 lines)   ← Kernel registry & dispatch
└── utils.py                 (40 lines)    ← Kernel utilities
```

These implement optimized fused kernels for quantized matrix multiplication.

### Calibration Module (quantization/calib/)

```
calib/
├── histogram.py             (434 lines)   ← Histogram-based statistics
└── [other calib algorithms via plugins]
```

### Plugin System (quantization/plugins/)

Framework-specific plugins that handle model-specific quantization:
- `huggingface.py` (602 lines) - HuggingFace Transformers integration
- `megatron.py` - Megatron-LM integration  
- `apex.py` - APEX (distributed) integration
- `accelerate.py` - Accelerate framework integration
- `transformer_engine.py` - Transformer Engine integration

---

## Trace Module: Deep Dive

### 1. **tracer.py** - FX Graph Tracing

**Purpose**: Wrapper around PyTorch's FX tracer that robustly handles unsupported operations.

**Key Classes**:
- `RobustTracer`: Main tracing class
- `RobustTracer._FxTracerPlus`: Augmented FX tracer
- `GraphCollection`: Container for traced graphs

**How It Works**:
```
RobustTracer.trace(model) 
    ↓
_FxTracerPlus.trace(model)  [tries standard FX tracing]
    ↓ [if fails due to unsupported op]
[Wraps failing module as leaf, retries]
    ↓
GraphCollection contains all traced subgraphs
```

**API Methods**:
```python
recursive_trace(model, concrete_args=None) → GraphCollection
    # Returns fx.Graph for model and all failing submodules
    
GraphCollection
    .is_failed(module) → bool
    .failure_msg(module) → str
    .is_unvisited(module) → bool
    .__getitem__(module) → fx.Graph
    .__iter__() → Generator[module]
```

### 2. **symbols.py** - Symbol Abstraction

**Purpose**: Define symbols that represent parameters (e.g., hidden size, kernel size) that affect tensor shapes across layers.

**Key Classes**:
- `Symbol`: Base symbolic parameter
- `SymInfo`: Information container for a module's symbols
- `SymMap`: Registry mapping modules to their symbols
- `Symbol.CLType`: Cross-layer type (NONE, INCOMING, OUTGOING)

**Symbol States** (mutually exclusive):
- **free**: Not bound, can be searched
- **searchable**: Free and cross-layer
- **constant**: Fixed value
- **dynamic**: Determined by parent symbol

**Symbol Properties** (cross-layer significance):
- **incoming**: Depends on input tensor
- **outgoing**: Output tensor depends on this
- **none**: Internal to module only
- **is_cross_layer**: incoming OR outgoing
- **is_dangling**: free AND cross_layer

**Example Symbol Registration** (from modules/nn.py):
```python
@SymMap.register(nn.Linear)
def get_linear_sym_info(mod: nn.Module) -> SymInfo:
    in_features = Symbol(
        cl_type=Symbol.CLType.INCOMING,  # input tensor depends on it
        elastic_dims={-1}
    )
    out_features = Symbol(
        is_searchable=True,               # can be optimized
        cl_type=Symbol.CLType.OUTGOING,  # output shape depends on it
        elastic_dims={-1}
    )
    return SymInfo(in_features=in_features, out_features=out_features)
```

### 3. **analyzer.py** - Dependency Analysis (1,368 lines)

**Purpose**: Build a dependency graph of how tensor shapes flow through the model, identifying which parameters must be synchronized across layers.

**Key Classes**:
- `NodeProcessor`: Base class for processing different node types
- `GraphDependencyProcessor`: Main orchestrator
- `Map`: Tracks dependencies and mappings
- `BoundaryNodeProcessor`: Default processor for unknown nodes

**Main Algorithm**:
```
analyze_symbols(model, gc, sym_map)
    ↓
For each layer in graph:
    Identify input/output nodes
    Check symbol compatibility
    ← Synchronize symbols if possible
    ← Or disable them if incompatible
    ↓
Result: Symbol dependency tree showing what parameters 
        must be kept consistent across layers
```

**Key Methods**:
```python
class GraphDependencyProcessor:
    .process()                      # Analyze all dependencies
    .register_node_processor(cls)   # Register custom processors
    
class Analyzer:
    .named_cross_layer_symbols(target)     # Cross-layer parameters
    .named_dangling_symbols(target)        # Unbound cross-layer params
    .named_searchable_out_symbols(target)  # Optimizable outputs
```

---

## Quantization and Trace Integration

### How Trace Enables Quantization

```
quantize(model, config) 
    ↓
recursive_trace(model)           ← Trace to FX graphs
    ↓ [for each traced module]
analyze_symbols()                ← Find layer dependencies
    ↓
[Apply quantization configs]
    ↓
[Identify what can be jointly quantized]
    ↓
QuantModule placement           ← Insert quantized ops
    ↓
calibrate(model, forward_loop)  ← Gather statistics
    ↓
Optimized quantized model
```

### Example: NVFP4 Quantization Flow

1. **Trace Phase**: Build graph of model structure
2. **Analysis Phase**: Find which linear layers have compatible output shapes
3. **Config Phase**: Apply per-layer or per-group quantization settings
4. **Quantization Phase**: Replace with `QuantLinear` (wraps TensorQuantizer)
5. **Calibration Phase**: Run data through to compute scales (amax)
6. **Export Phase**: Export to ONNX or deployment format

---

## Key File Locations and Relationships

### User-Facing APIs

| API Function | File | Line Range | Purpose |
|--------------|------|-----------|---------|
| `quantize()` | model_quant.py | 132-224 | Main quantization entry point |
| `calibrate()` | model_quant.py | 54-114 | Calibration step |
| `auto_quantize()` | model_quant.py | 228-250+ | Automatic config search |
| `recursive_trace()` | tracer.py | 310-331 | Graph tracing entry point |
| `analyze_symbols()` | analyzer.py | 1350+ | Symbol dependency analysis |

### Internal Classes

| Class | File | Purpose |
|-------|------|---------|
| `TensorQuantizer` | nn/modules/tensor_quantizer.py | Core quantization operator |
| `QuantModule` | nn/modules/quant_module.py | Base for quantized modules |
| `RobustTracer` | tracer.py | FX graph tracing |
| `GraphCollection` | tracer.py | Traced graph container |
| `Symbol` | symbols.py | Parameter abstraction |
| `SymMap` | symbols.py | Symbol registry |
| `GraphDependencyProcessor` | analyzer.py | Dependency analysis |

---

## Plugins System

### Trace Plugins (modelopt/torch/trace/plugins/)

Framework-specific graph extraction:

```
plugins/
├── transformers.py    ← HuggingFace models (extract special layers)
└── megatron.py       ← Megatron-LM models (distributed patterns)
```

### Quantization Plugins (modelopt/torch/quantization/plugins/)

Framework-specific quantization strategies:

```
plugins/
├── huggingface.py              (602 lines)  ← Transformers integration
├── megatron.py                 ← Megatron PTQ/QAT
├── apex.py                     ← Distributed training (APEX)
├── accelerate.py               ← Hugging Face Accelerate
└── transformer_engine.py       ← Transformer Engine ops
```

### Opt Plugins (modelopt/torch/opt/plugins/)

Optimization mode plugins:
```
plugins/
├── transformers_multi_process.py
├── transformers_tp.py          ← Tensor parallel aware
└── megatron_chaining.py        ← Megatron-specific modes
```

---

## Export Module (8,218 lines total)

Connected to quantization for deployment:

```
export/
├── unified_export_hf.py        (426 lines) ← HuggingFace export
├── unified_export_megatron.py  (1,381 lines) ← Megatron export
├── layer_utils.py              (1,782 lines) ← Layer conversion utils
├── quant_utils.py              (1,069 lines) ← Quantization export helpers
├── postprocess.py              (847 lines) ← Post-export optimization
├── model_config.py             (622 lines)
├── model_config_export.py      (555 lines)
└── plugins/
    └── [framework-specific export plugins]
```

The export module uses traced graphs to generate optimized ONNX or TRT-LLM compatible models.

---

## Utilities Module (modelopt/torch/utils/)

Shared utilities used across modules:

```
utils/
├── graph.py                ← FX graph utilities
├── network.py              (658 lines) ← Model architecture utilities
├── dataset_utils.py        (442 lines) ← Data loading
├── distributed.py          ← Distributed training helpers
├── tensor.py               ← Tensor operations
├── logging.py              ← Logging infrastructure
├── random.py               ← Random seed management
└── [other utilities]
```

**Key graph.py utilities**:
```python
NodeTarget         # Type for module/function references
_get_node_target() # Extract target from FX node
```

---

## C++ and CUDA Components

### Quantization CUDA Extensions (quantization/src/)

```
src/
└── [CUDA kernels for quantization operations]
```

Wrapped and exposed via `extensions.py`:
- `get_cuda_ext()` - Load general quantization CUDA extension
- `get_cuda_ext_fp8()` - FP8-specific kernels
- `get_cuda_ext_mx()` - MX format kernels

### Triton Kernels (quantization/triton/)

GPU-compiled kernels using Triton for modern quantization operations.

---

## ONNX Support Module (modelopt/onnx/)

Parallel quantization support for ONNX models:

```
onnx/
├── quantization/
│   ├── int4.py            (1,338 lines) ← INT4 ONNX quantization
│   ├── qdq_utils.py       (1,006 lines) ← Quantize-Dequantize ops
│   ├── quantize.py        (472 lines)   ← Main ONNX quantization
│   └── src/               ← ONNX CUDA kernels
└── autocast/
    ├── precisionconverter.py (1,022 lines) ← Precision conversion
    └── graphsanitizer.py     (422 lines)
```

---

## Testing Structure

```
tests/
├── unit/
│   └── torch/
│       └── quantization/
├── gpu/
│   └── torch/
│       ├── quantization/         ← Quantization tests
│       ├── export/               ← Export tests
│       ├── nas/                  ← NAS tests
│       └── [other optimization tests]
└── _test_utils/
    └── [Shared test utilities]
```

---

## Examples Structure

```
examples/
├── llm_ptq/                    ← Post-training quantization workflows
├── llm_qat/                    ← Quantization-aware training
├── cnn_qat/                    ← CNN quantization
├── onnx_ptq/                   ← ONNX quantization
├── distill/                    ← Knowledge distillation
├── pruning/                    ← Model pruning
├── nas/                        ← Neural architecture search
└── [other technique examples]
```

---

## Data Flow Summary

### Quantization Data Flow

```
User Model
    ↓
recursive_trace() ──→ GraphCollection (FX graphs)
    ↓
analyze_symbols() ──→ Symbol dependencies
    ↓
quantize(config) ──→ Apply QuantModule replacements
    ↓
    ├→ Insert TensorQuantizer wrappers
    ├→ Compute quantization scales
    └→ Fake quantization forward pass
    ↓
calibrate(data) ──→ Gather statistics
    ↓
export() ──→ ONNX/TRT-LLM format
```

### Symbol Resolution Data Flow

```
Layer Graph
    ↓
Identify Symbols (from SymMap)
    ↓
Build Dependency Graph
    ↓
Synchronize Cross-Layer Symbols
    ↓
Searchable Parameters for Auto Optimization
```

---

## Key Statistics

| Category | Count |
|----------|-------|
| Total Python Files | 214 |
| Trace Module Lines | 2,834 |
| Quantization Module Lines | 5,289 |
| Export Module Lines | 8,218 |
| Largest File | speculative/plugins/megatron.py (2,149 lines) |
| Supported Quantization Formats | 6+ (INT8, INT4, FP8, NVFP4, MX formats) |

---

## Architecture Patterns

### 1. **Plugin Architecture**
Multiple optimization domains (quantization, export, NAS) use plugins to support different frameworks (HuggingFace, Megatron, APEX).

### 2. **Registry Pattern**
- `SymMap` - Symbol registry
- `QuantModuleRegistry` - Quantized module registry
- `QuantizeModeRegistry` - Quantization mode registry

### 3. **Visitor Pattern**
- `NodeProcessor` - Process different node types
- `GraphDependencyProcessor` - Orchestrate node processing

### 4. **Facade Pattern**
- High-level APIs (quantize, calibrate, export)
- Hide complex internal machinery

### 5. **Strategy Pattern**
- Different calibration algorithms (max, smoothquant, AWQ)
- Different quantization formats (INT8, FP8, INT4, NVFP4)

---

## Conclusion

The TensorRT Model Optimizer is a sophisticated modular system for model optimization. The **trace module** provides the foundational graph extraction and symbol analysis that enables intelligent quantization decisions. The **quantization module** leverages this information to apply efficient quantization across different formats and frameworks. The entire system is designed around extensibility through plugins and registries, making it adaptable to new models and frameworks.

