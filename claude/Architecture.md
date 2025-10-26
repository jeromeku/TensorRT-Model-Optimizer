# TensorRT Model Optimizer - Architecture

This document provides a high-level map of the TensorRT Model Optimizer (ModelOpt) codebase. It is intended to help you understand where things are and how the major components fit together.

For detailed technical documentation, see [`claude/trace/`](claude/trace/) which contains comprehensive guides on the trace module, quantization system, and API reference.

## Bird's Eye View

TensorRT Model Optimizer is a PyTorch model optimization library (~214 Python files, 61K LOC) that provides:
- **Quantization** (INT8, INT4, FP8, NVFP4, MX formats)
- **Pruning** and **Sparsity**
- **Knowledge Distillation**
- **Neural Architecture Search (NAS)**
- **Export** to ONNX and TensorRT-LLM

The heart of the system is the **trace module**, which extracts and analyzes model structure to enable intelligent optimization decisions.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Model (PyTorch)                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    TRACE MODULE                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ tracer.py: FX Graph Extraction (RobustTracer)       │   │
│  │   - Converts PyTorch model to functional graph      │   │
│  │   - Handles unsupported ops gracefully              │   │
│  └──────────────────────────────────────────────────────┘   │
│                         ↓                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ symbols.py: Parameter Abstraction (Symbol, SymMap)  │   │
│  │   - Represents model parameters (e.g., hidden_dim)  │   │
│  │   - Tracks cross-layer dependencies                 │   │
│  └──────────────────────────────────────────────────────┘   │
│                         ↓                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ analyzer.py: Dependency Analysis                     │   │
│  │   - Builds dependency graph                          │   │
│  │   - Identifies optimization opportunities            │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                 OPTIMIZATION MODULES                        │
│  ┌───────────────────┐  ┌──────────────────┐               │
│  │  QUANTIZATION     │  │  PRUNING         │               │
│  │  (5,289 lines)    │  │  Structured/     │               │
│  │  - INT8, FP8      │  │  Unstructured    │               │
│  │  - NVFP4, MXfp8   │  └──────────────────┘               │
│  │  - Calibration    │                                      │
│  └───────────────────┘  ┌──────────────────┐               │
│           │             │  DISTILLATION    │               │
│           │             │  Knowledge       │               │
│           │             │  Transfer        │               │
│           │             └──────────────────┘               │
│           ↓                                                 │
│  ┌───────────────────────────────────────────────────┐     │
│  │  BACKENDS (CUDA Kernels, Triton)                 │     │
│  │  - fp8_per_tensor_gemm.py                        │     │
│  │  - nvfp4_gemm.py                                 │     │
│  │  - Fused operations                              │     │
│  └───────────────────────────────────────────────────┘     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                   EXPORT MODULE (8,218 lines)               │
│  - ONNX export                                              │
│  - TensorRT-LLM export                                      │
│  - Framework plugins (HuggingFace, Megatron-LM)             │
└─────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
modelopt/
├── torch/                         # PyTorch optimizations
│   ├── trace/                     # ⭐ Graph extraction & analysis (2,834 lines)
│   │   ├── tracer.py              # FX graph tracing (331 lines)
│   │   ├── symbols.py             # Symbol abstraction (545 lines)
│   │   ├── analyzer.py            # Dependency analysis (1,368 lines)
│   │   ├── modules/               # Symbol info for PyTorch layers
│   │   └── plugins/               # Framework-specific tracing
│   │
│   ├── quantization/              # ⭐ Quantization system (5,289 lines)
│   │   ├── model_quant.py         # Public API: quantize(), calibrate()
│   │   ├── tensor_quant.py        # Low-level quantization ops
│   │   ├── config.py              # Configuration system
│   │   ├── nn/modules/            # Quantized module replacements
│   │   │   └── tensor_quantizer.py  # Core quantization operator (1,293 lines)
│   │   ├── backends/              # CUDA kernels (FP8, NVFP4, etc.)
│   │   ├── calib/                 # Calibration algorithms
│   │   └── plugins/               # Framework integrations
│   │
│   ├── export/                    # Model export (8,218 lines)
│   │   ├── unified_export_hf.py   # HuggingFace export
│   │   ├── unified_export_megatron.py  # Megatron-LM export
│   │   └── quant_utils.py         # Quantization export helpers
│   │
│   ├── opt/                       # Optimization modes & search
│   ├── prune/                     # Model pruning
│   ├── distill/                   # Knowledge distillation
│   ├── nas/                       # Neural architecture search
│   ├── sparsity/                  # Sparsity optimization
│   └── utils/                     # Shared utilities
│
├── onnx/                          # ONNX-specific optimizations
│   └── quantization/              # ONNX quantization
│
└── deploy/                        # Deployment utilities
    └── llm/                       # LLM deployment
```

## Core Components

### 1. Trace Module (`modelopt/torch/trace/`)

**Purpose**: Extract model structure and analyze parameter dependencies to enable intelligent optimization.

**Key APIs**:
- `recursive_trace(model)` → [`GraphCollection`](modelopt/torch/trace/tracer.py#L239-L308) - Trace model to FX graphs
- `analyze_symbols(model, graphs)` → `Analyzer` - Analyze parameter dependencies

**Key Classes**:
- [`RobustTracer`](modelopt/torch/trace/tracer.py#L33-L238) - Wraps PyTorch FX tracer, handles unsupported ops
- [`Symbol`](modelopt/torch/trace/symbols.py#L29-L244) - Abstraction for model parameters (e.g., `hidden_dim`, `num_heads`)
- [`SymMap`](modelopt/torch/trace/symbols.py#L255-L437) - Registry mapping module types to their symbols
- [`GraphDependencyProcessor`](modelopt/torch/trace/analyzer.py#L433-L1368) - Analyzes cross-layer dependencies

**What it does**:
1. Converts PyTorch model to FX functional graph representation
2. Identifies which parameters affect tensor shapes across layers
3. Builds dependency graph showing which parameters must be synchronized
4. Provides this information to downstream optimization modules

**Example**:
```python
from modelopt.torch.trace import recursive_trace

model = MyTransformer()
graphs = recursive_trace(model)  # Extract all graphs

# Query structure
for module in graphs:
    graph = graphs[module]
    print(f"{type(module).__name__}: {len(list(graph.nodes))} operations")
```

See: [`claude/trace/SUMMARY.md#2-trace-module-deep-dive`](claude/trace/SUMMARY.md) for detailed walkthrough.

### 2. Quantization Module (`modelopt/torch/quantization/`)

**Purpose**: Apply various quantization formats to reduce model size and increase inference speed.

**Key APIs**:
- [`quantize(model, config, forward_loop)`](modelopt/torch/quantization/model_quant.py#L132-L224) - Main entry point
- [`calibrate(model, algorithm, forward_loop)`](modelopt/torch/quantization/model_quant.py#L54-L114) - Calibration step
- `auto_quantize(model, forward_loop)` - Automatic config search

**Supported Formats**:
| Format | Bits | Use Case | Backend |
|--------|------|----------|---------|
| INT8 | 8 | General quantization | CUDA |
| INT4 | 4 | Aggressive compression | CUDA |
| FP8 (e4m3/e5m2) | 8 | GPU-native format | CUDA, Triton |
| NVFP4 | 4 | LLMs on Blackwell+ | CUDA |
| MX (microscaling) | 4-8 | Block-wise quantization | CUDA |

**Core Classes**:
- [`TensorQuantizer`](modelopt/torch/quantization/nn/modules/tensor_quantizer.py) - Core fake quantization operator (1,293 lines)
- `QuantLinear`, `QuantConv2d` - Quantized layer replacements
- Backend kernels in [`backends/`](modelopt/torch/quantization/backends/)

**How it works**:
```
quantize(model, config)
    ↓
1. recursive_trace(model)        # Extract graph structure
    ↓
2. analyze_symbols()             # Find layer dependencies
    ↓
3. Apply quantization config     # Insert TensorQuantizer wrappers
    ↓
4. calibrate(data)               # Compute quantization scales
    ↓
Optimized quantized model
```

**Example - NVFP4 Quantization**:
```python
from modelopt.torch.quantization import quantize

NVFP4_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": 4, "format": "nvfp4"},
        "*input_quantizer": {"num_bits": 8, "format": "fp8"}
    },
    "algorithm": "max"
}

model = quantize(model, NVFP4_CFG, forward_loop=calibration_dataloader)
```

See: [`claude/trace/SUMMARY.md#5-quantization-details-fp8-and-nvfp4`](claude/trace/SUMMARY.md) for implementation details.

### 3. Export Module (`modelopt/torch/export/`)

**Purpose**: Export optimized models to deployment formats (ONNX, TensorRT-LLM).

**Key Components**:
- [`unified_export_hf.py`](modelopt/torch/export/unified_export_hf.py) (426 lines) - HuggingFace model export
- [`unified_export_megatron.py`](modelopt/torch/export/unified_export_megatron.py) (1,381 lines) - Megatron-LM export
- [`quant_utils.py`](modelopt/torch/export/quant_utils.py) (1,069 lines) - Quantization export helpers
- [`layer_utils.py`](modelopt/torch/export/layer_utils.py) (1,782 lines) - Layer conversion utilities

**Integration with Trace**:
Uses traced graph information to generate optimized ONNX/TRT-LLM compatible models with correct quantization annotations.

## How PyTorch APIs Are Used

### PyTorch FX (Functional Graph Extraction)

The trace module is built on [PyTorch FX](https://pytorch.org/docs/stable/fx.html), which enables symbolic tracing:

**Location**: [`tracer.py:33-238`](modelopt/torch/trace/tracer.py#L33-L238)

```python
from torch.fx import Tracer, GraphModule, Graph

class RobustTracer(Tracer):
    """Extended FX tracer that handles unsupported operations."""

    def trace(self, model):
        try:
            # Standard FX tracing
            graph = super().trace(model)
            return graph
        except Exception as e:
            # Wrap failing module as leaf, retry recursively
            return self._handle_trace_failure(model, e)
```

**What ModelOpt does with FX**:
1. **Graph Extraction** - Convert model to functional representation (nodes = ops, edges = data flow)
2. **Robust Handling** - FX can fail on dynamic control flow; ModelOpt wraps failures as "leaf" modules
3. **Recursive Tracing** - Traces entire model hierarchy, building graph collection
4. **Node Analysis** - Analyzes graph nodes to understand layer connectivity

**Key FX APIs Used**:
- `torch.fx.Tracer.trace()` - Extract symbolic graph
- `torch.fx.Graph` - Functional representation
- `torch.fx.Node` - Individual operations
- `torch.fx.GraphModule` - Executable graph

### PyTorch Module Hooks

Used for calibration and quantization parameter tracking:

**Location**: [`quantization/model_calib.py`](modelopt/torch/quantization/model_calib.py)

```python
# Register forward hooks to capture activations
def register_hooks(model):
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer):
            module.register_forward_hook(collect_stats_hook)

def collect_stats_hook(module, input, output):
    # Collect min/max statistics during calibration
    amax = output.abs().max()
    module.update_amax(amax)
```

### PyTorch CUDA Extensions

Custom CUDA kernels for optimized quantization:

**Location**: [`quantization/src/`](modelopt/torch/quantization/src/) (C++/CUDA)

**Example - FP8 Kernel**:
```cpp
// CUDA kernel for FP8 quantization
__global__ void fake_e4m3_kernel(
    const float* input,
    float* output,
    const float scale,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Quantize to FP8 e4m3, then dequantize
        float scaled = input[idx] * scale;
        float quantized = fake_quantize_e4m3(scaled);
        output[idx] = quantized / scale;
    }
}
```

**Python wrapper**:
```python
# quantization/extensions.py
def get_cuda_ext_fp8():
    """Load FP8 CUDA extension."""
    return torch.utils.cpp_extension.load(
        name="modelopt_cuda_ext_fp8",
        sources=["csrc/fp8_ops.cu"],
        extra_cuda_cflags=["-O3"]
    )
```

## Data Flow: End-to-End Example

### NVFP4 Quantization Flow

Let's trace a complete quantization workflow from user code to CUDA kernels:

**1. User Code**:
```python
from modelopt.torch.quantization import quantize

model = MyLLM()  # User's PyTorch model
config = {"quant_cfg": {"*weight_quantizer": {"num_bits": 4, "format": "nvfp4"}}}

quantized_model = quantize(model, config, forward_loop=calibration_data)
```

**2. Entry Point** - [`quantization/model_quant.py:132-224`](modelopt/torch/quantization/model_quant.py#L132-L224):
```python
def quantize(model, config, forward_loop=None):
    # Step 1: Apply quantization mode (uses trace internally)
    model = apply_mode(model, mode=[("quantize", config)])

    # Step 2: Calibrate
    model = calibrate(model, forward_loop)

    return model
```

**3. Mode Application** - [`opt/dynamic.py`](modelopt/torch/opt/dynamic.py):
```python
def apply_mode(model, mode):
    # Internally calls trace to understand model structure
    from modelopt.torch.trace import recursive_trace

    graphs = recursive_trace(model)  # ← Trace extraction

    # Use graph info to apply config intelligently
    return _apply_quantization_with_trace_info(model, config, graphs)
```

**4. Trace Extraction** - [`trace/tracer.py:310-331`](modelopt/torch/trace/tracer.py#L310-L331):
```python
def recursive_trace(model, concrete_args=None):
    """Extract FX graphs for entire model hierarchy."""
    gc = GraphCollection()
    gc.recursive_trace(model, concrete_args)  # ← Uses PyTorch FX
    return gc
```

**5. PyTorch FX Call** - [`trace/tracer.py:102-150`](modelopt/torch/trace/tracer.py#L102-L150):
```python
class _FxTracerPlus(torch.fx.Tracer):
    def trace(self, root, concrete_args=None):
        # ← PYTORCH API: torch.fx.Tracer.trace()
        return super().trace(root, concrete_args)
```

**6. Symbol Analysis** - [`trace/analyzer.py:1350+`](modelopt/torch/trace/analyzer.py):
```python
def analyze_symbols(model, gc, sym_map):
    """Build dependency graph of parameters."""
    processor = GraphDependencyProcessor(model, gc, sym_map)
    processor.process()  # Analyzes all layer connections
    return Analyzer(processor)
```

**7. QuantModule Insertion** - [`quantization/conversion.py`](modelopt/torch/quantization/conversion.py):
```python
def set_quantizer_by_cfg(model, config):
    """Replace layers with quantized versions."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Replace with QuantLinear
            quant_module = QuantLinear(module)
            quant_module.weight_quantizer = TensorQuantizer(
                num_bits=4,
                format="nvfp4"
            )
            setattr(parent, name, quant_module)
```

**8. Calibration** - [`quantization/model_calib.py`](modelopt/torch/quantization/model_calib.py):
```python
def calibrate(model, forward_loop):
    """Run calibration to compute quantization scales."""
    # Enable calibration mode
    for module in model.modules():
        if isinstance(module, TensorQuantizer):
            module.enable_calib()

    # Run forward pass
    forward_loop(model)  # ← PYTORCH API: model.forward() calls

    # Compute scales
    for module in model.modules():
        if isinstance(module, TensorQuantizer):
            module.compute_amax()  # Compute quantization scale
            module.disable_calib()
```

**9. Fake Quantization Forward** - [`quantization/nn/modules/tensor_quantizer.py`](modelopt/torch/quantization/nn/modules/tensor_quantizer.py):
```python
class TensorQuantizer(nn.Module):
    def forward(self, input):
        """Apply fake quantization."""
        if self._format == "nvfp4":
            # Call low-level quantization
            return nvfp4_quantize(input, self._amax, self._num_bits)
        # ...
```

**10. Low-Level Quantization** - [`quantization/tensor_quant.py:47-99`](modelopt/torch/quantization/tensor_quant.py#L47-L99):
```python
def nvfp4_quantize(inputs, amax, num_bits):
    """NVFP4 fake quantization."""
    # Get CUDA extension
    cuda_ext = get_cuda_ext()

    # Call fused CUDA kernel
    return cuda_ext.nvfp4_fake_quantize(
        inputs,
        amax.float(),
        num_bits
    )  # ← Calls C++/CUDA kernel
```

**11. CUDA Kernel** - [`quantization/backends/nvfp4_gemm.py`](modelopt/torch/quantization/backends/nvfp4_gemm.py):
```python
class NVFP4Kernel:
    @staticmethod
    def quantize(weights, scales):
        """Call optimized CUDA kernel for NVFP4 quantization."""
        # Dispatch to specialized GEMM kernel
        return _nvfp4_cuda_kernel(weights, scales)
```

**12. C++ Extension** - `quantization/src/nvfp4_kernels.cu` (C++/CUDA):
```cpp
// Optimized CUDA kernel for NVFP4 quantization
__global__ void nvfp4_quantize_kernel(
    const float* input,
    float* output,
    const float* scales,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Quantize to NVFP4 format
        float scaled = input[idx] / scales[idx / GROUP_SIZE];
        output[idx] = fake_quantize_nvfp4(scaled) * scales[idx / GROUP_SIZE];
    }
}
```

### Call Stack Summary

```
User Code: quantize(model, config)
    ↓ [Python]
quantization/model_quant.py:132
    ↓ [Python]
opt/dynamic.py: apply_mode()
    ↓ [Python]
trace/tracer.py:310: recursive_trace()
    ↓ [Python → PyTorch]
torch.fx.Tracer.trace()  ← PYTORCH FX API
    ↓ [Python]
trace/analyzer.py: analyze_symbols()
    ↓ [Python]
quantization/conversion.py: set_quantizer_by_cfg()
    ↓ [Python]
quantization/model_calib.py: calibrate()
    ↓ [Python → PyTorch]
model.forward()  ← PYTORCH API
    ↓ [Python]
nn/modules/tensor_quantizer.py: TensorQuantizer.forward()
    ↓ [Python]
quantization/tensor_quant.py: nvfp4_quantize()
    ↓ [Python → C++]
get_cuda_ext().nvfp4_fake_quantize()
    ↓ [C++/CUDA]
nvfp4_kernels.cu: nvfp4_quantize_kernel<<<>>>()  ← CUDA KERNEL
```

## Kernel Fusion Example

Kernel fusion combines multiple operations into a single optimized kernel to reduce memory bandwidth.

**Pattern**: Linear → ReLU → Linear

**1. Original Model**:
```python
class Model(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(768, 3072)
        self.fc2 = nn.Linear(3072, 768)

    def forward(self, x):
        x = self.fc1(x)           # GEMM 1
        x = F.relu(x)             # Activation
        x = self.fc2(x)           # GEMM 2
        return x
```

**2. Trace Analysis** - [`trace/tracer.py`](modelopt/torch/trace/tracer.py):
```python
graphs = recursive_trace(model)
graph = graphs[model]

# FX graph nodes:
# %fc1 : call_module[target=fc1](args=(%x,))
# %relu : call_function[target=torch.relu](args=(%fc1,))
# %fc2 : call_module[target=fc2](args=(%relu,))

# Analyzer identifies:
# - fc1.out_features (3072) → relu input
# - relu.output → fc2.in_features (3072)
# ← These are synchronized (cross-layer dependency)
```

**3. Fusion Opportunity Detection** - [`quantization/backends/gemm_registry.py`](modelopt/torch/quantization/backends/gemm_registry.py):
```python
class GEMMRegistry:
    def find_fusion_pattern(graph):
        """Detect fusable patterns in graph."""
        for node in graph.nodes:
            if _is_gemm_relu_gemm_pattern(node):
                # Register fusion kernel
                return FUSED_GEMM_RELU_GEMM_KERNEL
```

**4. Fused Kernel Application** - `backends/fp8_per_tensor_gemm.py`:
```python
def fused_gemm_relu_gemm(x, w1, w2):
    """
    Fused kernel combining:
    - GEMM 1 (x @ w1)
    - ReLU activation
    - GEMM 2 (relu_out @ w2)

    All in single CUDA kernel call.
    """
    return _cuda_fused_gemm_relu_gemm(x, w1, w2)
```

**5. CUDA Implementation**:
```cpp
// Fused GEMM-ReLU-GEMM kernel
__global__ void fused_gemm_relu_gemm_kernel(
    const float* x,      // Input
    const float* w1,     // First weight
    const float* w2,     // Second weight
    float* output,       // Final output
    int m, int k, int n
) {
    // Compute first GEMM in shared memory
    __shared__ float smem[BLOCK_SIZE];
    float gemm1_result = 0.0f;
    for (int i = 0; i < k; i++) {
        gemm1_result += x[i] * w1[i];
    }

    // Apply ReLU (fused, no memory write)
    float relu_result = max(0.0f, gemm1_result);

    // Compute second GEMM using intermediate result
    float gemm2_result = 0.0f;
    for (int j = 0; j < n; j++) {
        gemm2_result += relu_result * w2[j];
    }

    // Write final output (single memory write)
    output[blockIdx.x] = gemm2_result;
}
```

**Benefits**:
- **3 memory operations → 1** (read x, write output only)
- **Better cache utilization** (intermediate stays in registers/shared memory)
- **Reduced kernel launch overhead** (1 kernel vs 3)

## Entry Points

### For Users

**Quantization**:
```python
from modelopt.torch.quantization import quantize, calibrate
model = quantize(model, config, forward_loop=data_loader)
```
Entry: [`modelopt/torch/quantization/model_quant.py:132`](modelopt/torch/quantization/model_quant.py#L132)

**Tracing** (Advanced):
```python
from modelopt.torch.trace import recursive_trace, analyze_symbols
graphs = recursive_trace(model)
analyzer = analyze_symbols(model, graphs)
```
Entry: [`modelopt/torch/trace/tracer.py:310`](modelopt/torch/trace/tracer.py#L310)

**Export**:
```python
from modelopt.torch.export import export_to_onnx
export_to_onnx(model, output_path)
```
Entry: [`modelopt/torch/export/`](modelopt/torch/export/)

### For Developers

**Adding New Quantization Format**:
1. Implement kernel in `quantization/backends/`
2. Register in `quantization/tensor_quant.py`
3. Add CUDA extension in `quantization/src/`

**Adding Symbol Support for New Layer**:
1. Register in `trace/modules/nn.py` using `@SymMap.register()`
2. Define `SymInfo` with cross-layer parameters

**Adding Framework Plugin**:
1. Create plugin in `trace/plugins/` or `quantization/plugins/`
2. Implement framework-specific handling

## Testing

```
tests/
├── unit/torch/quantization/    # Unit tests for quantization
├── gpu/torch/                   # GPU-dependent tests
│   ├── quantization/            # Quantization integration tests
│   ├── export/                  # Export tests
│   └── trace/                   # Trace module tests
└── _test_utils/                 # Shared test utilities
```

Run tests: `pytest tests/gpu/torch/quantization/`

## Examples

```
examples/
├── llm_ptq/        # Post-training quantization for LLMs
├── llm_qat/        # Quantization-aware training
├── cnn_qat/        # CNN quantization examples
└── onnx_ptq/       # ONNX quantization examples
```

Each example demonstrates end-to-end workflow from model loading to quantization to export.

## Code Organization Principles

1. **Plugin Architecture** - Framework-specific code isolated in `plugins/` directories
2. **Registry Pattern** - Extensible registries for modules (`SymMap`), quantization modes, kernels
3. **Layered APIs** - High-level user APIs (`quantize()`) hide complex internals
4. **Separation of Concerns** - Trace, quantization, and export are independent modules
5. **Performance Critical in C++** - Quantization kernels in CUDA, exposed via Python

## Further Reading

- **Detailed Architecture**: [`claude/trace/STRUCTURE.md`](claude/trace/STRUCTURE.md)
- **Technical Deep Dive**: [`claude/trace/SUMMARY.md`](claude/trace/SUMMARY.md)
- **Quick Reference**: [`claude/trace/QUICK_REFERENCE.md`](claude/trace/QUICK_REFERENCE.md)
- **Getting Started**: [`claude/trace/README.md`](claude/trace/README.md)
- **Navigation Index**: [`claude/trace/INDEX.md`](claude/trace/INDEX.md)

## Key Numbers

- **Total Python files**: 214
- **Total lines of code**: ~61,440
- **Trace module**: 2,834 lines (tracer: 331, symbols: 545, analyzer: 1,368)
- **Quantization module**: 5,289 lines
- **Export module**: 8,218 lines
- **Supported formats**: 6+ (INT8, INT4, FP8, NVFP4, MX, etc.)
- **Framework integrations**: 5+ (HuggingFace, Megatron-LM, APEX, Accelerate, Transformer Engine)

---

**Generated**: October 26, 2025
**Repository**: [NVIDIA TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
**Maintainer**: NVIDIA
