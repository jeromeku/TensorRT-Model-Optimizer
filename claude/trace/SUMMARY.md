# TensorRT Model Optimizer - Codebase Exploration Summary

This document summarizes the comprehensive exploration of the TensorRT Model Optimizer repository, providing you with the detailed mapping requested in CLAUDE.md.

## Executive Summary

The TensorRT Model Optimizer is a sophisticated PyTorch model optimization library (~214 Python files, 61K total lines) focused on quantization, pruning, distillation, and other optimization techniques. The **trace module** (2,834 lines) is the foundational component that extracts and analyzes model structure, enabling intelligent optimization decisions throughout the system.

---

## 1. Overall Codebase Organization

### Main Directory Structure

The codebase is organized into three main domains:

```
modelopt/
├── torch/           - PyTorch-specific optimizations (13 submodules)
├── onnx/            - ONNX-specific support  
└── deploy/          - Deployment utilities
```

### PyTorch Module Organization (13 submodules)

| Module | Purpose | Size |
|--------|---------|------|
| **trace** | Graph extraction and symbol analysis | 2,834 lines |
| **quantization** | Quantization algorithms & kernels | 5,289 lines |
| **export** | Model export (ONNX, TRT-LLM) | 8,218 lines |
| opt | Optimization modes & search | 1,359 lines |
| nas | Neural architecture search | 856+ lines |
| distill | Knowledge distillation | - |
| sparsity | Sparsity techniques | - |
| prune | Pruning algorithms | - |
| speculative | Speculative decoding | 2,149 lines (megatron plugin) |
| _deploy | Deployment runtime | - |
| utils | Shared utilities (658 lines network.py) | - |

### Code Distribution

- **Python**: 214 files (61,440 lines in modelopt/)
- **C++/CUDA**: Core extensions in quantization/src/
- **Triton**: Modern GPU kernels in quantization/triton/

### Python vs C++ Organization

**Python Layer** (user-facing APIs):
- Model quantization APIs (model_quant.py, model_calib.py)
- Configuration system (config.py)
- Framework plugins (huggingface.py, megatron.py, etc.)

**C++ Layer** (performance-critical):
- CUDA kernels for FP8, INT4, NVFP4, MX formats
- Fused quantization operations
- Triton JIT-compiled kernels

---

## 2. Trace Module Deep Dive

### Trace Directory Structure

```
modelopt/torch/trace/
├── __init__.py              - Public API exports
├── tracer.py                - FX graph tracing (331 lines)
├── symbols.py               - Symbol definitions (545 lines)
├── analyzer.py              - Dependency analysis (1,368 lines)
├── modules/
│   ├── nn.py                - PyTorch symbol info (110 lines)
│   └── concat.py            - Concat layer handling (480 lines)
└── plugins/
    ├── transformers.py      - HuggingFace integration
    └── megatron.py          - Megatron-LM integration
```

### What Files Do

| File | Lines | Core Responsibility |
|------|-------|---------------------|
| **tracer.py** | 331 | Wraps PyTorch's FX tracer; handles unsupported ops by treating them as leaf modules |
| **symbols.py** | 545 | Defines `Symbol` abstraction for model parameters; tracks cross-layer dependencies |
| **analyzer.py** | 1,368 | Analyzes FX graphs to find parameter relationships; determines which must be synchronized |
| **modules/nn.py** | 110 | Registers symbol info for standard PyTorch modules (Linear, Conv, etc.) |
| **modules/concat.py** | 480 | Special handling for concatenation layers (complex symbol merging) |

### Key Classes in Trace Module

**1. RobustTracer (tracer.py:33-331)**
```python
class RobustTracer:
    """Augmented FX tracer that handles unsupported operations gracefully."""
    
    def trace(model, concrete_args=None) -> fx.Graph
        # Tries to trace, wraps failing modules as leaves, retries
        # Returns GraphCollection with all traced subgraphs
    
    @classmethod
    def register_leaf(cls, module_type) -> None
        # Mark module types to never trace into
```

**2. GraphCollection (tracer.py:239-308)**
```python
class GraphCollection:
    """Container for all traced FX graphs."""
    
    def __getitem__(module) -> fx.Graph      # Get graph for module
    def is_failed(module) -> bool            # Did this module fail to trace?
    def failure_msg(module) -> str|None      # Why did it fail?
    def is_unvisited(module) -> bool         # Was this module ever attempted?
    def __iter__() -> Generator[module]      # Iterate over traced modules
```

**3. Symbol (symbols.py:29-244)**
```python
class Symbol:
    """Abstraction for model parameters that affect tensor shapes."""
    
    # States (mutually exclusive):
    # - free: unbound, searchable
    # - constant: fixed
    # - dynamic: depends on parent symbol
    
    # Cross-layer significance (CLType):
    # - INCOMING: input tensor depends on this
    # - OUTGOING: output shape depends on this
    # - NONE: internal only
    
    def link_to(parent_symbol) -> None      # Make dynamic (depend on parent)
    def disable() -> None                    # Mark as constant
    
    @property
    def is_cross_layer -> bool               # INCOMING or OUTGOING?
    @property
    def is_dangling -> bool                  # free AND cross_layer?
    @property
    def is_searchable -> bool                # Can be optimized?
```

**4. SymMap (symbols.py:255-437)**
```python
class SymMap:
    """Registry mapping module types to their symbols."""
    
    @classmethod
    def register(module_type) -> decorator   # Register symbol info
    
    def get_sym_info(module) -> SymInfo      # Get symbols for module
    def named_symbols(module) -> Iterator    # All symbols
    def named_in_symbols(module) -> Iterator # Incoming symbols
    def named_out_symbols(module) -> Iterator# Outgoing symbols
    def named_dangling_symbols(module) -> Iterator  # Unbound cross-layer
```

**5. GraphDependencyProcessor (analyzer.py:433-1368)**
```python
class GraphDependencyProcessor:
    """Analyzes dependencies in a traced graph."""
    
    def process() -> None                    # Analyze all nodes
    
    @classmethod
    def register_node_processor(processor_class) -> decorator
        # Register custom logic for specific node types
```

### How Trace APIs Work

**Top-Level API - recursive_trace()**
```python
# From tracer.py:310-331
def recursive_trace(model: nn.Module, concrete_args=None) -> GraphCollection:
    """Main entry point for tracing a model."""
    gc = GraphCollection()
    gc.recursive_trace(model, concrete_args)
    return gc

# Usage:
from modelopt.torch.trace import recursive_trace

model = MyModel()
graphs = recursive_trace(model)

# Access traced graphs
for module in graphs:
    fx_graph = graphs[module]
    print(f"{module}: {fx_graph}")
    
    if graphs.is_failed(module):
        print(f"Failed: {graphs.failure_msg(module)}")
```

**Internal Flow**
```
recursive_trace(model)
    ↓ [creates GraphCollection]
GraphCollection.recursive_trace()
    ↓ [enters recursion loop]
RobustTracer.trace() attempts tracing
    ↓ [if fails, marks module]
Recurse on children of failed modules
    ↓ [repeats until all traced]
Result: GraphCollection with graphs for all modules
```

### What Analysis It Performs

**1. Graph Structure Extraction** (tracer.py)
- Converts PyTorch model into FX (functional) graph representation
- Nodes represent operations, edges represent data flow
- Preserves module hierarchy for layer-wise analysis

**2. Symbol Identification** (symbols.py)
- Maps module parameters to `Symbol` objects
- Example: `nn.Linear` has `in_features` (INCOMING) and `out_features` (OUTGOING)
- Classifies symbols as cross-layer or internal

**3. Dependency Graph Analysis** (analyzer.py:1-200)
```python
# From analyzer.py - Map class tracking dependencies
class Map:
    def create_root(node, target, id, is_free, priority) -> None
        # Create new dependency entry
    
    def link_nodes(node, other_node) -> None
        # Create dependency between nodes
    
    def root(node) -> Node
        # Get root node this depends on
```

**4. Cross-Layer Synchronization** (analyzer.py:321-388)
```python
def _synchronize_nodes(nodes: list[Node], disable=False):
    """Match and link symbols across layer boundaries.
    
    Algorithm:
    1. Find first node in DAG order (will be "free" node)
    2. For each other input node:
       - Extract its searchable output symbols
       - Try to match with first node's symbols
       - If match found: link them together (make dynamic)
       - If no match: disable symbols (make constant)
    """
```

### How It Modifies User Torch Code

**The trace module does NOT modify the original model**. Instead it:

1. **Creates a functional representation** - Converts model to FX graph
   - Preserves original model untouched
   - Annotations available in GraphCollection

2. **Annotates with dependency info** - Adds metadata about parameter relationships
   - Symbol states stored in Map data structure
   - Dependency links recorded but not in model

3. **Provides query interface** - Users access via:
   ```python
   graphs = recursive_trace(model)
   
   # Query traced info
   fx_graph = graphs[linear_layer]          # Get FX graph
   failed = graphs.is_failed(conv_layer)    # Check if failed
   ```

4. **Downstream consumption** - Quantization module uses trace output:
   ```python
   # In quantization/model_quant.py
   def quantize(model, config):
       # Internally calls trace to understand model structure
       # Uses dependencies to make smart quantization choices
       # Returns modified model with QuantModules inserted
   ```

---

## 3. Downstream Use in Optimization

### How Trace Feeds into Quantization

**Quantization Flow (from quantization/model_quant.py)**

```python
# User API at line 132-224
def quantize(model, config, forward_loop=None):
    """Main quantization entry point."""
    
    # Step 1: Model preparation
    model.eval()
    
    # Step 2: Apply quantization mode
    # ← Internally uses trace to understand model structure
    model = apply_mode(
        model,
        mode=[("quantize", config)],
        registry=QuantizeModeRegistry
    )
    
    # Step 3: Calibrate on data
    model = calibrate(model, forward_loop)
    
    return model
```

**Tracing Happens Inside opt/dynamic.py**
```python
# The opt module uses trace when resolving configurations
# Location: opt/dynamic.py (1,359 lines)
```

### Example: NVFP4 Quantization with Tracing

**NVFP4 = NVIDIA's 4-bit floating point format**

Quantization Config Example:
```python
from modelopt.torch import quantization as mtq

NVFP4_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "num_bits": 4,
            "format": "nvfp4"
        },
        "*input_quantizer": {
            "num_bits": 8,
            "format": "fp8"
        }
    },
    "algorithm": "max"
}

# Usage
model = quantize(model, NVFP4_CFG, forward_loop=data_loader)
```

**Behind the Scenes (Trace Role)**
1. Trace extracts graph → identifies linear layers
2. Analyzer finds which layers' outputs feed to same inputs
3. These can use shared quantization (grouped quantization)
4. Config applied with this knowledge

### Example: Kernel Fusion

**Kernel Fusion = Combining multiple ops into single fused kernel**

The trace module identifies fusion opportunities:
```python
# Example: Detect Linear → ReLU → Linear pattern

# After tracing and analysis, quantization can identify:
# - Linear output feeds to ReLU
# - ReLU output feeds to next Linear
# - Can fuse: (Linear + ReLU + Linear) -> single fused kernel

# This requires knowing exact connectivity -> from trace!
```

**Related Code in quantization/backends/**
```
backends/
├── fp8_per_tensor_gemm.py     # Fused FP8 GEMM
├── nvfp4_gemm.py              # Fused NVFP4 GEMM
└── gemm_registry.py           # Kernel dispatch based on graph analysis
```

### APIs Used by Downstream Modules

**From quantization/mode.py**
```python
class QuantizeModeRegistry:
    """Registry for quantization modes."""
    
    def register(mode_name) -> decorator
    def get_mode(name) -> QuantizationMode
```

**From quantization/conversion.py**
```python
def set_quantizer_by_cfg(model, config) -> None:
    """Apply quantization config to model.
    
    Uses traced graph info to:
    1. Find all quantizable modules
    2. Match config patterns to modules
    3. Insert TensorQuantizer wrappers
    """
```

**From opt/searcher.py** (1,359 lines)
```python
def search_space_optimization():
    """Auto-optimize quantization config.
    
    Uses traced symbol information to:
    1. Identify search dimensions (cross-layer params)
    2. Propose configurations respecting constraints
    3. Evaluate candidates
    """
```

---

## 4. Key Public APIs Summary

### Trace Module Public APIs

**Location**: `/modelopt/torch/trace/__init__.py`

```python
# Exported from __init__.py (lines 16-26)
from .tracer import recursive_trace, GraphCollection, RobustTracer
from .symbols import Symbol, SymMap, SymInfo
from .analyzer import analyze_symbols
from . import plugins

# Public API Functions:

def recursive_trace(
    model: nn.Module, 
    concrete_args: dict[str, Any] | None = None
) -> GraphCollection:
    """Extract and trace all graphs in model.
    
    Returns:
        GraphCollection with FX graphs for all modules
    """

def analyze_symbols(
    model: nn.Module,
    gc: GraphCollection,
    sym_map: SymMap | None = None
) -> Analyzer:
    """Analyze symbol dependencies in model.
    
    Returns:
        Analyzer object with dependency info
    """

# Core Classes (user-facing)

class GraphCollection:
    """Container for traced FX graphs."""
    def __getitem__(module) -> fx.Graph
    def is_failed(module) -> bool
    def failure_msg(module) -> str | None
    def __iter__() -> Generator

class Symbol:
    """Represents a model parameter affecting shapes."""
    def link_to(parent: Symbol) -> None
    def disable() -> None
    @property
    def is_cross_layer() -> bool
    @property
    def is_dangling() -> bool

class SymMap:
    """Registry of symbols per module type."""
    @classmethod
    def register(module_type) -> decorator
    def get_sym_info(module) -> SymInfo
    def named_symbols(module)
    def named_dangling_symbols(module)
```

### Quantization Module Public APIs

**Location**: `/modelopt/torch/quantization/__init__.py`

```python
from .model_quant import quantize, calibrate, auto_quantize
from .config import QuantizeConfig, QuantizerAttributeConfig
from .conversion import set_quantizer_by_cfg

def quantize(
    model: nn.Module,
    config: dict | QuantizeConfig,
    forward_loop: Callable | None = None
) -> nn.Module:
    """Apply quantization to model.
    
    Args:
        model: PyTorch model
        config: Quantization configuration
        forward_loop: Data loader for calibration
    
    Returns:
        Quantized model ready for inference
    """

def calibrate(
    model: nn.Module,
    algorithm: str = "max",
    forward_loop: Callable | None = None
) -> nn.Module:
    """Calibrate quantization scales using data."""

def auto_quantize(
    model: nn.Module,
    forward_loop: Callable,
    max_trials: int | None = None
) -> nn.Module:
    """Automatically search for optimal quantization config."""
```

### Entry Points in modelopt.torch

**From /modelopt/torch/__init__.py**
```python
from . import distill, nas, opt, prune, quantization, sparsity, speculative, utils

# Main sub-packages available:
import modelopt.torch.quantization        # Quantization APIs
import modelopt.torch.trace                # Tracing APIs
import modelopt.torch.export               # Export functionality
import modelopt.torch.opt                  # Optimization modes
```

---

## 5. Quantization Details: FP8 and NVFP4

### FP8 (8-bit Floating Point) Quantization

**Configuration (from quantization/config.py)**
```python
FP8_DEFAULT_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "num_bits": 8,
            "format": "fp8"  # Can be e4m3 or e5m2
        },
        "*input_quantizer": {
            "num_bits": 8,
            "format": "fp8"
        }
    },
    "algorithm": "max"
}
```

**Implementation (quantization/tensor_quant.py:47-99)**
```python
def scaled_e4m3_impl(
    inputs: torch.Tensor,
    amax: torch.Tensor,
    disable_fused_kernel=True,
) -> torch.Tensor:
    """Fake quantize to FP8 (e4m3 format)."""
    
    cuda_ext_fp8 = get_cuda_ext_fp8(raise_if_failed=True)
    
    if is_fusable():
        # Use fused CUDA kernel
        outputs = cuda_ext_fp8.fused_fake_e4m3fy(inputs, amax.float(), threshold)
    else:
        # Fallback: scale, fake quantize, unscale
        scale = 448.0 / amax  # Max representable in e4m3
        outputs = cuda_ext_fp8.fake_e4m3fy(inputs * scale) / scale
    
    return outputs
```

**CUDA Kernel (quantization/src/)**
- Located in C++/CUDA extensions
- Optimized for NVIDIA GPUs
- Implements fused operations to minimize memory bandwidth

### NVFP4 (4-bit Floating Point) Quantization

**New Format (2025 release)**
- More aggressive compression than INT4
- Scales better for LLMs
- Supported on Blackwell and newer GPUs

**Configuration**
```python
NVFP4_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "num_bits": 4,
            "format": "nvfp4"
        },
        "*input_quantizer": {
            "num_bits": 8,
            "format": "fp8"
        }
    },
    "algorithm": "max"
}
```

**Implementation (quantization/backends/nvfp4_gemm.py:1-60)**
```python
class NVFP4Kernel:
    """Optimized NVFP4 GEMM (matrix multiply) kernel."""
    
    def apply_quantization(weights, scales, offsets):
        """Convert weights to NVFP4 representation."""
        # Uses Triton or CUDA kernels
    
    def gemm(
        input_fp8: Tensor,      # Activation (FP8)
        weights_nvfp4: Tensor,  # Weights (NVFP4)
        scales: Tensor
    ) -> Tensor:
        """Fused quantized matrix multiply."""
```

### Kernel Fusion Example

**Pattern Recognition via Trace**
```
# Original model:
x = linear1(x)           # Output shape: (batch, 2048)
x = relu(x)              # Same shape, inplace
x = linear2(x)           # Input shape: (batch, 2048)

# After trace analysis:
# - Identifies linear1 output shape
# - Sees relu doesn't change shape
# - Finds linear2 consumes this exact shape

# Enables fusion:
# Fused kernel: (GEMM + ReLU + GEMM) in one call
# Benefit: Reduced memory bandwidth, better cache usage
```

**Fusion Registration** (backends/gemm_registry.py:1-60)
```python
class GEMMRegistry:
    """Registry of available GEMM kernels."""
    
    def register_kernel(pattern, kernel_impl):
        """Register new fusion pattern and kernel."""
    
    def find_kernel(input_type, weight_type, output_type):
        """Find best kernel for pattern."""
        # Uses trace info to match pattern
        # Returns optimized kernel
```

---

## 6. Specific File Mappings

### User-Facing Quantization APIs

| User Function | File | Lines | What It Does |
|---------------|------|-------|-------------|
| `quantize()` | quantization/model_quant.py | 132-224 | Main entry point, applies config and calibrates |
| `calibrate()` | quantization/model_quant.py | 54-114 | Runs calibration algorithm on data |
| `auto_quantize()` | quantization/model_quant.py | 228+ | Searches for best quantization config |
| `enable_quantizer()` | quantization/model_quant.py | - | Re-enable disabled quantizers |
| `disable_quantizer()` | quantization/model_quant.py | - | Disable specific quantizers |

### Trace APIs

| User Function | File | Lines | What It Does |
|---------------|------|-------|-------------|
| `recursive_trace()` | trace/tracer.py | 310-331 | Main entry, returns GraphCollection |
| `GraphCollection.__iter__()` | trace/tracer.py | 247-249 | Iterate over traced modules |
| `analyze_symbols()` | trace/analyzer.py | 1350+ | Extract symbol dependencies |

### Core Quantization Classes

| Class | File | Lines | Purpose |
|-------|------|-------|---------|
| `TensorQuantizer` | nn/modules/tensor_quantizer.py | 1-200 | Core fake quantization operator |
| `QuantLinear` | nn/modules/quant_linear.py | 1-100 | Quantized linear layer |
| `QuantConv2d` | nn/modules/quant_conv.py | 1-100 | Quantized convolution |
| `QuantModule` | nn/modules/quant_module.py | 1-100 | Base class for quantized modules |

### Configuration and Conversion

| Class/Function | File | Lines | Purpose |
|---|---|---|---|
| `QuantizeConfig` | config.py | 1-100 | Main config class |
| `QuantizerAttributeConfig` | config.py | 100-300 | Per-quantizer config |
| `set_quantizer_by_cfg()` | conversion.py | 1-100 | Apply config to model |
| `set_quantizer_attribute()` | conversion.py | 100-200 | Set individual quantizer attrs |

### Calibration

| Function | File | Lines | Purpose |
|---|---|---|---|
| `calibrate_amax()` | model_calib.py | - | Calibration main loop |
| `histogram_calibration()` | calib/histogram.py | 1-200 | Histogram-based statistics |
| `smoothquant_calibration()` | model_calib.py | - | SmoothQuant algorithm |
| `awq_calibration()` | model_calib.py | - | AWQ (Activation-aware Weight Quantization) |

---

## 7. Code Examples: Complete Flow

### Example 1: Simple INT8 Quantization

```python
import torch
import torch.nn as nn
from modelopt.torch.quantization import quantize, calibrate

# User model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(768, 3072)
        self.linear2 = nn.Linear(3072, 768)
    
    def forward(self, x):
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        return x

# Initialization
model = MyModel().cuda().eval()

# Quantization configuration
INT8_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": 8},
        "*input_quantizer": {"num_bits": 8},
    },
    "algorithm": "max"
}

# Calibration data
def calibration_loop(model):
    for batch in calibration_dataloader:
        model(batch.cuda())

# Apply quantization
# Behind the scenes:
# 1. recursive_trace(model) extracts FX graphs
# 2. analyze_symbols() finds cross-layer dependencies
# 3. QuantModules inserted with TensorQuantizer wrappers
# 4. calibrate() runs forward pass to compute scales
model = quantize(model, INT8_CFG, forward_loop=calibration_loop)

# Export
model.save_pretrained("quantized_model")
```

### Example 2: Using Trace Directly

```python
from modelopt.torch.trace import recursive_trace, analyze_symbols

model = MyModel()

# Trace the model
graphs = recursive_trace(model)

# Query traced information
print("Traced modules:")
for module in graphs:
    graph = graphs[module]
    print(f"  {type(module).__name__}: {len(list(graph.nodes))} nodes")
    
    if graphs.is_failed(module):
        print(f"    FAILED: {graphs.failure_msg(module)}")

# Analyze symbols
analyzer = analyze_symbols(model, graphs)

# Find cross-layer parameters
from modelopt.torch.trace import Symbol
for module, symbols in analyzer.named_modules():
    for sym_name, sym in symbols.named_dangling_symbols(module):
        print(f"Optimizable param: {module}.{sym_name}")
```

### Example 3: NVFP4 Quantization (Advanced)

```python
from modelopt.torch.quantization import quantize
from modelopt.torch import quantization as mtq

NVFP4_CFG = {
    "quant_cfg": {
        # Linear layer weights: NVFP4 (4-bit)
        "*.linear.weight_quantizer": {
            "num_bits": 4,
            "format": "nvfp4"
        },
        # Linear layer activations: FP8
        "*.linear.input_quantizer": {
            "num_bits": 8,
            "format": "fp8"
        },
        # LM head: special case (maybe no quantization)
        "*lm_head*": {
            "num_bits": 8,
        },
        # Default for everything else
        "default": {
            "num_bits": 8,
            "format": "fp8"
        }
    },
    "algorithm": "max"
}

def calib_loop(model):
    # Calibration on representative data
    for i, batch in enumerate(calib_dataloader):
        if i >= 32:  # Use first 32 batches
            break
        model(batch.cuda())

# Apply NVFP4 quantization
# Backends will automatically use fused NVFP4 GEMM kernels
model = quantize(model, NVFP4_CFG, forward_loop=calib_loop)

# Export for deployment
from modelopt.torch.export import export_torch_model
export_torch_model(model, "nvfp4_model.pt")
```

---

## 8. File and Line Number Reference

### Critical Code Paths

#### Trace Initialization Path
```
modelopt/torch/trace/__init__.py:18-24
    ├─ tracer.py:330
    │  └─ class GraphCollection:239-308
    ├─ symbols.py:26
    │  └─ class Symbol:29-244
    └─ analyzer.py:42
       └─ class GraphDependencyProcessor:433-500
```

#### Quantization Entry Path
```
modelopt/torch/quantization/model_quant.py:132-224
    ├─ quantization/mode.py:50-100
    │  └─ QuantizeModeRegistry
    ├─ quantization/conversion.py:1-100
    │  └─ set_quantizer_by_cfg()
    └─ quantization/model_calib.py:1-100
       └─ calibrate_amax()
```

#### Quantization Execution Path
```
quantization/nn/modules/quant_module.py:1-100
    ├─ TensorQuantizer.forward():tensor_quantizer.py:200-300
    │  └─ backends/fp8_per_tensor_gemm.py:1-60 [FP8 kernel]
    │  └─ backends/nvfp4_gemm.py:1-60 [NVFP4 kernel]
    └─ model_calib.py:100-200 [Calibration loop]
```

---

## 9. Quick Reference: Important Line Numbers

### Trace Module
- `RobustTracer` class: tracer.py:33-194
- `GraphCollection` class: tracer.py:239-308
- `recursive_trace()` function: tracer.py:310-331
- `Symbol` class: symbols.py:29-244
- `SymMap` class: symbols.py:255-437
- `GraphDependencyProcessor`: analyzer.py:433-500

### Quantization Module
- `quantize()` function: model_quant.py:132-224
- `calibrate()` function: model_quant.py:54-114
- `TensorQuantizer` class: nn/modules/tensor_quantizer.py:1-300
- `QuantLinear` class: nn/modules/quant_linear.py:1-100
- FP8 kernels: backends/fp8_per_tensor_gemm.py:1-250
- NVFP4 kernels: backends/nvfp4_gemm.py:1-280

### Configuration
- `QuantizeConfig` class: config.py:1-150
- `QuantizerAttributeConfig` class: config.py:150-300
- Pre-built configs: config.py:600+

---

## 10. Architecture Patterns Used

### Design Patterns Observed

1. **Visitor Pattern** (Node processing)
   - NodeProcessor visits each node
   - GraphDependencyProcessor orchestrates

2. **Registry Pattern** (Module registration)
   - SymMap.register() decorators
   - QuantModuleRegistry
   - QuantizeModeRegistry

3. **Facade Pattern** (Simple public APIs)
   - High-level quantize(), calibrate()
   - Hide internal complexity

4. **Strategy Pattern** (Algorithm selection)
   - Different calibration algorithms
   - Different quantization formats

5. **Plugin Architecture**
   - Framework-specific plugins
   - HuggingFace, Megatron support

---

## Conclusion

The TensorRT Model Optimizer is a well-structured system with clear separation of concerns:

1. **Trace Module** - Extracts model structure and identifies optimization opportunities
2. **Quantization Module** - Applies quantization with various formats (INT8, FP8, NVFP4)
3. **Export Module** - Packages optimized models for deployment
4. **Downstream Modules** - Use trace/quant for pruning, distillation, etc.

The trace module is fundamental, enabling intelligent quantization decisions by analyzing layer-wise dependencies. All quantization strategies benefit from this graph-level understanding.

