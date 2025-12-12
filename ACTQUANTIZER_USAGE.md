# ActQuantizer: When and What It's Used For

## Overview

`ActQuantizer` is a PyTorch module that **quantizes activations** (intermediate values) flowing through neural network layers. It's used to reduce memory usage and computation cost by converting floating-point activations to lower-bit integers.

---

## What is ActQuantizer?

`ActQuantizer` is defined in `fake_quant/quant_utils.py`:

```python
class ActQuantizer(torch.nn.Module):
    """
    A class for quantizing the activations. We only support (both sym. and asym.) 
    per-token quantization for the activations.
    """
```

**Purpose**: Quantize activations (input/output values of layers) to reduce precision from FP16/FP32 to INT8/INT4.

**Key Features**:
- Supports **symmetric** and **asymmetric** quantization
- Supports **per-token** quantization (different scale per token)
- Supports **per-tensor** quantization (single scale for entire tensor)
- Supports both **static** (pre-computed scales) and **dynamic** (runtime-computed scales) quantization

---

## When is ActQuantizer Used?

### 1. **Model Setup Phase** - Wrapping Layers

`ActQuantizer` is **created** when wrapping linear/conv layers with `ActQuantWrapper`:

```python
# From quant_qwen2vl.py:101
quant_utils.qwen2vl_add_act_qaunt(model, args)
```

This function (`qwen2vl_add_act_qaunt`) wraps target layers:

```python
# From quant_utils.py:559-573
def qwen2vl_add_act_qaunt(model, args):
    if args.quant_llm:
        # Wrap all Linear layers in the LLM
        add_actquant(model.model.model, args.act_per_tensor)
    
    if args.quant_visual_clip:
        # Wrap visual encoder layers
        model.model.visual.patch_embed.proj = ActQuantWrapper(...)
        add_actquant(model.model.visual.blocks, args.act_per_tensor)
    
    if args.quant_cross_attention:
        # Wrap cross-attention layers
        add_actquant(model.model.visual.merger, args.act_per_tensor)
```

**What happens**: Each `torch.nn.Linear` layer is replaced with an `ActQuantWrapper`, which contains:
- The original `Linear` layer (`self.module`)
- An `ActQuantizer` for input activations (`self.quantizer`)
- An `ActQuantizer` for output activations (`self.out_quantizer`)

### 2. **Configuration Phase** - Setting Quantization Parameters

After wrapping, quantization parameters are configured:

```python
# From quant_qwen2vl.py:187-209
if args.llm_a_bits < 16 or args.llm_static:
    qlayers = quant_utils.find_qlayers(
        model.model.model, layers=[quant_utils.ActQuantWrapper]
    )
    for name in qlayers:
        qlayers[name].quantizer.configure(
            bits=layer_input_bits,        # e.g., 8 bits
            groupsize=layer_groupsize,     # Group size for quantization
            sym=layer_a_sym,              # Symmetric or asymmetric
            clip_ratio=layer_a_clip,      # Clipping ratio
            act_per_tensor=args.act_per_tensor,
            static=args.llm_static,       # Static or dynamic quantization
            observer_type="minmax",
        )
```

**What happens**: Each `ActQuantizer` is configured with:
- Bit width (e.g., 8 bits)
- Quantization mode (static/dynamic, symmetric/asymmetric)
- Observer type (for static quantization calibration)

### 3. **Calibration Phase** - Collecting Statistics (Static Quantization Only)

If `static=True`, calibration runs to collect activation statistics:

```python
# From quant_qwen2vl.py:217-218
if args.llm_static or args.visual_static:
    quant_utils.calib_qwen2vl_plus(model, args, dataset, args.calib_num)
```

During calibration:
```python
# From quant_utils.py:116-122
def forward(self, x):
    if self.static:
        if self.calibrate:
            # Collect min/max statistics
            self.quantizer.observer.update(x)
            if self.last_calibrate:
                # Finalize quantization parameters
                self.quantizer.update_quantization_params(x)
            return x  # Don't quantize during calibration
```

**What happens**: 
- Model runs forward passes on calibration data
- `ActQuantizer` collects min/max values via observer
- After calibration, scale/zero_point are computed and frozen

### 4. **Inference Phase** - Quantizing Activations

During inference, `ActQuantizer` quantizes activations in the forward pass:

```python
# From quant_utils.py:330-391 (ActQuantWrapper.forward)
def forward(self, x):
    # ... Hadamard rotation (if enabled) ...
    
    # Quantize INPUT activations
    if self.quantizer.static:
        x = self.quantizer(x)  # Use pre-computed scale/zero_point
    elif self.quantizer.bits < 16:
        self.quantizer.find_params(x)  # Compute scale/zero_point dynamically
        x = self.quantizer(x).to(x_dtype)
        self.quantizer.free()
    
    # Apply linear layer
    x = self.module(x).to(x_dtype)
    
    # Quantize OUTPUT activations
    if self.out_quantizer.bits < 16:
        self.out_quantizer.find_params(x)
        x = self.out_quantizer(x).to(x_dtype)
        self.out_quantizer.free()
    
    return x
```

**What happens**:
- **Input quantization**: Activations entering the layer are quantized
- **Layer computation**: Linear layer operates on quantized inputs
- **Output quantization**: Activations leaving the layer are quantized

---

## What is ActQuantizer Used For?

### 1. **Quantizing Input Activations**

Before a linear layer processes inputs, `ActQuantizer` quantizes them:

```
Original:  [0.234, -0.567, 1.234, ...]  (FP16, 16 bits)
           ↓ ActQuantizer.forward()
Quantized: [23, -57, 123, ...]          (INT8, 8 bits)
           ↓ Linear layer
Output:    [quantized computation]
```

**Benefits**:
- Reduces memory bandwidth (8 bits vs 16 bits)
- Enables INT8 matrix multiplication (faster on hardware)

### 2. **Quantizing Output Activations**

After a linear layer, `ActQuantizer` quantizes outputs:

```
Layer Output: [0.456, -0.789, 2.345, ...]  (FP16)
              ↓ out_quantizer.forward()
Quantized:    [46, -79, 234, ...]          (INT8)
              ↓ Next layer receives quantized input
```

**Benefits**:
- Maintains low precision throughout the network
- Reduces memory for intermediate activations

### 3. **Supporting Different Quantization Modes**

`ActQuantizer` adapts based on configuration:

#### Static Quantization Mode
```python
# Pre-computed scale/zero_point
x_quantized = (x / scale + zero_point).round().clamp(minq, maxq)
x_dequantized = (x_quantized - zero_point) * scale
```

#### Dynamic Quantization Mode
```python
# Compute scale/zero_point on-the-fly
scale = (x.max() - x.min()) / maxq
zero_point = round(-x.min() / scale)
x_quantized = (x / scale + zero_point).round().clamp(minq, maxq)
x_dequantized = (x_quantized - zero_point) * scale
```

### 4. **Per-Token vs Per-Tensor Quantization**

**Per-Token** (default):
- Each token in a sequence gets its own scale/zero_point
- Better accuracy, more computation overhead
- Example: `[batch_size, seq_len, hidden_dim]` → scale shape: `[batch_size, seq_len, 1]`

**Per-Tensor**:
- Single scale/zero_point for entire tensor
- Less accurate, less computation overhead
- Example: `[batch_size, seq_len, hidden_dim]` → scale shape: `[1]`

---

## Where is ActQuantizer Applied?

### For Qwen2-VL Specifically:

1. **LLM Layers** (`args.quant_llm=True`):
   - All `Linear` layers in `model.model.model` (transformer blocks)
   - Input/output activations of attention and MLP layers

2. **Visual Encoder Layers** (`args.quant_visual_clip=True`):
   - Patch embedding projection
   - All `Linear` layers in visual transformer blocks (`model.model.visual.blocks`)
   - MLP layers (`mlp.fc2`)

3. **Cross-Attention Layers** (`args.quant_cross_attention=True`):
   - Merger layers (`model.model.visual.merger`)
   - Layers connecting visual and text modalities

### Example: MLP Layer Quantization

```python
# Original MLP layer
class MLP(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(4096, 11008)  # Up projection
        self.fc2 = nn.Linear(11008, 4096)  # Down projection
    
    def forward(self, x):
        x = self.fc1(x)      # Input: FP16 activations
        x = gelu(x)          # Activation function
        x = self.fc2(x)      # Output: FP16 activations
        return x

# After wrapping with ActQuantWrapper
class MLP(nn.Module):
    def __init__(self):
        self.fc1 = ActQuantWrapper(nn.Linear(4096, 11008))
        # fc1.quantizer: quantizes input to fc1
        # fc1.out_quantizer: quantizes output of fc1
        self.fc2 = ActQuantWrapper(nn.Linear(11008, 4096))
        # fc2.quantizer: quantizes input to fc2
        # fc2.out_quantizer: quantizes output of fc2
    
    def forward(self, x):
        x = self.fc1(x)      # Input quantized → fc1 → output quantized
        x = gelu(x)          # Activation function (on quantized values)
        x = self.fc2(x)      # Input quantized → fc2 → output quantized
        return x
```

---

## Key Code Locations

### Creation
- `quant_utils.py:97-110`: `ActQuantizer` class definition
- `quant_utils.py:286`: Created inside `ActQuantWrapper.__init__()`
- `quant_utils.py:626-662`: `add_actquant()` wraps layers with `ActQuantWrapper`

### Configuration
- `quant_qwen2vl.py:187-209`: LLM activation quantization configuration
- `quant_qwen2vl.py:162-185`: Visual activation quantization configuration
- `quant_utils.py:142-179`: `ActQuantizer.configure()` method

### Usage During Forward Pass
- `quant_utils.py:116-133`: `ActQuantizer.forward()` - quantizes activations
- `quant_utils.py:330-391`: `ActQuantWrapper.forward()` - uses `ActQuantizer` before/after layer

### Calibration
- `quant_utils.py:1105-1129`: `calib_qwen2vl_plus()` - calibration loop
- `quant_utils.py:678-711`: Calibration control functions (`model_open_calibrate`, etc.)

---

## Summary

| Aspect | Details |
|--------|---------|
| **What** | Quantizes activations (intermediate values) in neural networks |
| **When Created** | During model setup, when wrapping layers with `ActQuantWrapper` |
| **When Configured** | After wrapping, sets quantization parameters (bits, mode, etc.) |
| **When Calibrated** | Before inference (static mode only), collects statistics |
| **When Used** | Every forward pass, quantizes inputs/outputs of layers |
| **Where Applied** | LLM layers, visual encoder layers, cross-attention layers |
| **Purpose** | Reduce memory usage, enable faster INT8 computation, maintain accuracy |

**Key Insight**: `ActQuantizer` is the core component that enables activation quantization in MQuant. It wraps around linear layers and quantizes activations flowing through them, supporting both static (pre-computed) and dynamic (runtime-computed) quantization modes.

