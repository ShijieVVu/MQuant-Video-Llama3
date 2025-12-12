# MQuant for Qwen2-VL: Code Walkthrough

This document explains how MQuant implements quantization for Qwen2-VL, focusing on:
1. **Dynamic vs Static Quantization**
2. **Prefill Stage vs Decoding Stage**

---

## 1. Dynamic vs Static Quantization

### Overview

**Dynamic Quantization**: Scale and zero-point are computed **on-the-fly** during inference for each input tensor.

**Static Quantization**: Scale and zero-point are **pre-computed during calibration** and then **fixed** for inference.

### Code Implementation

#### ActQuantizer Class (`fake_quant/quant_utils.py`)

The `ActQuantizer` class handles both modes:

```python
class ActQuantizer(torch.nn.Module):
    def __init__(self, act_per_tensor=False):
        super(ActQuantizer, self).__init__()
        self.register_buffer("scale", torch.zeros(1))
        self.register_buffer("zero", torch.zeros(1))
        self.bits = 16
        self.static = False  # Flag to switch between modes
        self.calibrate = False  # Used during calibration phase
        self.quant = False      # Used during quantized inference
```

#### Forward Pass Logic (`quant_utils.py:116-133`)

The `forward()` method shows the key difference:

```python
def forward(self, x):
    if self.static:
        # STATIC MODE: Uses pre-computed quantization parameters
        if self.calibrate:
            # During calibration: collect statistics, don't quantize
            self.quantizer.observer.update(x)
            if self.last_calibrate:
                # Final calibration step: compute and freeze parameters
                self.quantizer.update_quantization_params(x)
            return x  # Return unquantized during calibration
        elif self.quant:
            # During inference: use pre-computed scale/zero_point
            return self.quantizer(x)  # Uses self.scale and self.zero_point
        else:
            return x
    else:
        # DYNAMIC MODE: Compute quantization parameters on-the-fly
        x_dtype = x.dtype
        if self.bits == 16:
            return x
        elif self.sym:
            # Compute scale dynamically and quantize
            return sym_quant_dequant(x, self.scale, self.maxq).to(x_dtype)
        return asym_quant_dequant(x, self.scale, self.zero, self.maxq).to(x_dtype)
```

#### Dynamic Quantization Flow (`quant_utils.py:205-268`)

In dynamic mode, `find_params()` is called **every forward pass**:

```python
def find_params(self, x):
    """Compute scale and zero_point dynamically from input tensor"""
    if self.act_per_tensor:
        # Per-tensor: single scale/zero for entire tensor
        xmin = torch.minimum(x.min(), tmp) * self.clip_ratio
        xmax = torch.maximum(x.max(), tmp) * self.clip_ratio
        self.scale = (xmax - xmin) / self.maxq
        self.zero = torch.round(-xmin / self.scale)
    else:
        # Per-token: different scale/zero for each token
        reshaped_x = x.reshape((-1, x.shape[-1]))
        xmin = torch.minimum(reshaped_x.min(1)[0], tmp) * self.clip_ratio
        xmax = torch.maximum(reshaped_x.max(1)[0], tmp) * self.clip_ratio
        # ... compute per-token scales ...
```

#### Static Quantization Setup (`quant_utils.py:142-179`)

When `static=True`, an observer is created to collect statistics:

```python
def configure(self, ..., static=False, observer_type="minmax", ...):
    self.static = static
    if self.static:
        # Create observer to collect min/max statistics
        self.observer = build_observer(
            observer_type,  # "minmax" observer
            module_a_type,
            bit_type_a,
            calibration_mode,
        )
        # Create quantizer that will use observer's statistics
        self.quantizer = build_quantizer(
            "uniform", bit_type_a, self.observer, module_a_type
        )
        self.calibrate = False
        self.quant = False
```

#### Observer Pattern (`observer/minmax.py`)

The `MinmaxObserver` collects min/max values during calibration:

```python
class MinmaxObserver(BaseObserver):
    def update(self, v):
        """Accumulate min/max statistics across calibration data"""
        cur_max = v.max(axis=-1).values
        if self.max_val is None:
            self.max_val = torch.max(cur_max, torch.zeros_like(cur_max))
        else:
            self.max_val = torch.max(cur_max, self.max_val)  # Track global max
        
        cur_min = v.min(axis=-1).values
        if self.min_val is None:
            self.min_val = torch.min(cur_min, torch.zeros_like(cur_min))
        else:
            self.min_val = torch.min(cur_min, self.min_val)  # Track global min
    
    def get_quantization_params(self, *args, **kwargs):
        """Compute final scale and zero_point from accumulated statistics"""
        scale = (max_val - min_val) / float(qmax - qmin)
        zero_point = qmin - torch.round(min_val / scale)
        return scale, zero_point
```

### Configuration in Main Script (`exam/quant_qwen2vl.py`)

The quantization mode is controlled by flags:

```python
# Lines 187-209: LLM activation quantization
if args.llm_a_bits < 16 or args.llm_static:
    qlayers[name].quantizer.configure(
        bits=layer_input_bits,
        groupsize=layer_groupsize,
        sym=layer_a_sym,
        clip_ratio=layer_a_clip,
        act_per_tensor=args.act_per_tensor,
        static=args.llm_static,  # <-- Controls static vs dynamic
        observer_type="minmax",
    )

# Lines 162-185: Visual activation quantization  
if args.visual_a_bits < 16 or args.visual_static:
    qlayers[name].quantizer.configure(
        static=args.visual_static,  # <-- Controls static vs dynamic
        ...
    )
```

---

## 2. Prefill Stage vs Decoding Stage

### Overview

**Prefill Stage**: Processes the entire input sequence (prompt + images) in parallel. All tokens attend to all previous tokens.

**Decoding Stage**: Generates tokens one-by-one. Each new token only attends to previous tokens (cached KV).

### Calibration Strategy (`quant_utils.py:1105-1129`)

MQuant's calibration function `calib_qwen2vl_plus()` handles both stages:

```python
def calib_qwen2vl_plus(model, args, dataset, calib_num):
    model_open_calibrate(model.model, args)  # Enable calibration mode
    max_new_tokens = model.generate_kwargs["max_new_tokens"]
    
    # MOST CALIBRATION SAMPLES: Prefill + Multiple Decoding Steps
    model.generate_kwargs["max_new_tokens"] = 20  # <-- Prefill + ~20 decoding steps
    
    for i in tqdm(range(0, lt, step)):
        if i + step >= lt:
            # LAST CALIBRATION SAMPLE: Prefill + Single Decoding Step
            print("last calibrate")
            model_open_last_calibrate(model.model, args)  # Finalize parameters
            model.generate_kwargs["max_new_tokens"] = 1   # <-- Only 1 decoding step
        
        model.generate(message=struct, dataset=args.dataset_name)
        # During generate():
        #   - Prefill: processes full prompt (long sequence)
        #   - Decoding: generates max_new_tokens tokens (short sequences)
    
    model_close_calibrate(model.model, args)  # Disable calibration
    model_quant(model.model, args)            # Enable quantization
```

### Why Different max_new_tokens?

1. **Most samples (`max_new_tokens=20`)**: 
   - Captures statistics from **prefill** (long sequences) 
   - AND **multiple decoding steps** (short sequences)
   - Gives observer a mix of both patterns

2. **Last sample (`max_new_tokens=1`)**:
   - Finalizes quantization parameters
   - Ensures parameters work for **single decoding step** (most common case)

### Forward Pass Handling (`quant_utils.py:330-391`)

The `ActQuantWrapper.forward()` method processes activations:

```python
def forward(self, x):
    x_dtype = x.dtype
    
    if self.quantizer.static:
        # STATIC: Use pre-computed scale/zero_point
        x = self.quantizer(x)  # No per-token computation needed
    elif self.quantizer.bits < 16:
        # DYNAMIC: Compute scale/zero_point for this input
        self.quantizer.find_params(x)  # <-- Computes per-token scales
        x = self.quantizer(x).to(x_dtype)
        self.quantizer.free()  # Free memory after use
    
    x = self.module(x).to(x_dtype)  # Apply linear layer
    
    # Quantize output if needed
    if self.out_quantizer.bits < 16:
        self.out_quantizer.find_params(x)  # <-- Dynamic: compute again
        x = self.out_quantizer(x).to(x_dtype)
        self.out_quantizer.free()
    
    return x
```

### Key Differences in Practice

#### Prefill Stage Characteristics:
- **Long sequences**: Full prompt + image tokens (can be 1000+ tokens)
- **Parallel processing**: All tokens processed simultaneously
- **Activation patterns**: Different from decoding (larger activations, different distributions)

#### Decoding Stage Characteristics:
- **Short sequences**: Single new token + cached KV
- **Sequential processing**: One token at a time
- **Activation patterns**: Smaller activations, more consistent patterns

### Why Static Quantization Helps TTFT (Time-To-First-Token)

From the README:
> "MQuant proposes the **Modality-Specific Static Quantization (MSQ)** to significantly reduce the Time-to-First-Token (TTFT)"

**TTFT** = Time to generate the first token (prefill stage)

**Benefits of Static Quantization for Prefill:**
1. **No runtime overhead**: Scale/zero_point pre-computed, no `find_params()` calls
2. **Faster prefill**: Eliminates per-token scale computation during long sequences
3. **Consistent performance**: No variation from dynamic parameter computation

**Dynamic Quantization Overhead:**
- Each forward pass calls `find_params()` → computes min/max → computes scale/zero
- For prefill with 1000 tokens: 1000× overhead per layer
- For decoding with 1 token: 1× overhead (less significant)

### Example: Quantization Flow

```
┌─────────────────────────────────────────────────────────┐
│ CALIBRATION PHASE (Static Quantization)                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Sample 1: max_new_tokens=20                           │
│    ├─ Prefill: [prompt tokens] → observer.update()    │
│    └─ Decode: [token1] → observer.update()            │
│       Decode: [token2] → observer.update()            │
│       ... (20 steps)                                    │
│                                                         │
│  Sample 2-N: max_new_tokens=20 (similar)                │
│                                                         │
│  Last Sample: max_new_tokens=1                         │
│    ├─ Prefill: [prompt tokens] → observer.update()    │
│    └─ Decode: [token1] → observer.update()            │
│       → observer.get_quantization_params()             │
│       → quantizer.update_quantization_params()         │
│                                                         │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ INFERENCE PHASE                                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  STATIC MODE:                                          │
│    Prefill: [prompt] → quantizer(x) [uses fixed scale] │
│    Decode: [token] → quantizer(x) [uses fixed scale]   │
│                                                         │
│  DYNAMIC MODE:                                         │
│    Prefill: [prompt] → find_params() → quantize()     │
│    Decode: [token] → find_params() → quantize()       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Summary

### Dynamic vs Static

| Aspect | Dynamic | Static |
|--------|---------|--------|
| **Scale/Zero Computation** | Every forward pass | Pre-computed during calibration |
| **Runtime Overhead** | High (per-token computation) | Low (just lookup) |
| **Accuracy** | Adapts to input | Fixed, may be suboptimal |
| **TTFT Impact** | Slower (overhead in prefill) | Faster (no overhead) |
| **Memory** | Temporary scales | Persistent scales |

### Prefill vs Decoding

| Aspect | Prefill | Decoding |
|--------|---------|----------|
| **Sequence Length** | Long (1000+ tokens) | Short (1 token) |
| **Processing** | Parallel | Sequential |
| **Calibration** | Included in samples | Included in samples |
| **Static Quantization Benefit** | Large (eliminates 1000× overhead) | Small (eliminates 1× overhead) |

### MQuant's Innovation

MQuant uses **static quantization** specifically to optimize the **prefill stage**, which is critical for reducing TTFT. The calibration process ensures quantization parameters work well for both prefill (long sequences) and decoding (short sequences) patterns.

