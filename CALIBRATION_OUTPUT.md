# Calibration Output: What Does `calib_qwen2vl_plus()` Produce?

## Overview

The calibration step (`calib_qwen2vl_plus`) **modifies the model in-place** by computing and storing quantization parameters (scale and zero_point) for each `ActQuantizer` instance.

---

## Step-by-Step Process

### 1. **Enable Calibration Mode** (Line 1109)
```python
model_open_calibrate(model.model, args)
```
**What happens**: Sets `quantizer.calibrate = True` for all `ActQuantWrapper` layers

**Result**: All quantizers enter "calibration mode" where they collect statistics but don't quantize

### 2. **Run Forward Passes on Calibration Data** (Lines 1112-1123)
```python
for i in tqdm(range(0, lt, step)):
    # ... prepare data ...
    model.generate(message=struct, dataset=args.dataset_name)
```

**What happens during each forward pass**:
- Model processes calibration samples (images + text prompts)
- For each `ActQuantizer.forward()` call:
  ```python
  if self.calibrate:
      self.quantizer.observer.update(x)  # Collect min/max statistics
      return x  # Return unquantized values
  ```

**Observer accumulates**:
- `observer.max_val`: Maximum activation value seen across all calibration samples
- `observer.min_val`: Minimum activation value seen across all calibration samples

### 3. **Finalize Parameters** (Lines 1113-1115)
```python
if i + step >= lt:
    print("last calibrate")
    model_open_last_calibrate(model.model, args)  # Sets last_calibrate = True
```

**What happens on last sample**:
```python
if self.last_calibrate:
    self.quantizer.update_quantization_params(x)
```

**This calls**:
```python
# From UniformQuantizer.update_quantization_params()
self.scale, self.zero_point = self.observer.get_quantization_params()
```

**Which computes**:
```python
# From MinmaxObserver.get_quantization_params()
scale = (max_val - min_val) / float(qmax - qmin)
zero_point = qmin - torch.round(min_val / scale)
```

### 4. **Disable Calibration, Enable Quantization** (Lines 1127-1129)
```python
model_close_calibrate(model.model, args)  # Sets calibrate = False
print("Calibrate End...")
model_quant(model.model, args)           # Sets quant = True
```

---

## Output: What Gets Stored

### For Each `ActQuantizer` Instance:

#### 1. **Pre-computed Scale Values**
- **Location**: `quantizer.scale` (stored in `UniformQuantizer`)
- **Type**: `torch.Tensor` (FP32)
- **Shape**: Depends on quantization mode:
  - **Per-tensor**: `[1]` (single scale for entire tensor)
  - **Per-token**: `[batch_size, seq_len, 1]` (one scale per token)
  - **Layer-wise**: `[1]` (single scale for entire layer)

**Example**:
```python
# For INT8 asymmetric quantization
scale = (max_val - min_val) / 255.0
# If max_val=2.5, min_val=-1.0:
# scale = (2.5 - (-1.0)) / 255 = 3.5 / 255 ≈ 0.0137
```

#### 2. **Pre-computed Zero Point Values**
- **Location**: `quantizer.zero_point` (stored in `UniformQuantizer`)
- **Type**: `torch.Tensor` (INT64)
- **Shape**: Same as scale

**Example**:
```python
# For INT8 asymmetric quantization
zero_point = 0 - round(-1.0 / 0.0137) = round(73.0) = 73
```

#### 3. **Observer Statistics** (in observer, not directly used after calibration)
- **Location**: `quantizer.observer.max_val`, `quantizer.observer.min_val`
- **Purpose**: Used to compute scale/zero_point, then can be discarded

#### 4. **State Flags**
- **Location**: `quantizer.calibrate = False`, `quantizer.quant = True`
- **Purpose**: Control quantization behavior during inference

---

## Visual Representation

### Before Calibration:
```python
ActQuantWrapper
└── quantizer (ActQuantizer)
    ├── quantizer (UniformQuantizer)
    │   ├── scale = None
    │   ├── zero_point = None
    │   └── observer (MinmaxObserver)
    │       ├── max_val = None
    │       └── min_val = None
    ├── calibrate = False
    └── quant = False
```

### After Calibration:
```python
ActQuantWrapper
└── quantizer (ActQuantizer)
    ├── quantizer (UniformQuantizer)
    │   ├── scale = tensor([0.0137])      # ← PRE-COMPUTED
    │   ├── zero_point = tensor([73])    # ← PRE-COMPUTED
    │   └── observer (MinmaxObserver)
    │       ├── max_val = tensor([2.5])  # Used to compute scale/zero
    │       └── min_val = tensor([-1.0])  # Used to compute scale/zero
    ├── calibrate = False                 # ← Calibration disabled
    └── quant = True                      # ← Quantization enabled
```

---

## What Happens During Inference After Calibration

After calibration, when the model runs inference:

```python
# ActQuantizer.forward() during inference
def forward(self, x):
    if self.static:
        if self.quant:  # ← This is True after calibration
            return self.quantizer(x)  # Uses pre-computed scale/zero_point
```

**The quantizer uses stored values**:
```python
# UniformQuantizer.quant()
def quant(self, inputs, scale=None, zero_point=None):
    if scale is None:
        scale = self.scale  # ← Uses pre-computed scale
    if zero_point is None:
        zero_point = self.zero_point  # ← Uses pre-computed zero_point
    
    outputs = inputs / scale + zero_point  # Quantize
    outputs = outputs.round().clamp(minq, maxq)
    return outputs
```

---

## Summary

### Output of `calib_qwen2vl_plus()`:

1. **Modified Model** (in-place):
   - All `ActQuantizer` instances now have pre-computed `scale` and `zero_point` values
   - These values are stored as `torch.Tensor` buffers in the model

2. **State Changes**:
   - `quantizer.calibrate = False` (calibration mode disabled)
   - `quantizer.quant = True` (quantization mode enabled)

3. **No Return Value**:
   - Function returns `None`
   - Model is modified in-place

4. **What Gets Computed**:
   - For each layer: `scale` and `zero_point` based on min/max statistics
   - Statistics collected from `calib_num` calibration samples
   - Parameters computed to cover the full range of activations seen during calibration

### Key Point:

**The calibration step doesn't return anything—it modifies the model in-place by storing quantization parameters that will be used during inference.** The model is now ready for static quantized inference, where these pre-computed scales and zero_points are used instead of computing them dynamically.

