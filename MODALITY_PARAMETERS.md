# Modality-Specific vs Shared Quantization Parameters

## Overview

MQuant uses **Modality-Specific Static Quantization (MSQ)**, where vision and text groups can have different quantization configurations, but share some common parameters.

---

## Parameter Comparison Table

| Parameter | Vision Group | Text/LLM Group | Shared? | Notes |
|-----------|--------------|---------------|---------|-------|
| **ACTIVATION QUANTIZATION** |
| `bits` (activation bits) | `args.visual_a_bits` (default: 8) | `args.llm_a_bits` (default: 8) | ❌ **Different** | Can set independently (e.g., visual=8, llm=8) |
| `static` (static mode) | `args.visual_static` | `args.llm_static` | ❌ **Different** | Can enable/disable independently |
| `groupsize` | `args.a_groupsize` (default: -1) | `args.a_groupsize` (default: -1) | ✅ **Shared** | Same groupsize for both modalities |
| `sym` (symmetric) | `not (args.a_asym)` | `not (args.a_asym)` | ✅ **Shared** | Same symmetric/asymmetric mode |
| `clip_ratio` | `args.a_clip_ratio` (default: 1.0) | `args.a_clip_ratio` (default: 1.0) | ✅ **Shared** | Same clipping ratio |
| `act_per_tensor` | `args.act_per_tensor` | `args.act_per_tensor` | ✅ **Shared** | Same per-tensor/per-token setting |
| `observer_type` | `"minmax"` | `"minmax"` | ✅ **Shared** | Always minmax observer |
| `calibration_mode` | `"layer_wise"` | `"layer_wise"` | ✅ **Shared** | Always layer-wise calibration |
| **WEIGHT QUANTIZATION** |
| `w_bits` (weight bits) | `args.visual_w_bits` (default: 4) | `args.llm_w_bits` (default: 4) | ❌ **Different** | Can set independently (e.g., visual=4, llm=4) |
| `w_rtn` (Round-to-Nearest) | `args.visual_w_rtn` | `args.llm_w_rtn` | ❌ **Different** | Can use RtN independently |
| `w_clip` (weight clipping) | `args.visual_w_clip` | `args.llm_w_clip` | ❌ **Different** | Can enable clipping independently |
| `w_groupsize` | `args.w_groupsize` (default: -1) | `args.w_groupsize` (default: -1) | ✅ **Shared** | Same groupsize for weights |
| `w_asym` (weight asymmetric) | `args.w_asym` | `args.w_asym` | ✅ **Shared** | Same symmetric/asymmetric mode |
| `percdamp` (GPTQ damping) | `args.percdamp` (default: 0.01) | `args.percdamp` (default: 0.01) | ✅ **Shared** | Same damping for GPTQ |
| `act_order` (GPTQ act-order) | `args.act_order` | `args.act_order` | ✅ **Shared** | Same act-order setting |
| **COMPUTED PARAMETERS** |
| `scale` | Computed per-layer | Computed per-layer | ❌ **Not shared** | Each layer computes its own scale |
| `zero_point` | Computed per-layer | Computed per-layer | ❌ **Not shared** | Each layer computes its own zero_point |

---

## Code Evidence

### Different Parameters (Modality-Specific)

```python
# From quant_qwen2vl.py:172-185 (Vision)
layer_input_bits = args.visual_a_bits      # ← Vision-specific
static = args.visual_static                # ← Vision-specific

# From quant_qwen2vl.py:196-209 (LLM)
layer_input_bits = args.llm_a_bits         # ← LLM-specific
static = args.llm_static                  # ← LLM-specific
```

### Shared Parameters (Common to Both)

```python
# Both vision and LLM use the same values:
layer_groupsize = args.a_groupsize         # ← Shared
layer_a_sym = not (args.a_asym)           # ← Shared
layer_a_clip = args.a_clip_ratio          # ← Shared
act_per_tensor = args.act_per_tensor       # ← Shared
observer_type = "minmax"                   # ← Shared (hardcoded)
```

---

## Example Configuration

### Typical W4A8 Setting (from README):

```bash
--visual_w_bits 4      # Vision weights: 4 bits
--visual_a_bits 8      # Vision activations: 8 bits
--llm_w_bits 4         # LLM weights: 4 bits
--llm_a_bits 8         # LLM activations: 8 bits
--visual_static         # Vision: static quantization
--llm_static            # LLM: static quantization
--a_groupsize -1        # Shared: per-token quantization
--a_asym                # Shared: asymmetric quantization
--a_clip_ratio 1.0      # Shared: no clipping
```

### What This Means:

- **Vision layers**: All use 4-bit weights, 8-bit activations, static mode
- **LLM layers**: All use 4-bit weights, 8-bit activations, static mode
- **Both groups**: Share groupsize, symmetric mode, clip ratio, per-tensor setting
- **Each layer**: Computes its own scale/zero_point independently

---

## Summary

### Modality-Specific Parameters (Can Differ):
1. **Activation bits**: `visual_a_bits` vs `llm_a_bits`
2. **Activation static mode**: `visual_static` vs `llm_static`
3. **Weight bits**: `visual_w_bits` vs `llm_w_bits`
4. **Weight RtN**: `visual_w_rtn` vs `llm_w_rtn`
5. **Weight clipping**: `visual_w_clip` vs `llm_w_clip`

### Shared Parameters (Same for Both):
1. **Activation groupsize**: `a_groupsize`
2. **Activation symmetric mode**: `a_asym`
3. **Activation clip ratio**: `a_clip_ratio`
4. **Activation per-tensor**: `act_per_tensor`
5. **Weight groupsize**: `w_groupsize`
6. **Weight symmetric mode**: `w_asym`
7. **GPTQ damping**: `percdamp`
8. **GPTQ act-order**: `act_order`
9. **Observer type**: Always `"minmax"`
10. **Calibration mode**: Always `"layer_wise"`

### Per-Layer Parameters (Computed Independently):
1. **Scale**: Each layer computes from its own min/max
2. **Zero point**: Each layer computes from its own min/max

---

## Key Insight

**"Modality-Specific"** means:
- Vision and text groups can have **different quantization configurations** (bits, static mode)
- But they **share common quantization settings** (groupsize, symmetric mode, clip ratio)
- Within each modality, all layers use the **same configuration parameters**
- But each layer still computes its own **scale/zero_point** from activation statistics

This allows fine-tuning quantization for each modality while maintaining consistency within each group.

