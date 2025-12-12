# Qwen2VL vs VideoLLaMA3 Quantization: Side-by-Side Comparison

This document provides a detailed comparison between the Qwen2VL and VideoLLaMA3 quantization implementations to help understand their architectural differences and how they affect quantization strategies.

## Quick Reference Table

| Aspect | Qwen2VL | VideoLLaMA3 |
|--------|---------|-------------|
| **Implementation File** | `qwen2vl_gptq_plus.py` | `videollama3_gptq_plus.py` |
| **Base LLM** | Qwen2 | Qwen2 |
| **Vision Encoder** | Qwen2-VL ViT | Custom VideoLLaMA3 encoder |
| **Total Components** | 4 (Conv1, ResBlocks, Cross-Attn, LLM) | 3 (Encoder, Projector, LLM) |
| **Cross-Attention** | ✅ Yes (`visual.merger`) | ❌ No |
| **Vision Projector** | ❌ No separate module | ✅ Yes (`mm_projector`) |
| **Spatial Processing** | Standard | Bilinear interpolation |
| **Configuration Flags** | `quant_visual_clip`, `quant_cross_attention`, `quant_llm` | `quant_visual_clip`, `quant_visual_projector`, `quant_llm` |

## Architecture Comparison

### Model Structure

#### Qwen2VL
```
Input (Image/Video)
    ↓
Visual Patch Embedding (Conv1) ← model.visual.patch_embed.proj.module
    ↓
Visual Transformer Blocks      ← model.visual.blocks[i]
    ├─ Attention (qkv, proj)
    └─ MLP (fc1, fc2)
    ↓
Visual Cross-Attention Merger  ← model.visual.merger
    └─ MLP layers (mlp.0, mlp.2)
    ↓
Language Model                 ← model.model.language_model.layers[i]
    ├─ Self-Attention
    └─ MLP
    ↓
Output (Text)
```

#### VideoLLaMA3
```
Input (Video/Image)
    ↓
Vision Encoder Patch Embedding ← model.vision_encoder.embeddings.patch_embedding
    ↓
Vision Encoder Layers          ← model.vision_encoder.encoder.layers[i]
    ├─ Self-Attention (qkv, proj)
    └─ MLP (fc1, fc2)
    ↓
Vision Projector (mm_projector) ← model.mm_projector
    └─ Linear or MLP
    ↓
Language Model                  ← model.model.model.layers[i]
    ├─ Self-Attention
    └─ MLP
    ↓
Output (Text)
```

## Component-by-Component Comparison

### 1. Vision Patch Embedding

#### Qwen2VL: Conv1 Layer
```python
# Location
model.visual.patch_embed.proj.module

# RTN Function
qwen2vl_visual_clip_rtn()

# GPTQ Function
gptq_qwen2vl_fwrd_visual_clip_conv1()

# Special: Uses GPTQConv for convolution layer
conv1_gptq = GPTQConv(model.model.visual.patch_embed.proj.module)
```

#### VideoLLaMA3: Patch Embedding
```python
# Location
model.vision_encoder.embeddings.patch_embedding

# RTN Function
videollama3_visual_encoder_rtn()

# GPTQ Function
gptq_videollama3_fwrd_visual_encoder_patch_embedding()

# Uses standard GPTQ for linear layer
patch_gptq = GPTQ(vision_encoder.embeddings.patch_embedding)
```

**Key Difference**: Qwen2VL uses convolution (`GPTQConv`), VideoLLaMA3 uses linear layer (`GPTQ`).

### 2. Vision Transformer Blocks

#### Qwen2VL: Visual Blocks
```python
# Location
layers = model.model.visual.blocks

# Sequential order
sequential = [
    ["attn.qkv.module"],
    ["attn.proj.module"],
    ["mlp.fc1.module"],
    ["mlp.fc2.L2" or "mlp.fc2.module"]  # Depends on visual_split
]

# Forward pass captures
- cu_seqlens
- rotary_pos_emb
- position_embeddings (optional)

# Function
gptq_qwen2vl_fwrd_visual_clip_resblocks()
```

#### VideoLLaMA3: Encoder Layers
```python
# Location
layers = model.vision_encoder.encoder.layers

# Sequential order
sequential = [
    ["self_attn.qkv"],
    ["self_attn.proj"],
    ["mlp.fc1"],
    ["mlp.fc2"]
]

# Forward pass captures
- grid_sizes
- merge_sizes

# Function
gptq_videollama3_fwrd_visual_encoder_layers()
```

**Key Differences**:
- Different module paths (`attn` vs `self_attn`)
- Different auxiliary inputs (rotary embeddings vs grid sizes)
- VideoLLaMA3 encoder supports bilinear spatial downsampling

### 3. Vision Processing Module

#### Qwen2VL: Cross-Attention Merger
```python
# Location
model.visual.merger

# Sequential order
sequential = [
    ["mlp.0.module"],
    ["mlp.2.module"]
]

# Functions
qwen2vl_visual_cross_attention_rtn()         # RTN
gptq_qwen2vl_fwrd_visual_clip_cross_attention()  # GPTQ

# Purpose: Multi-modal fusion with cross-attention
```

#### VideoLLaMA3: Vision Projector
```python
# Location
model.mm_projector

# Types
- nn.Linear (single layer)
- MlpGeluProjector (multi-layer MLP)

# Functions
videollama3_visual_projector_rtn()        # RTN
gptq_videollama3_fwrd_visual_projector()  # GPTQ

# Purpose: Simple projection to LLM embedding space
```

**Key Difference**: Qwen2VL has a complex cross-attention module for fusion; VideoLLaMA3 uses a simpler projector.

### 4. Language Model

#### Qwen2VL: Language Model
```python
# Location
layers = model.model.model.language_model.layers

# Sequential order
sequential = [
    ["self_attn.q_proj.module", "self_attn.k_proj.module", "self_attn.v_proj.module"],
    ["self_attn.o_proj.module"],
    ["mlp.up_proj.module", "mlp.gate_proj.module"],
    ["mlp.down_proj.L2" or "mlp.down_proj.module"]
]

# Function
gptq_qwen2vl_fwrd_llm()
```

#### VideoLLaMA3: Language Model
```python
# Location
layers = model.model.model.layers

# Sequential order
sequential = [
    ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    ["self_attn.o_proj"],
    ["mlp.up_proj", "mlp.gate_proj"],
    ["mlp.down_proj.L2" or "mlp.down_proj"]
]

# Function
gptq_videollama3_fwrd_llm()
```

**Key Difference**: Different paths (`model.language_model.layers` vs `model.model.layers`), but otherwise very similar structure.

## Quantization Configuration

### Qwen2VL Configuration
```python
# Flags
args.quant_visual_clip = True       # Quantize visual encoder
args.quant_cross_attention = True   # Quantize cross-attention
args.quant_llm = True               # Quantize LLM

# RTN switches
args.visual_w_rtn = False           # Use GPTQ for visual
args.llm_w_rtn = False              # Use GPTQ for LLM

# Bit-widths
args.visual_w_bits = 4              # Visual weights
args.llm_w_bits = 4                 # LLM weights

# GPTQ parameters
args.w_groupsize = 128              # Group size
args.percdamp = 0.01                # Damping factor
args.act_order = False              # Activation order
args.visual_w_clip = True           # MSE clipping for visual
args.llm_w_clip = True              # MSE clipping for LLM

# Splitting
args.visual_split = False           # Split visual mlp.fc2
args.llm_split = False              # Split llm down_proj
```

### VideoLLaMA3 Configuration
```python
# Flags
args.quant_visual_clip = True       # Quantize visual encoder
args.quant_visual_projector = True  # Quantize projector ← Different
args.quant_llm = True               # Quantize LLM

# RTN switches
args.visual_w_rtn = False           # Use GPTQ for visual
args.llm_w_rtn = False              # Use GPTQ for LLM

# Bit-widths
args.visual_w_bits = 4              # Visual weights
args.llm_w_bits = 4                 # LLM weights

# GPTQ parameters (same as Qwen2VL)
args.w_groupsize = 128
args.percdamp = 0.01
args.act_order = False
args.visual_w_clip = True
args.llm_w_clip = True

# Splitting
args.visual_split = False           # Not typically used
args.llm_split = False              # Split llm down_proj
```

**Key Difference**: VideoLLaMA3 uses `quant_visual_projector` instead of `quant_cross_attention`.

## Main Entry Points

### Qwen2VL
```python
from fake_quant.gptq.qwen2vl_gptq_plus import qwen2vl_rtn_gptq_fwrd_plus

quantizers = qwen2vl_rtn_gptq_fwrd_plus(
    model=model,
    dataset=dataset,
    dev=device,
    dataset_name=dataset_name,
    args=args
)
```

### VideoLLaMA3
```python
from fake_quant.gptq.videollama3_gptq_plus import videollama3_rtn_gptq_fwrd_plus

quantizers = videollama3_rtn_gptq_fwrd_plus(
    model=model,
    dataset=dataset,
    dev=device,
    dataset_name=dataset_name,
    args=args
)
```

## Quantization Flow Comparison

### Qwen2VL Flow
```
1. Visual Encoder (if quant_visual_clip)
   a. Conv1 layer (patch embedding)
   b. Visual transformer blocks
   
2. Cross-Attention (if quant_cross_attention)
   a. Merger MLP layers
   
3. Language Model (if quant_llm)
   a. Attention layers
   b. MLP layers
```

### VideoLLaMA3 Flow
```
1. Vision Encoder (if quant_visual_clip)
   a. Patch embedding
   b. Encoder transformer layers
   
2. Vision Projector (if quant_visual_projector)
   a. Linear or MLP projection
   
3. Language Model (if quant_llm)
   a. Attention layers
   b. MLP layers
```

## Code Structure Comparison

### Function Naming Patterns

#### Qwen2VL
- RTN: `qwen2vl_<component>_rtn()`
- GPTQ: `gptq_qwen2vl_fwrd_<component>()`
- Main: `qwen2vl_rtn_gptq_fwrd_plus()`

Components:
- `visual_clip` (includes both conv1 and resblocks)
- `visual_clip_conv1` (specific)
- `visual_clip_resblocks` (specific)
- `visual_clip_cross_attention`
- `llm`

#### VideoLLaMA3
- RTN: `videollama3_<component>_rtn()`
- GPTQ: `gptq_videollama3_fwrd_<component>()`
- Main: `videollama3_rtn_gptq_fwrd_plus()`

Components:
- `visual_encoder` (includes patch embedding)
- `visual_encoder_patch_embedding` (specific)
- `visual_encoder_layers` (specific)
- `visual_projector`
- `llm`

## Special Features

### Qwen2VL Special Features
1. **3D Rotary Position Embeddings**: Supports temporal modeling with rotary embeddings
2. **Cross-Attention Fusion**: Sophisticated multi-modal fusion mechanism
3. **Conv-based Patch Embedding**: Uses `GPTQConv` for convolution quantization
4. **cu_seqlens**: Efficient sequence length handling

### VideoLLaMA3 Special Features
1. **Bilinear Spatial Downsampling**: Unique spatial processing for videos
2. **Grid Sizes & Merge Sizes**: Flexible spatial resolution handling
3. **Simpler Projector**: Lightweight vision-to-language mapping
4. **Video Token Compression**: Optional token compression for efficiency

## When to Use Each

### Use Qwen2VL Quantization When:
- Working with Qwen2-VL models
- Need sophisticated vision-language fusion
- Working with image understanding tasks
- Model has explicit cross-attention modules

### Use VideoLLaMA3 Quantization When:
- Working with VideoLLaMA3 models
- Processing video inputs with temporal information
- Need bilinear spatial processing
- Model has simple vision projector

## Migration Guide

### From Qwen2VL to VideoLLaMA3

If you have code using Qwen2VL quantization and want to adapt it for VideoLLaMA3:

1. **Change import**:
   ```python
   # Old
   from fake_quant.gptq.qwen2vl_gptq_plus import qwen2vl_rtn_gptq_fwrd_plus
   
   # New
   from fake_quant.gptq.videollama3_gptq_plus import videollama3_rtn_gptq_fwrd_plus
   ```

2. **Update configuration flag**:
   ```python
   # Old
   args.quant_cross_attention = True
   
   # New
   args.quant_visual_projector = True
   ```

3. **Update function call**:
   ```python
   # Old
   quantizers = qwen2vl_rtn_gptq_fwrd_plus(model, dataset, dev, dataset_name, args)
   
   # New
   quantizers = videollama3_rtn_gptq_fwrd_plus(model, dataset, dev, dataset_name, args)
   ```

4. **Check quantizer keys**: The saved quantizer dictionary will have different key names reflecting the different module paths.

### From VideoLLaMA3 to Qwen2VL

Reverse the changes above and add:
```python
args.quant_visual_projector = False  # Not used in Qwen2VL
args.quant_cross_attention = True    # Specific to Qwen2VL
```

## Performance Considerations

### Memory Usage
- **Qwen2VL**: Slightly higher due to cross-attention module
- **VideoLLaMA3**: Slightly lower with simpler projector

### Quantization Speed
- Both implementations are similar in speed
- GPTQ requires calibration data (~2-3x slower than RTN)
- Layer-by-layer processing keeps memory usage manageable

### Accuracy
- Vision encoder: Both achieve similar accuracy with W4A16
- Fusion module: Cross-attention (Qwen2VL) may be more sensitive to quantization
- Projector (VideoLLaMA3): Simple structure quantizes well
- LLM: Identical quantization approach and accuracy

## Common Issues and Solutions

### Issue 1: Module Not Found
**Error**: `AttributeError: 'Model' object has no attribute 'visual'`

**Solution**: Check if you're using the correct quantization file for your model:
- Qwen2VL models → `qwen2vl_gptq_plus.py`
- VideoLLaMA3 models → `videollama3_gptq_plus.py`

### Issue 2: Wrong Quantizer Keys
**Error**: Keys in quantizer dict don't match model structure

**Solution**: The quantizer keys are model-specific:
```python
# Qwen2VL
"model.visual.blocks.0.attn.qkv.module"

# VideoLLaMA3
"model.vision_encoder.encoder.layers.0.self_attn.qkv"
```

### Issue 3: Missing Configuration Flag
**Error**: Projector/cross-attention not quantized

**Solution**: Use correct flags:
```python
# Qwen2VL
args.quant_cross_attention = True

# VideoLLaMA3
args.quant_visual_projector = True
```

## Summary

Both implementations follow similar GPTQ patterns but are adapted to their respective model architectures:

- **Qwen2VL**: More complex with cross-attention fusion, suited for sophisticated image-text understanding
- **VideoLLaMA3**: Simpler architecture with focus on video processing and bilinear spatial handling

Choose the implementation that matches your model architecture. The quantization quality and approach are similar, with the main differences being architectural adaptations.

