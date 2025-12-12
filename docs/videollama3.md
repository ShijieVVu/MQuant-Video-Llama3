# VideoLLaMA3 Quantization Guide

This document describes the quantization implementation for VideoLLaMA3 multimodal models.

## Overview

VideoLLaMA3 is a video-language model based on Qwen2 architecture with the following components:

1. **Vision Encoder**: Custom vision transformer for processing video/image inputs
2. **Vision Projector (mm_projector)**: MLP or linear layer mapping vision features to LLM space
3. **LLM Backbone**: Qwen2-based language model for text generation

## Architecture Comparison: VideoLLaMA3 vs Qwen2VL

### Similarities
- Both use Qwen2 as the LLM backbone
- Both have vision encoder + projector + LLM structure
- Both support GPTQ and RTN quantization methods
- Both use pre-norm architecture (compatible with QuaRot)

### Key Differences

| Component | Qwen2VL | VideoLLaMA3 |
|-----------|---------|-------------|
| Vision Encoder | Qwen2-VL ViT | Custom VideoLLaMA3 encoder |
| Patch Embedding | `visual.patch_embed.proj` | `vision_encoder.embeddings.patch_embedding` |
| Encoder Layers | `visual.blocks[i]` | `vision_encoder.encoder.layers[i]` |
| Cross-Attention | Has `visual.merger` | No cross-attention module |
| Projector | N/A | `mm_projector` (MLP or Linear) |
| Spatial Processing | Standard downsampling | **Bilinear interpolation** |
| LLM Access | `model.model.language_model.layers` | `model.model.model.layers` |

## Quantization Components

### 1. Vision Encoder Quantization

The vision encoder processes video/image inputs through:

#### A. Patch Embedding
- **RTN**: `videollama3_visual_encoder_rtn()`
- **GPTQ**: `gptq_videollama3_fwrd_visual_encoder_patch_embedding()`
- Location: `model.vision_encoder.embeddings.patch_embedding`

#### B. Transformer Layers
- **GPTQ**: `gptq_videollama3_fwrd_visual_encoder_layers()`
- Location: `model.vision_encoder.encoder.layers[i]`
- Sequential quantization order:
  1. `self_attn.qkv` - Attention QKV projection
  2. `self_attn.proj` - Attention output projection
  3. `mlp.fc1` - MLP first layer
  4. `mlp.fc2` - MLP second layer

**Special Feature**: The encoder uses bilinear interpolation for spatial downsampling, which is unique to VideoLLaMA3 and important for video processing.

### 2. Vision Projector Quantization

Maps vision features to LLM embedding space:
- **RTN**: `videollama3_visual_projector_rtn()`
- **GPTQ**: `gptq_videollama3_fwrd_visual_projector()`
- Location: `model.mm_projector`
- Can be either:
  - Single `nn.Linear` layer
  - MLP with multiple layers (`MlpGeluProjector`)

### 3. LLM Quantization

Quantizes the Qwen2-based language model:
- **RTN**: `videollama3_llm_rtn()`
- **GPTQ**: `gptq_videollama3_fwrd_llm()`
- Location: `model.model.model.layers[i]`
- Sequential quantization order:
  1. `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj` - Attention projections
  2. `self_attn.o_proj` - Attention output
  3. `mlp.up_proj`, `mlp.gate_proj` - MLP gates
  4. `mlp.down_proj` - MLP down projection

## Usage

### Configuration Arguments

```python
args.quant_visual_clip = True      # Enable vision encoder quantization
args.quant_visual_projector = True # Enable projector quantization
args.quant_llm = True              # Enable LLM quantization

args.visual_w_rtn = False          # Use GPTQ for vision (False) or RTN (True)
args.llm_w_rtn = False             # Use GPTQ for LLM (False) or RTN (True)

args.visual_w_bits = 4             # Vision weight bit-width
args.llm_w_bits = 4                # LLM weight bit-width
args.w_groupsize = 128             # GPTQ group size
args.visual_w_clip = True          # Use MSE clipping for vision
args.llm_w_clip = True             # Use MSE clipping for LLM
```

### Main Entry Point

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

### Quantization Flow

The quantization process follows this order:

1. **Vision Encoder** (if `args.quant_visual_clip=True`)
   - Patch embedding quantization
   - Transformer layer quantization (layer by layer)

2. **Vision Projector** (if `args.quant_visual_projector=True`)
   - Projector layer quantization

3. **LLM** (if `args.quant_llm=True`)
   - LLM layer quantization (layer by layer)

## Implementation Details

### Catcher Mechanism

For GPTQ quantization, we use a "Catcher" wrapper to capture layer inputs:

```python
class Catcher(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inp, **kwargs):
        # Capture inputs for calibration
        inps[cache["i"]] = inp
        cache["i"] += 1
        raise ValueError  # Stop forward pass
```

This allows us to:
1. Run the model on calibration data
2. Capture intermediate activations
3. Use activations for GPTQ weight quantization

### Monkey Patching

For capturing inputs to middle layers, we use monkey patching:

```python
def monkey_patched_forward(self, *args, **kwargs):
    inps[cache["i"]] = args[0]
    grid_sizes[cache["i"]] = kwargs.get("grid_sizes", None)
    cache["i"] += 1
    raise ValueError

layers[0].forward = monkey_patched_forward.__get__(layers[0], layers[0].__class__)
```

### Sequential Processing

Each layer is quantized sequentially to minimize memory usage:
1. Quantize subset of weights (e.g., attention QKV)
2. Run forward pass to get next layer inputs
3. Move to next subset
4. Repeat for all layers

## Differences from Qwen2VL Implementation

### 1. No Cross-Attention Module
VideoLLaMA3 doesn't have the `visual.merger` cross-attention module present in Qwen2VL. Instead, it uses a simpler `mm_projector`.

### 2. Different Vision Encoder Structure
- Qwen2VL: `model.visual.blocks[i].attn.qkv.module`
- VideoLLaMA3: `model.vision_encoder.encoder.layers[i].self_attn.qkv`

### 3. Different LLM Access Path
- Qwen2VL: `model.model.language_model.layers[i]`
- VideoLLaMA3: `model.model.model.layers[i]`

### 4. Grid Sizes and Merge Sizes
VideoLLaMA3 encoder layers may use `grid_sizes` and `merge_sizes` for spatial processing, which need to be captured and passed through.

## Calibration Data Requirements

The quantization requires multimodal calibration data containing:
- **Videos/Images**: For vision encoder calibration
- **Text prompts**: For LLM calibration
- Typical dataset size: 128-1024 samples

The dataset should provide:
```python
dataset.data.iloc[i]  # Access to individual samples
dataset.build_prompt(sample)  # Prompt building function
```

## Tips and Best Practices

1. **Memory Management**: GPTQ is memory-intensive. Use `torch.cuda.empty_cache()` between layers.

2. **Bit-width Selection**: 
   - W4A16 (4-bit weights, 16-bit activations) is standard
   - Vision encoder can often use lower bits (3-4 bits)
   - LLM typically needs 4+ bits for good quality

3. **Group Size**: 
   - Smaller group size (e.g., 64) → better accuracy, more overhead
   - Larger group size (e.g., 256) → faster, slightly lower accuracy
   - 128 is a good default

4. **RTN vs GPTQ**:
   - RTN: Faster, simpler, slightly lower quality
   - GPTQ: Slower, requires calibration data, better quality
   - For quick testing: use RTN
   - For deployment: use GPTQ

5. **Skip Layers**: Use `args.skip_names` to skip quantizing specific layers (e.g., layer norms).

## Future Extensions

Potential improvements to the quantization:
1. **Activation Quantization**: Currently only weights are quantized
2. **Mixed Precision**: Different bit-widths for different layers
3. **KV Cache Quantization**: Quantize key-value cache for inference
4. **Batch Quantization**: Process multiple samples simultaneously

## References

- VideoLLaMA3 Repository: [DAMO-NLP-SG/VideoLLaMA3](https://huggingface.co/DAMO-NLP-SG/VideoLLaMA3-7B)
- GPTQ Paper: [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
- Qwen2 Documentation: [Qwen2 Model](https://huggingface.co/Qwen)

