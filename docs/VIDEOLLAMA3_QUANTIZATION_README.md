# VideoLLaMA3 Quantization - Quick Start Guide

## 📁 Files Created

This quantization implementation includes:

1. **`fake_quant/gptq/videollama3_gptq_plus.py`** (609 lines)
   - Main quantization implementation for VideoLLaMA3
   - Supports both RTN and GPTQ methods
   - Quantizes vision encoder, projector, and LLM

2. **`docs/videollama3.md`** (76 lines)
   - Detailed documentation for VideoLLaMA3 quantization
   - Architecture overview and usage guide
   - Configuration and best practices

3. **`docs/qwen2vl_vs_videollama3_comparison.md`** (84 lines)
   - Side-by-side comparison of Qwen2VL vs VideoLLaMA3
   - Migration guide between implementations
   - Component-by-component breakdown

## 🚀 Quick Start

### Basic Usage

```python
from fake_quant.gptq.videollama3_gptq_plus import videollama3_rtn_gptq_fwrd_plus

# Configure quantization
args.quant_visual_clip = True        # Enable vision encoder quantization
args.quant_visual_projector = True   # Enable projector quantization  
args.quant_llm = True                # Enable LLM quantization

args.visual_w_rtn = False            # Use GPTQ (False) or RTN (True)
args.llm_w_rtn = False               # Use GPTQ (False) or RTN (True)

args.visual_w_bits = 4               # Vision weight bits
args.llm_w_bits = 4                  # LLM weight bits
args.w_groupsize = 128               # GPTQ group size

# Run quantization
quantizers = videollama3_rtn_gptq_fwrd_plus(
    model=model,
    dataset=calibration_dataset,
    dev=device,
    dataset_name="coco",
    args=args
)
```

## 🏗️ Architecture Overview

VideoLLaMA3 has three main components that are quantized:

```
┌─────────────────────────────────────────────────┐
│  Input: Video/Image                             │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  1. VISION ENCODER                              │
│     - Patch Embedding (Linear)                  │
│     - Transformer Layers (32 layers)            │
│       • Self-Attention (qkv, proj)              │
│       • MLP (fc1, fc2)                          │
│     - Bilinear spatial downsampling             │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  2. VISION PROJECTOR (mm_projector)             │
│     - Linear or MLP projection                  │
│     - Maps vision features → LLM embedding      │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  3. LANGUAGE MODEL (Qwen2-based)                │
│     - Transformer Layers (28 layers)            │
│       • Self-Attention (q/k/v/o proj)           │
│       • MLP (up/gate/down proj)                 │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  Output: Text Generation                        │
└─────────────────────────────────────────────────┘
```

## 🔑 Key Differences from Qwen2VL

| Feature | Qwen2VL | VideoLLaMA3 |
|---------|---------|-------------|
| **Vision Components** | Conv1 + ResBlocks + Cross-Attention | Encoder + Projector |
| **Config Flag** | `quant_cross_attention` | `quant_visual_projector` |
| **Spatial Processing** | Standard | Bilinear interpolation |
| **Patch Embedding** | Convolution (`GPTQConv`) | Linear (`GPTQ`) |
| **LLM Path** | `model.model.language_model.layers` | `model.model.model.layers` |

See `docs/qwen2vl_vs_videollama3_comparison.md` for detailed comparison.

## 📊 Quantization Components

### 1. Vision Encoder Quantization

**Location**: `model.vision_encoder`

**Functions**:
- RTN: `videollama3_visual_encoder_rtn()`
- GPTQ: 
  - `gptq_videollama3_fwrd_visual_encoder_patch_embedding()`
  - `gptq_videollama3_fwrd_visual_encoder_layers()`

**Layers Quantized**:
```python
# Patch embedding
vision_encoder.embeddings.patch_embedding

# Each encoder layer (32 layers)
vision_encoder.encoder.layers[i].self_attn.qkv
vision_encoder.encoder.layers[i].self_attn.proj
vision_encoder.encoder.layers[i].mlp.fc1
vision_encoder.encoder.layers[i].mlp.fc2
```

### 2. Vision Projector Quantization

**Location**: `model.mm_projector`

**Functions**:
- RTN: `videollama3_visual_projector_rtn()`
- GPTQ: `gptq_videollama3_fwrd_visual_projector()`

**Types**:
- `nn.Linear`: Single projection layer
- `MlpGeluProjector`: Multi-layer MLP

### 3. LLM Quantization

**Location**: `model.model.model.layers`

**Functions**:
- RTN: `videollama3_llm_rtn()`
- GPTQ: `gptq_videollama3_fwrd_llm()`

**Layers Quantized**:
```python
# Each LLM layer (28 layers for 7B model)
model.layers[i].self_attn.q_proj
model.layers[i].self_attn.k_proj
model.layers[i].self_attn.v_proj
model.layers[i].self_attn.o_proj
model.layers[i].mlp.up_proj
model.layers[i].mlp.gate_proj
model.layers[i].mlp.down_proj
```

## ⚙️ Configuration Options

### Quantization Flags
```python
args.quant_visual_clip = True       # Quantize vision encoder
args.quant_visual_projector = True  # Quantize vision projector
args.quant_llm = True               # Quantize language model
```

### Method Selection
```python
args.visual_w_rtn = False           # False=GPTQ, True=RTN for vision
args.llm_w_rtn = False              # False=GPTQ, True=RTN for LLM
```

### Bit-width Configuration
```python
args.visual_w_bits = 4              # Vision encoder weight bits (2-8)
args.llm_w_bits = 4                 # LLM weight bits (2-8)
```

### GPTQ Parameters
```python
args.w_groupsize = 128              # Group size (64/128/256)
args.percdamp = 0.01                # Damping factor (0.0-1.0)
args.act_order = False              # Activation ordering
args.visual_w_clip = True           # MSE-based clipping for vision
args.llm_w_clip = True              # MSE-based clipping for LLM
```

### Advanced Options
```python
args.visual_split = False           # Split visual MLP layers
args.llm_split = False              # Split LLM down_proj
args.skip_names = []                # List of layer names to skip
args.nsamples = 128                 # Number of calibration samples
```

## 📈 Recommended Settings

### For Best Quality (Deployment)
```python
args.visual_w_rtn = False           # Use GPTQ
args.llm_w_rtn = False              # Use GPTQ
args.visual_w_bits = 4              # W4 for vision
args.llm_w_bits = 4                 # W4 for LLM
args.w_groupsize = 128              # Standard group size
args.nsamples = 512                 # More calibration samples
```

### For Fast Prototyping (Testing)
```python
args.visual_w_rtn = True            # Use RTN (faster)
args.llm_w_rtn = True               # Use RTN (faster)
args.visual_w_bits = 4              # W4 for vision
args.llm_w_bits = 4                 # W4 for LLM
args.nsamples = 128                 # Fewer samples
```

### For Maximum Compression
```python
args.visual_w_rtn = False           # Use GPTQ for better accuracy
args.llm_w_rtn = False              # Use GPTQ for better accuracy
args.visual_w_bits = 3              # W3 for vision (more aggressive)
args.llm_w_bits = 3                 # W3 for LLM (more aggressive)
args.w_groupsize = 64               # Smaller group size for better accuracy
args.nsamples = 1024                # More calibration for low-bit
```

## 💾 Memory Requirements

Approximate GPU memory usage during quantization:

| Model Size | RTN | GPTQ (128 samples) | GPTQ (512 samples) |
|------------|-----|--------------------|--------------------|
| 7B | ~16 GB | ~24 GB | ~32 GB |
| 13B | ~28 GB | ~40 GB | ~56 GB |

**Tips to Reduce Memory**:
1. Reduce `args.nsamples` (minimum: 32)
2. Process vision and LLM separately
3. Use gradient checkpointing if available
4. Clear cache between components

## 🎯 Calibration Data

The quantization requires multimodal calibration data:

### Dataset Requirements
- **Format**: Video/Image + Text pairs
- **Size**: 128-1024 samples recommended
- **Content**: Representative of target use case

### Example Dataset Structure
```python
dataset.data = pd.DataFrame({
    'video_path': [...],      # Path to video/image files
    'question': [...],         # Text prompts/questions
    'answer': [...]            # Expected responses (optional)
})

# Must provide
dataset.build_prompt(sample)  # Function to build model input
```

### Common Datasets
- **COCO Captions**: Image understanding
- **MSR-VTT**: Video captioning
- **ActivityNet**: Video QA
- **Custom**: Domain-specific data

## 🐛 Troubleshooting

### Issue: "AttributeError: no attribute 'visual'"
**Cause**: Using wrong quantization file  
**Solution**: Use `videollama3_gptq_plus.py` for VideoLLaMA3, not `qwen2vl_gptq_plus.py`

### Issue: "CUDA out of memory"
**Solutions**:
1. Reduce `args.nsamples` to 64 or 128
2. Use RTN instead of GPTQ for vision encoder
3. Quantize components separately
4. Use a GPU with more memory

### Issue: "Module not found in full"
**Cause**: Layer name not found in model  
**Solution**: Check layer names with `quant_utils.find_qlayers(layer)` or add to `args.skip_names`

### Issue: Poor quality after quantization
**Solutions**:
1. Increase calibration samples (`args.nsamples`)
2. Reduce bit-width for less critical components
3. Use smaller group size (`args.w_groupsize = 64`)
4. Enable MSE clipping (`args.visual_w_clip = True`)

## 📚 Documentation

- **Main Guide**: `docs/videollama3.md`
- **Comparison**: `docs/qwen2vl_vs_videollama3_comparison.md`
- **Implementation**: `fake_quant/gptq/videollama3_gptq_plus.py`

## 🔬 Testing

To verify quantization works correctly:

```python
import torch
from fake_quant.gptq.videollama3_gptq_plus import videollama3_rtn_gptq_fwrd_plus

# 1. Load model
model = load_videollama3_model("DAMO-NLP-SG/VideoLLaMA3-7B")

# 2. Prepare calibration data
dataset = load_calibration_dataset("coco", num_samples=128)

# 3. Configure quantization (start with RTN for quick test)
args.quant_visual_clip = True
args.quant_visual_projector = True
args.quant_llm = True
args.visual_w_rtn = True  # RTN for speed
args.llm_w_rtn = True
args.visual_w_bits = 4
args.llm_w_bits = 4

# 4. Run quantization
quantizers = videollama3_rtn_gptq_fwrd_plus(
    model, dataset, "cuda", "coco", args
)

# 5. Test inference
output = model.generate(test_input)
print(output)
```

## 🎓 Next Steps

1. **Read the detailed guide**: `docs/videollama3.md`
2. **Compare with Qwen2VL**: `docs/qwen2vl_vs_videollama3_comparison.md`
3. **Understand the code**: Review `fake_quant/gptq/videollama3_gptq_plus.py`
4. **Try different configurations**: Experiment with bit-widths and methods
5. **Evaluate on your task**: Test quantized model on target benchmarks

## 📖 References

- [VideoLLaMA3 Model](https://huggingface.co/DAMO-NLP-SG/VideoLLaMA3-7B)
- [GPTQ Paper](https://arxiv.org/abs/2210.17323)
- [Qwen2 Documentation](https://huggingface.co/Qwen)
- [MQuant Framework](../README.md)

## 🤝 Contributing

To extend or improve this quantization:

1. Review the implementation in `videollama3_gptq_plus.py`
2. Follow the same structure as Qwen2VL quantization
3. Test thoroughly with different configurations
4. Update documentation accordingly

## ✅ Summary

You now have a complete quantization implementation for VideoLLaMA3 that:

- ✅ Supports both vision and language modalities
- ✅ Provides RTN and GPTQ methods
- ✅ Handles VideoLLaMA3's unique architecture (bilinear sampling, projector)
- ✅ Includes comprehensive documentation and comparison with Qwen2VL
- ✅ Follows the same patterns as existing MQuant implementations

Happy quantizing! 🚀

