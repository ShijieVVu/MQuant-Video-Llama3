# VideoEOC Dataset for MQuant

This implementation integrates the VideoEOC (Event-Oriented Comprehension) benchmark into the MQuant evaluation framework.

## Dataset Structure

VideoEOC expects the following directory structure:

```
<LMU_ROOT>/VideoEOC/
├── videos/
│   ├── 126.mp4
│   ├── 127.mp4
│   └── ...
└── meta_infos.json  # or meta_infos_subset.json
```

## Dataset Features

VideoEOC evaluates video understanding across three temporal dimensions:
- **Past**: Object State Retrospection, Location Retrospection, Object Relationship Evolution, Absolute Time Perception
- **Present**: Immediate State Recognition, Object Relationship, Purpose and Function Inference, Anomaly Perception
- **Future**: Trajectory and Motion Prediction, State Change Prediction, Dynamic Relationship Prediction

### Question Types
- **Single-choice**: Standard multiple-choice questions
- **Multi-choice**: Questions with multiple correct answers
- **Open-ended**: Time perception questions requiring numerical answers

### Special Features
- **Object Grounding**: Questions may reference objects with `<object N>` tags, which correspond to bounding boxes overlaid on video frames
- **Temporal Understanding**: Questions test understanding across past, present, and future temporal contexts

## Usage

### Basic Usage

```python
from vlmeval.dataset import build_dataset

# Build VideoEOC dataset
dataset = build_dataset("VideoEOC")

# The dataset will look for:
# - <LMU_ROOT>/VideoEOC/meta_infos.json or
# - <LMU_ROOT>/VideoEOC/meta_infos_subset.json
```

### Using with MQuant Evaluation

```python
from vlmeval.dataset import VideoEOC
from vlmeval.config import supported_VLM

# Load model
model = supported_VLM["Qwen2-VL-7B-Instruct"](
    model_path="Qwen/Qwen2-VL-7B-Instruct"
)

# Load dataset
dataset = VideoEOC(dataset="VideoEOC")

# Build prompt for a sample
num_frames = 8
video_llm = True  # Set to True if model supports native video input
message = dataset.build_prompt(line=0, num_frames=num_frames, video_llm=video_llm)

# Generate prediction
response = model.generate(message)
```

### Running Quantization Evaluation

```bash
python exam/quant_videollama3.py \
    --model_name VideoLLaMA3-7B \
    --dataset_name VideoEOC \
    --quant \
    --visual_w_bits 4 \
    --llm_w_bits 4 \
    --visual_a_bits 8 \
    --llm_a_bits 8 \
    --rotate \
    --rotate_visual_clip \
    --rotate_llm
```

## Data Format

### Input Format (meta_infos.json)

```json
[
  {
    "idx": 126,
    "video_path": "126.mp4",
    "question": "Has the status of <object 0> changed?",
    "choices": {
      "a": "No, it was always warm color",
      "b": "No, it was always cool color",
      "c": "Yes, it went from warm color to cool color",
      "d": "Yes, it went from cool color to warm color"
    },
    "answer": ["D"],
    "choice_type": "single-choice",
    "video_source": "charades",
    "video_type": "Object State Retrospection",
    "frame_number": 1043,
    "video_time": 43.41,
    "fps": 24.0,
    "box": [[53, 199, 273, 370]]
  }
]
```

### Output Format (predictions)

The model should output predictions in the format:
- **Single/Multi-choice**: `<choice>A</choice>` or `<choice>A, B</choice>`
- **Time perception**: Direct numerical value in seconds (e.g., "5.2")

## Evaluation Metrics

The evaluation returns metrics organized by:
- **Temporal categories**: Past, Present, Future, Overall
- **Question types**: Single-choice accuracy, Multi-choice accuracy
- **Fine-grained**: Per video-type accuracies

Example output:
```python
{
    "total": {
        "total_acc": 0.75,
        "single_acc": 0.80,
        "multi_acc": 0.65,
        "total_cnt": 100,
        ...
    },
    "Past": {...},
    "Present": {...},
    "Future": {...},
    "Object State Retrospection": {...},
    ...
}
```

## Implementation Details

### Key Components

1. **VideoEOC Class** (`videoeoc.py`):
   - Inherits from `VideoBaseDataset`
   - Handles JSON to TSV conversion
   - Implements prompt construction with object grounding support
   - Frame extraction from videos

2. **Evaluation Utils** (`utils/videoeoc_eval.py`):
   - Answer extraction from model outputs
   - Multi-dimensional metric calculation
   - Time perception scoring with error thresholds

### Prompt Construction

The prompt format varies based on:
- Presence of object tags (`<object N>`)
- Video type (Absolute Time Perception vs others)
- Number of correct answers (single vs multi-choice)

Example prompts:

**With object grounding:**
```
Question: I have overlaid the box on the last frame of the video, <object 0>:red. 
Has the status of <object 0> changed?
Options:
A. No, it was always warm color
B. No, it was always cool color
C. Yes, it went from warm color to cool color
D. Yes, it went from cool color to warm color
Answer directly using the letters of the options given and wrap your response.
```

**Time perception:**
```
Question: When does <object 0> first appear? Please output the answer directly in seconds.
```

## Notes

- The dataset automatically extracts and caches video frames to `<LMU_ROOT>/images/VideoEOC/`
- Frames are sampled uniformly across the video duration
- For models that support native video input, set `video_llm=True` in `build_prompt()`
- The evaluation handles partial credit for multi-choice questions

