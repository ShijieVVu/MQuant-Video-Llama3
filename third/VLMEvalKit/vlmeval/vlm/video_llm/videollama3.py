import torch
import warnings
from PIL import Image
import numpy as np
from ..base import BaseModel


class VideoLLaMA3(BaseModel):
    """VideoLLaMA3 model wrapper for VLMEval framework.
    
    VideoLLaMA3 is based on Qwen2 architecture with a custom vision encoder.
    """
    
    INSTALL_REQ = False
    INTERLEAVE = False
    VIDEO_LLM = True

    def __init__(self, model_path="DAMO-NLP-SG/VideoLLaMA3-7B", verbose=True, **kwargs):
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError:
            warnings.warn("Please install transformers: pip install transformers")
            raise

        assert model_path is not None
        self.model_path = model_path
        self.verbose = verbose
        self.kwargs = kwargs
        
        # Load model and processor
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        ).eval()
        
        torch.cuda.empty_cache()

    def generate_inner(self, message, dataset=None):
        """Generate response for a given message.
        
        Args:
            message: List of dicts with 'type' and 'value' keys
            dataset: Optional dataset name for dataset-specific handling
            
        Returns:
            str: Generated response text
        """
        # Convert message format to conversation format
        conversation = [{
            "role": "user",
            "content": []
        }]

        for item in message:
            if item['type'] == 'image':
                img = Image.open(item['value']).convert("RGB")
                conversation[0]["content"].append({
                    "type": "image",
                    "image": np.array(img)
                })
            elif item['type'] == 'text':
                conversation[0]["content"].append({
                    "type": "text",
                    "text": item['value']
                })

        inputs = self.processor(
            conversation=conversation,
            return_tensors="pt"
        )
        
        # Move inputs to correct device and dtype
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
        
        # Convert pixel_values to bfloat16 to match model weights
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        
        # Generate
        generation_config = {
            "max_new_tokens": self.kwargs.get("max_new_tokens", 1024),
            "temperature": self.kwargs.get("temperature", 0.2),
            "do_sample": self.kwargs.get("do_sample", True),
        }
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                **generation_config
            )
        
        # Decode output
        generated_text = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        if self.verbose:
            print(f"\033[32mGenerated: {generated_text}\033[0m")
        
        return generated_text

