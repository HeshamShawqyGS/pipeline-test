import sys, os, threading, time
import os.path as op

import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from controlnet_aux import AnylineDetector
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download
import gc

def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()

_pipeline = None
_loading_complete = False
_loading_thread = None

# model control
def load_pipeline():
    # Model configuration
    base_model_id = "SG161222/RealVisXL_V4.0"  
    controlnet_model_id = "xinsir/controlnet-canny-sdxl-1.0"
    
    # Load models
    controlnet = ControlNetModel.from_pretrained(controlnet_model_id, torch_dtype=torch.float16)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(base_model_id, controlnet=controlnet, 
                                                              torch_dtype=torch.float16)
    
    
    # Configure hardware
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        pipe.enable_model_cpu_offload()
    
    return pipe

# starting a daemon thread 
def initialize_models():
    global _loading_thread
    if _loading_thread is None or not _loading_thread.is_alive():
        _loading_thread = threading.Thread(target=_load_models)
        _loading_thread.daemon = True
        _loading_thread.start()

# handeling loading the model pipeline
def _load_models():

    global _pipeline, _loading_complete, _loading_error

    _loading_complete, _loading_error = False, None

    _pipeline = load_pipeline()

    _loading_complete = True

# calling the pipeline
def get_pipeline():
    return _pipeline

# check if pipeline is loaded
def is_loading_complete():
    return _loading_complete

# resizing the image 
def resize_image_small(image, max_size=1024):
    def make_divisible_by_8(value):
        return (value // 8) * 8

    width, height = image.size
    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int((max_size / width) * height)
        else:
            new_height = max_size
            new_width = int((max_size / height) * width)
        new_width = make_divisible_by_8(new_width)
        new_height = make_divisible_by_8(new_height)
        image = image.resize((new_width, new_height), Image.LANCZOS)
    
    width, height = image.size
    width = make_divisible_by_8(width)
    height = make_divisible_by_8(height)
    
    return image.resize((width, height), Image.LANCZOS)

# generate a canny edge image 
def preprocess_image(image):
    img_processor = AnylineDetector.from_pretrained("TheMistoAI/MistoLine", filename="MTEED.pth", subfolder="Anyline")
    edges_rgb = img_processor(image)
    if isinstance(edges_rgb, np.ndarray):
        edges_rgb = Image.fromarray(edges_rgb)
    return edges_rgb

# image generation
def generate_from_rhino_view(image, prompt, pipeline=None, negative_prompt="ugly, low quality", 
                             guidance_scale=5, control_strength=0.5, num_inference_steps=8, seed=None):
    
    pipe = pipeline if pipeline is not None else get_pipeline()
    
    small_image = resize_image_small(image)
    processed_image = preprocess_image(small_image)
    
    generator = torch.Generator("cuda").manual_seed(seed) if seed is not None else None
    
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=processed_image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=control_strength,
        generator=generator
    ).images[0]

    flush()
    return result