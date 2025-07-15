import threading
import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from controlnet_aux import AnylineDetector
from PIL import Image
import numpy as np
import gc


_pipeline = None
_loading_complete = False
_loading_thread = None


def flush():
    """Dispose loaded models and flush caches"""
    global _pipeline, _loading_thread
    _pipeline = None
    _loading_thread = None
    gc.collect()

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()


def load_pipeline():
    """Load models, create image generation pipeline, and return the pipeline"""
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
        # pipe.enable_model_cpu_offload()
    
    return pipe


def initialize_models():
    """Start a daemon thread that loads pipeline"""
    def _load_models():
        global _pipeline, _loading_complete, _loading_error
        _loading_complete, _loading_error = False, None
        _pipeline = load_pipeline()
        _loading_complete = True

    global _loading_thread
    if _loading_thread is None or not _loading_thread.is_alive():
        _loading_thread = threading.Thread(target=_load_models)
        _loading_thread.daemon = True
        _loading_thread.start()


def get_pipeline():
    """Get loaded pipeline singleton"""
    return _pipeline


def is_loading_complete():
    """Check if pipeline is loaded"""
    return _loading_complete


def resize_image_small(image, max_size=1024):
    """Resize image magically"""
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


def preprocess_image(image):
    """Generate a canny edge image"""
    img_processor = AnylineDetector.from_pretrained("TheMistoAI/MistoLine", filename="MTEED.pth", subfolder="Anyline")
    edges_rgb = img_processor(image)
    if isinstance(edges_rgb, np.ndarray):
        edges_rgb = Image.fromarray(edges_rgb)
    return edges_rgb


def generate_from_rhino_view(image, prompt, pipeline=None, negative_prompt="ugly, low quality", 
                             guidance_scale=5, control_strength=0.5, num_inference_steps=8, seed=None):
    """Do magic!"""
    pipe = pipeline or get_pipeline() or load_pipeline()
    
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