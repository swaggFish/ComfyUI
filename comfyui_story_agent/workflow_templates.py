"""
Workflow Templates — ComfyUI API Format JSON Builders

Advanced 2-pass upscale workflow with SamplerCustomAdvanced,
ManualSigmas, and Distilled LoRA for LTX 2.3 22B.
ManualSigmas, and Distilled LoRA for LTX 2.3 22B.
Matches the optimized user workflow for highest visual quality.
Includes Gemini Veo 3 hybrid mode for SOTA visual consistency.
"""

import copy
import random as _random
from . import (
    MODEL_DIFFUSION, DEFAULT_WIDTH, DEFAULT_HEIGHT,
    DEFAULT_FRAMES, DEFAULT_FPS, DEFAULT_CFG, DEFAULT_SEED,
)

# Hardcoded model names for the advanced LTX 2.3 pipeline
LORA_NAME = "ltx-2.3-22b-distilled-lora-384.safetensors"
UPSCALER_NAME = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
TEXT_ENCODER = "gemma_3_12B_it_fp4_mixed.safetensors"

def _seed(s):
    return s if s >= 0 else _random.randint(0, 2**32 - 1)

def build_text_to_video(prompt, negative="", width=DEFAULT_WIDTH,
                        height=DEFAULT_HEIGHT, length=DEFAULT_FRAMES,
                        fps=DEFAULT_FPS, seed=DEFAULT_SEED,
                        cfg=DEFAULT_CFG, steps=24,
                        filename_prefix="story_t2v"):
    """Text-to-Video: 2-pass advanced upscale generation."""
    
    w2 = width // 2
    h2 = height // 2
    
    # 2-pass sigmas for distilled model
    sigmas_1 = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
    sigmas_2 = "0.85, 0.7250, 0.4219, 0.0"
    
    w = {}
    
    # Models
    w["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": MODEL_DIFFUSION}}
    w["2"] = {"class_type": "LTXAVTextEncoderLoader", "inputs": {
        "text_encoder": TEXT_ENCODER, "ckpt_name": MODEL_DIFFUSION, "device": "default"}}
    w["3"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
        "lora_name": LORA_NAME, "strength_model": 0.5, "model": ["1", 0]}}
    
    # Text Prompts
    w["4"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}}
    w["5"] = {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["2", 0]}}
    w["6"] = {"class_type": "LTXVConditioning", "inputs": {
        "frame_rate": float(fps), "positive": ["4", 0], "negative": ["5", 0]}}
        
    # Empty Latent (Low Res)
    w["7"] = {"class_type": "EmptyLTXVLatentVideo", "inputs": {
        "width": w2, "height": h2, "length": length, "batch_size": 1}}
        
    # Pass 1 (Low Res)
    w["8"] = {"class_type": "CFGGuider", "inputs": {
        "cfg": float(cfg), "model": ["3", 0], "positive": ["6", 0], "negative": ["6", 1]}}
    w["9"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral_cfg_pp"}}
    w["10"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": sigmas_1}}
    s = _seed(seed)
    w["11"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": s}}
    w["12"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": ["11", 0], "guider": ["8", 0], "sampler": ["9", 0], 
        "sigmas": ["10", 0], "latent_image": ["7", 0]}}
        
    # Upscale Latent
    w["13"] = {"class_type": "LatentUpscaleModelLoader", "inputs": {"model_name": UPSCALER_NAME}}
    w["14"] = {"class_type": "LTXVLatentUpsampler", "inputs": {
        "samples": ["12", 0], "upscale_model": ["13", 0], "vae": ["1", 2]}}
        
    # Pass 2 (High Res)
    w["15"] = {"class_type": "LTXVCropGuides", "inputs": {
        "positive": ["6", 0], "negative": ["6", 1], "latent": ["14", 0]}}
    w["16"] = {"class_type": "CFGGuider", "inputs": {
        "cfg": float(cfg), "model": ["3", 0], "positive": ["15", 0], "negative": ["15", 1]}}
    w["17"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_cfg_pp"}}
    w["18"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": sigmas_2}}
    w["19"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": s}}
    w["20"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": ["19", 0], "guider": ["16", 0], "sampler": ["17", 0],
        "sigmas": ["18", 0], "latent_image": ["15", 2]}}
        
    # Decode and Save
    w["21"] = {"class_type": "VAEDecodeTiled", "inputs": {
        "tile_size": 768, "overlap": 64, "temporal_size": 4096, "temporal_overlap": 4,
        "samples": ["20", 0], "vae": ["1", 2]}}
    w["22"] = {"class_type": "SaveAnimatedWEBP", "inputs": {
        "images": ["21", 0], "filename_prefix": filename_prefix,
        "fps": fps, "lossless": True, "quality": 100, "method": "default"}}
    w["23"] = {"class_type": "SaveImage", "inputs": {
        "images": ["21", 0], "filename_prefix": f"{filename_prefix}_frames"}}
        
    return w

def build_image_to_video(prompt, image_name, negative="",
                         width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT,
                         length=DEFAULT_FRAMES, fps=DEFAULT_FPS,
                         seed=DEFAULT_SEED, cfg=DEFAULT_CFG,
                         steps=24, strength=0.7,
                         img_compression=18,
                         filename_prefix="story_i2v"):
    """Image-to-Video: injects first frame at both passes."""
    w = build_text_to_video(
        prompt, negative, width, height, length, fps, seed, cfg, steps, filename_prefix
    )
    
    # Load Image
    w["30"] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
    
    # Low-res image resize and prep
    w["31"] = {"class_type": "ResizeImageMaskNode", "inputs": {
        "input": ["30", 0], "resize_type": "scale dimensions",
        "resize_type.width": width//2, "resize_type.height": height//2,
        "resize_type.crop": "center", "scale_method": "lanczos"}}
    w["32"] = {"class_type": "LTXVPreprocess", "inputs": {
        "image": ["31", 0], "img_compression": img_compression}}
        
    # Pass 1 (Low Res) injection ONLY
    w["35"] = {"class_type": "LTXVAddGuide", "inputs": {
        "positive": ["6", 0], "negative": ["6", 1], "vae": ["1", 2],
        "latent": ["7", 0], "image": ["32", 0], "frame_idx": 0, "strength": 1.0}}
    w["8"]["inputs"]["positive"] = ["35", 0]
    w["8"]["inputs"]["negative"] = ["35", 1]
    w["12"]["inputs"]["latent_image"] = ["35", 2]
    
    # Pass 2 will upsample the generated animation naturally
    
    return w

def build_flf2v(prompt, first_image_name, last_image_name,
                negative="", width=DEFAULT_WIDTH,
                height=DEFAULT_HEIGHT, length=DEFAULT_FRAMES,
                fps=DEFAULT_FPS, seed=DEFAULT_SEED,
                cfg=DEFAULT_CFG, steps=24,
                first_strength=1.0, last_strength=1.0,
                img_compression=18,
                filename_prefix="story_flf2v"):
    """First/Last Frame to Video: injects guides at both passes."""
    w = build_text_to_video(
        prompt, negative, width, height, length, fps, seed, cfg, steps, filename_prefix
    )
    
    # First Frame Preprocessing
    w["30"] = {"class_type": "LoadImage", "inputs": {"image": first_image_name}}
    w["31"] = {"class_type": "ResizeImageMaskNode", "inputs": {
        "input": ["30", 0], "resize_type": "scale dimensions",
        "resize_type.width": width//2, "resize_type.height": height//2,
        "resize_type.crop": "center", "scale_method": "lanczos"}}
    w["32"] = {"class_type": "LTXVPreprocess", "inputs": {"image": ["31", 0], "img_compression": img_compression}}
    
    # Last Frame Preprocessing
    w["40"] = {"class_type": "LoadImage", "inputs": {"image": last_image_name}}
    w["41"] = {"class_type": "ResizeImageMaskNode", "inputs": {
        "input": ["40", 0], "resize_type": "scale dimensions",
        "resize_type.width": width//2, "resize_type.height": height//2,
        "resize_type.crop": "center", "scale_method": "lanczos"}}
    w["42"] = {"class_type": "LTXVPreprocess", "inputs": {"image": ["41", 0], "img_compression": img_compression}}
    
    # Pass 1 (Low Res) injection ONLY
    w["50"] = {"class_type": "LTXVAddGuide", "inputs": {
        "positive": ["6", 0], "negative": ["6", 1], "vae": ["1", 2],
        "latent": ["7", 0], "image": ["32", 0], "frame_idx": 0, "strength": float(first_strength)}}
    w["51"] = {"class_type": "LTXVAddGuide", "inputs": {
        "positive": ["50", 0], "negative": ["50", 1], "vae": ["1", 2],
        "latent": ["50", 2], "image": ["42", 0], "frame_idx": -1, "strength": float(last_strength)}}
        
    w["8"]["inputs"]["positive"] = ["51", 0]
    w["8"]["inputs"]["negative"] = ["51", 1]
    w["12"]["inputs"]["latent_image"] = ["51", 2]
    
    # Pass 2 will upsample the fully guided generated animation naturally
    
    return w

def build_hybrid_i2v(prompt, api_key, image_name=None, negative="", width=DEFAULT_WIDTH,
                      height=DEFAULT_HEIGHT, length=DEFAULT_FRAMES,
                      fps=DEFAULT_FPS, seed=DEFAULT_SEED,
                      cfg=DEFAULT_CFG, steps=None,
                      filename_prefix="hybrid_story"):
    """
    Hybrid Workflow:
    1. Gemini Veo 3 generates a cinematic 720p base animation.
       If image_name is provided, it uses it as a reference for continuity.
    2. LTX 2.3 uses the Gemini preview frame + local LoRA to refine/continue.
    """
    # Start with base T2V but override for color accuracy
    w = build_text_to_video(
        prompt, negative, width, height, length, fps, seed, cfg=3.0, steps=20, filename_prefix=filename_prefix
    )
    
    # 29: Load Initial Image for Gemini (if provided)
    if image_name:
        w["29"] = {
            "class_type": "LoadImage",
            "inputs": {
                "image": image_name
            }
        }

    # 30: Gemini Video Generator (External)
    w["30"] = {
        "class_type": "GeminiVideoGenerator",
        "inputs": {
            "prompt": prompt,
            "api_key": api_key,
            "model": "veo-3.0-generate-preview",
            "aspect_ratio": "16:9",
            "person_generation": "default",
            "max_wait_minutes": 10.0,
            "poll_interval_seconds": 15
        }
    }
    if image_name:
        w["30"]["inputs"]["initial_image"] = ["29", 0]
    
    # 31: LTX Preprocess for Gemini frame
    w["31"] = {
        "class_type": "LTXVPreprocess",
        "inputs": {
            "image": ["30", 1], # preview_frame from Gemini
            "img_compression": 18
        }
    }
    
    # 32: Inject Gemini Frame into LTX Pipeline (Pass 1)
    w["32"] = {
        "class_type": "LTXVAddGuide",
        "inputs": {
            "positive": ["6", 0],
            "negative": ["6", 1],
            "vae": ["1", 2],
            "latent": ["7", 0],
            "image": ["31", 0],
            "frame_idx": 0,
            "strength": 1.0
        }
    }
    
    # Re-route LTX Sampler to use the Gemini-guided latent
    w["8"]["inputs"]["positive"] = ["32", 0]
    w["8"]["inputs"]["negative"] = ["32", 1]
    w["12"]["inputs"]["latent_image"] = ["32", 2]
    return w

def build_hybrid_flf2v(prompt, api_key, negative="", width=DEFAULT_WIDTH,
                        height=DEFAULT_HEIGHT, length=DEFAULT_FRAMES,
                        fps=DEFAULT_FPS, seed=DEFAULT_SEED,
                        cfg=DEFAULT_CFG, steps=20,
                        filename_prefix="hybrid_flf2v"):
    """
    True Hybrid FLF2V:
    1. Gemini generates Start and End keyframes for the shot.
    2. LTX 2.3 interpolates between them using FLF2V conditioning.
    """
    # Start with base LTX T2V
    w = build_text_to_video(
        prompt, negative, width, height, length, fps, seed, cfg, steps, filename_prefix
    )
    
    # 30: Gemini Image Generation (Advanced) - Generate 2 images
    w["30"] = {
        "class_type": "GeminiImageGenADV",
        "inputs": {
            "inputcount": 2,
            "api_key": api_key,
            "model": "models/gemini-2.0-flash-preview-image-generation",
            "temperature": 1.0,
            "max_retries": 3,
            "prompt_1": f"{prompt} (Starting frame of the shot, consistent with previous context)",
            "prompt_2": f"{prompt} (Ending frame of the shot, showing movement progress)",
            "aspect_ratio": "16:9",
            "seed": seed if seed >= 0 else 0
        }
    }
    
    w["31"] = {
        "class_type": "SimpleImageBatchSelector",
        "inputs": {
            "images": ["30", 0],
            "index": 0
        }
    }
    w["32"] = {
        "class_type": "SimpleImageBatchSelector",
        "inputs": {
            "images": ["30", 0],
            "index": 1
        }
    }

    # 33: LTX Preprocess for Gemini frames
    w["33"] = {
        "class_type": "LTXVPreprocess",
        "inputs": {
            "image": ["31", 0],
            "img_compression": 18
        }
    }
    w["34"] = {
        "class_type": "LTXVPreprocess",
        "inputs": {
            "image": ["32", 0],
            "img_compression": 18
        }
    }
    
    # 35: Inject Gemini Frames into LTX Pipeline (Pass 1)
    w["35"] = {
        "class_type": "LTXVAddGuide",
        "inputs": {
            "positive": ["6", 0],
            "negative": ["6", 1],
            "vae": ["1", 2],
            "latent": ["7", 0],
            "image": ["33", 0],
            "frame_idx": 0,
            "strength": 1.0
        }
    }
    w["36"] = {
        "class_type": "LTXVAddGuide",
        "inputs": {
            "positive": ["35", 0],
            "negative": ["35", 1],
            "vae": ["1", 2],
            "latent": ["35", 2],
            "image": ["34", 0],
            "frame_idx": -1,
            "strength": 1.0
        }
    }
    
    # Re-route LTX Sampler
    w["8"]["inputs"]["positive"] = ["36", 0]
    w["8"]["inputs"]["negative"] = ["36", 1]
    w["12"]["inputs"]["latent_image"] = ["36", 2]
    
    return w

# ── Utility Mutators ──────────────────────────────────────────────────

def set_seed(workflow, seed):
    w = copy.deepcopy(workflow)
    s = _seed(seed)
    if "11" in w:
        w["11"]["inputs"]["noise_seed"] = s
    if "19" in w:
        w["19"]["inputs"]["noise_seed"] = s
    return w

def set_prompt(workflow, prompt, negative=None):
    w = copy.deepcopy(workflow)
    if "4" in w:
        w["4"]["inputs"]["text"] = prompt
    if negative is not None and "5" in w:
        w["5"]["inputs"]["text"] = negative
    return w

def set_resolution(workflow, width, height, length=None):
    w = copy.deepcopy(workflow)
    if "7" in w:
        w["7"]["inputs"]["width"] = width // 2
        w["7"]["inputs"]["height"] = height // 2
        if length is not None:
            w["7"]["inputs"]["length"] = length
            
    # Update resizer nodes if present
    for node_id in ["31", "41"]:
        if node_id in w:
            w[node_id]["inputs"]["resize_type.width"] = width // 2
            w[node_id]["inputs"]["resize_type.height"] = height // 2
            
    # Update Gemini nodes if present
    if "30" in w and w["30"].get("class_type") == "GeminiImageGenADV":
        # GeminiImageGenADV uses strings for aspect ratio
        w["30"]["inputs"]["aspect_ratio"] = "16:9" 
            
    return w
