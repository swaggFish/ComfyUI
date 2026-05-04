"""
ComfyUI Video Story Agent — LTX 2.3 FLF2V Sequence Generator

Orchestrates ComfyUI to generate sequences of continuity-linked video clips
using the LTX 2.3 22B model with First/Last Frame to Video (FLF2V) conditioning,
then assembles them into cohesive stories with transitions.

Hardware Target: RTX 5060 Ti (16GB VRAM, Blackwell)
Model: ltx-2.3-22b-dev-fp8.safetensors
"""

__version__ = "0.1.0"
__author__ = "mikeyb"

# Resolution must be divisible by 32
RESOLUTION_STEP = 32

# Frame count must follow 8n + 1
FRAME_STEP = 8

# Default generation parameters (VRAM-safe for 16GB)
DEFAULT_WIDTH = 768
DEFAULT_HEIGHT = 512
DEFAULT_FRAMES = 33       # 8*4 + 1 = 33 frames
DEFAULT_FPS = 25
DEFAULT_CFG = 1.0         # LTX 2.3 distilled uses low CFG
DEFAULT_STEPS = 8         # Distilled model = fewer steps
DEFAULT_SEED = -1         # Random

# ComfyUI connection
COMFYUI_HOST = "127.0.0.1"
COMFYUI_PORT = 8188

# Model filenames (as installed)
MODEL_DIFFUSION = "ltx-2.3-22b-dev-fp8.safetensors"
MODEL_VIDEO_VAE = "LTX23_video_vae_bf16.safetensors"
MODEL_TINY_VAE = "taeltx2_3.safetensors"
MODEL_TEXT_ENCODER = "gemma_3_12B_it_fp4_mixed.safetensors"
MODEL_TEXT_PROJECTION = "ltx-2.3_text_projection_bf16.safetensors"
