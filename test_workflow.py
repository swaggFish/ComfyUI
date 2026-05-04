import json
import urllib.request

def build_workflow():
    return {
        "1": {
            "inputs": {"ckpt_name": "ltx-2.3-22b-dev-fp8.safetensors"},
            "class_type": "CheckpointLoaderSimple"
        },
        "2": {
            "inputs": {
                "text_encoder": "gemma_3_12B_it_fp4_mixed.safetensors",
                "ckpt_name": "ltx-2.3-22b-dev-fp8.safetensors",
                "device": "default"
            },
            "class_type": "LTXAVTextEncoderLoader"
        },
        "3": {
            "inputs": {
                "lora_name": "ltx-2.3-22b-distilled-lora-384.safetensors",
                "strength_model": 0.5,
                "model": ["1", 0]
            },
            "class_type": "LoraLoaderModelOnly"
        },
        "4": {
            "inputs": {"text": "A beautiful cinematic shot of a mountain.", "clip": ["2", 0]},
            "class_type": "CLIPTextEncode"
        },
        "5": {
            "inputs": {"text": "ugly, blurry", "clip": ["2", 0]},
            "class_type": "CLIPTextEncode"
        },
        "6": {
            "inputs": {"frame_rate": 25.0, "positive": ["4", 0], "negative": ["5", 0]},
            "class_type": "LTXVConditioning"
        },
        "7": {
            "inputs": {"width": 384, "height": 256, "length": 9, "batch_size": 1},
            "class_type": "EmptyLTXVLatentVideo"
        },
        "8": {
            "inputs": {"cfg": 1.0, "model": ["3", 0], "positive": ["6", 0], "negative": ["6", 1]},
            "class_type": "CFGGuider"
        },
        "9": {
            "inputs": {"sampler_name": "euler_ancestral_cfg_pp"},
            "class_type": "KSamplerSelect"
        },
        "10": {
            "inputs": {"sigmas": "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"},
            "class_type": "ManualSigmas"
        },
        "11": {
            "inputs": {"noise_seed": 42},
            "class_type": "RandomNoise"
        },
        "12": {
            "inputs": {"noise": ["11", 0], "guider": ["8", 0], "sampler": ["9", 0], "sigmas": ["10", 0], "latent_image": ["7", 0]},
            "class_type": "SamplerCustomAdvanced"
        },
        "13": {
            "inputs": {"model_name": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"},
            "class_type": "LatentUpscaleModelLoader"
        },
        "14": {
            "inputs": {"samples": ["12", 0], "upscale_model": ["13", 0], "vae": ["1", 2]},
            "class_type": "LTXVLatentUpsampler"
        },
        "15": {
            "inputs": {"positive": ["6", 0], "negative": ["6", 1], "latent": ["14", 0]},
            "class_type": "LTXVCropGuides"
        },
        "16": {
            "inputs": {"cfg": 1.0, "model": ["3", 0], "positive": ["15", 0], "negative": ["15", 1]},
            "class_type": "CFGGuider"
        },
        "17": {
            "inputs": {"sampler_name": "euler_cfg_pp"},
            "class_type": "KSamplerSelect"
        },
        "18": {
            "inputs": {"sigmas": "0.85, 0.7250, 0.4219, 0.0"},
            "class_type": "ManualSigmas"
        },
        "19": {
            "inputs": {"noise_seed": 43},
            "class_type": "RandomNoise"
        },
        "20": {
            "inputs": {"noise": ["19", 0], "guider": ["16", 0], "sampler": ["17", 0], "sigmas": ["18", 0], "latent_image": ["15", 2]},
            "class_type": "SamplerCustomAdvanced"
        },
        "21": {
            "inputs": {"tile_size": 768, "overlap": 64, "temporal_size": 4096, "temporal_overlap": 4, "samples": ["20", 0], "vae": ["1", 2]},
            "class_type": "VAEDecodeTiled"
        },
        "22": {
            "inputs": {"images": ["21", 0], "filename_prefix": "test_ltx23_t2v", "fps": 25.0, "lossless": False, "quality": 85, "method": "default"},
            "class_type": "SaveAnimatedWEBP"
        }
    }

prompt = {"prompt": build_workflow()}
req = urllib.request.Request("http://127.0.0.1:8188/prompt", data=json.dumps(prompt).encode('utf-8'))
with urllib.request.urlopen(req) as response:
    print(response.read().decode())
