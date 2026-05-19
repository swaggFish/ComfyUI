"""
RIFE Frame Interpolation Post-Processor

Takes a generated 8fps WebP/MP4 clip and uses ComfyUI's TensorRT RIFE node
to upsample it to 24fps, producing buttery smooth video without exceeding
the 97-frame VRAM limit during the base generation step.

Usage:
    python -m comfyui_story_agent.rife_interpolator \
        --input clip_8fps.mp4 \
        --output clip_24fps.mp4 \
        --multiplier 3
"""

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def build_rife_workflow(
    input_frames_dir: str,
    output_prefix: str,
    multiplier: int = 3,
    width: int = 1024,
    height: int = 576,
) -> dict:
    """
    Build a ComfyUI API workflow that:
    1. Loads all frames from a directory
    2. Runs AutoLoadRifeTensorrtModel
    3. Runs AutoRifeTensorrt (frame interpolation)
    4. Saves to VHS_VideoCombine (or frame sequence)
    """
    # Resolution profile: 'medium' covers 576-1312px (our 1024x576 fits)
    workflow = {
        "1": {
            "class_type": "VHS_LoadVideo",
            "inputs": {
                "video": input_frames_dir,
                "force_rate": 0,
                "force_size": "Disabled",
                "custom_width": width,
                "custom_height": height,
                "frame_load_cap": 0,
                "skip_first_frames": 0,
                "select_every_nth": 1,
            }
        },
        "2": {
            "class_type": "AutoLoadRifeTensorrtModel",
            "inputs": {
                "model": "rife49_ensemble_True_scale_1_sim",
                "precision": "fp16",
                "resolution_profile": "medium",
            }
        },
        "3": {
            "class_type": "AutoRifeTensorrt",
            "inputs": {
                "frames": ["1", 0],
                "rife_trt_model": ["2", 0],
                "clear_cache_after_n_frames": 50,
                "multiplier": multiplier,
                "keep_model_loaded": False,
            }
        },
        "4": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["3", 0],
                "frame_rate": 24,
                "loop_count": 0,
                "filename_prefix": output_prefix,
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 18,
                "save_metadata": False,
                "pingpong": False,
                "save_output": True,
            }
        }
    }
    return workflow


def interpolate_clip(
    input_path: str,
    output_path: str,
    multiplier: int = 3,
    comfyui_host: str = "127.0.0.1",
    comfyui_port: int = 8188,
) -> bool:
    """
    Interpolate a single clip using ComfyUI's RIFE TensorRT node.

    Strategy:
    - Extract frames from the input video at native fps
    - Upload to ComfyUI input dir
    - Queue a RIFE workflow with the given multiplier
    - Wait for completion and download the result

    Args:
        input_path: Path to the 8fps input video
        output_path: Path to write the 24fps output video
        multiplier: Frame multiplication factor (3x: 8fps→24fps)
        comfyui_host: ComfyUI server host
        comfyui_port: ComfyUI server port
    Returns:
        True on success, False on failure
    """
    import requests
    import websocket
    import uuid

    input_path = Path(input_path)
    output_path = Path(output_path)
    server = f"{comfyui_host}:{comfyui_port}"

    # ── Step 1: Extract input fps and probe clip ──────────────────────
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,width,height,nb_frames",
        "-of", "json", str(input_path)
    ]
    probe = json.loads(subprocess.check_output(probe_cmd).decode())
    stream = probe["streams"][0]
    width = stream["width"]
    height = stream["height"]
    raw_fps = stream["r_frame_rate"]  # e.g. "8/1"
    num, den = map(int, raw_fps.split("/"))
    input_fps = num / den
    logger.info(f"[RIFE] Input: {input_path.name}  {width}x{height}  {input_fps:.1f}fps")

    # ── Step 2: Copy input video to ComfyUI input dir ──────────────────
    comfyui_input_dir = Path(__file__).parent.parent / "input"
    comfyui_input_dir.mkdir(exist_ok=True)
    dest = comfyui_input_dir / input_path.name
    import shutil
    shutil.copy2(input_path, dest)
    logger.info(f"[RIFE] Copied input to ComfyUI input dir: {dest}")

    # ── Step 3: Build and queue the workflow ──────────────────────────
    output_prefix = f"rife_{input_path.stem}"
    workflow = build_rife_workflow(
        input_frames_dir=input_path.name,
        output_prefix=output_prefix,
        multiplier=multiplier,
        width=width,
        height=height,
    )

    client_id = str(uuid.uuid4())
    payload = {"prompt": workflow, "client_id": client_id}

    try:
        resp = requests.post(f"http://{server}/prompt", json=payload, timeout=30)
        resp.raise_for_status()
        prompt_id = resp.json()["prompt_id"]
        logger.info(f"[RIFE] Queued RIFE workflow: {prompt_id}")
    except Exception as e:
        logger.error(f"[RIFE] Failed to queue workflow: {e}")
        return False

    # ── Step 4: Wait for completion via WebSocket ─────────────────────
    ws_url = f"ws://{server}/ws?clientId={client_id}"
    timeout = 600
    start = time.time()
    output_file = None

    try:
        ws = websocket.create_connection(ws_url, timeout=10)
        while True:
            if time.time() - start > timeout:
                logger.error("[RIFE] WebSocket timeout waiting for RIFE completion")
                ws.close()
                return False
            try:
                msg = json.loads(ws.recv())
            except Exception:
                continue

            if msg.get("type") == "executing":
                data = msg.get("data", {})
                if data.get("node") is None and data.get("prompt_id") == prompt_id:
                    logger.info("[RIFE] Execution complete!")
                    break

        ws.close()
    except Exception as e:
        logger.error(f"[RIFE] WebSocket error: {e}")
        return False

    # ── Step 5: Retrieve the output file ─────────────────────────────
    try:
        history = requests.get(f"http://{server}/history/{prompt_id}", timeout=30).json()
        outputs = history.get(prompt_id, {}).get("outputs", {})
        for node_id, node_out in outputs.items():
            for video in node_out.get("videos", []):
                fname = video["filename"]
                subfolder = video.get("subfolder", "")
                params = {"filename": fname, "subfolder": subfolder, "type": "output"}
                dl = requests.get(f"http://{server}/view", params=params, timeout=120)
                dl.raise_for_status()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(dl.content)
                logger.info(f"[RIFE] Saved interpolated clip → {output_path}")
                output_file = str(output_path)
                break
            if output_file:
                break
    except Exception as e:
        logger.error(f"[RIFE] Failed to download output: {e}")
        return False

    return output_file is not None


def interpolate_sequence(
    clips_dir: str,
    output_dir: str,
    source_fps: int = 8,
    target_fps: int = 24,
    comfyui_host: str = "127.0.0.1",
    comfyui_port: int = 8188,
) -> list[str]:
    """
    Interpolate all clips in a directory from source_fps to target_fps.

    Returns list of output clip paths.
    """
    multiplier = target_fps // source_fps  # e.g. 24 // 8 = 3
    clips_dir = Path(clips_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all video clips (webp animated + mp4)
    clips = sorted(
        list(clips_dir.glob("*.mp4")) + list(clips_dir.glob("*.webp"))
    )
    clips = [c for c in clips if "frames" not in c.name]  # skip frame dumps

    results = []
    for clip in clips:
        out = output_dir / f"{clip.stem}_24fps.mp4"
        logger.info(f"[RIFE] Interpolating {clip.name} ({source_fps}fps → {target_fps}fps, {multiplier}x)")
        success = interpolate_clip(
            input_path=str(clip),
            output_path=str(out),
            multiplier=multiplier,
            comfyui_host=comfyui_host,
            comfyui_port=comfyui_port,
        )
        if success:
            results.append(str(out))
            logger.info(f"[RIFE] ✓ {out.name}")
        else:
            logger.error(f"[RIFE] ✗ Failed to interpolate {clip.name}")

    return results


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="RIFE frame interpolation via ComfyUI")
    parser.add_argument("--input", required=True, help="Input clip path (mp4 or webp)")
    parser.add_argument("--output", required=True, help="Output clip path")
    parser.add_argument("--multiplier", type=int, default=3, help="Frame multiplication factor (default: 3, i.e. 8fps→24fps)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8188)
    args = parser.parse_args()

    ok = interpolate_clip(
        input_path=args.input,
        output_path=args.output,
        multiplier=args.multiplier,
        comfyui_host=args.host,
        comfyui_port=args.port,
    )
    if ok:
        print(f"✓ Done → {args.output}")
    else:
        print("✗ Interpolation failed")
        exit(1)
