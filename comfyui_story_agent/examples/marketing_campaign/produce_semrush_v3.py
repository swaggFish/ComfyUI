#!/usr/bin/env python3
"""
Producer Agent — Independent Scene Generation & Assembly

Generates each scene in its own ComfyUI session to prevent cumulative
VRAM leaks, then assembles the final commercial with voiceover.

This solves the persistent GPU freeze issue by ensuring each shot
starts with a clean VRAM state.
"""

import json
import subprocess
import sys
import time
import shutil
from pathlib import Path

COMFYUI_ROOT = Path("/home/mikeyb/Documents/AI/ComfyUI")
VENV_PYTHON = COMFYUI_ROOT / "venv" / "bin" / "python"
ART = Path("/home/mikeyb/.gemini/antigravity/brain/38e67136-715f-4218-afb0-2345a89a1faa/artifacts")

# ── Scene definitions ────────────────────────────────────────────
SCENES = [
    {
        "id": 1,
        "prompt": (
            "Cinematic slow pan across a sleek modern office desk. A glowing laptop "
            "screen displays a professional analytics dashboard with dramatically "
            "rising traffic charts in green and blue. Warm professional office lighting "
            "with soft bokeh background. A cup of coffee sits beside the laptop. The "
            "charts animate upward showing exponential growth. Clean, minimal, premium "
            "tech aesthetic."
        ),
        "first_frame": str(ART / "semrush_v3_anchors/shot1_first.png"),
        "last_frame": str(ART / "semrush_v3_anchors/shot1_last.png"),
        "camera": "pan right",
        "voiceover": str(ART / "semrush_v3_vo1.mp3"),
    },
    {
        "id": 2,
        "prompt": (
            "Extreme close-up of hands typing rapidly on a backlit mechanical keyboard. "
            "The keys glow with a subtle blue-white light. A large curved ultrawide "
            "monitor in the background displays a colorful keyword research dashboard "
            "with search volume graphs and ranking data. Shallow depth of field focuses "
            "on the fingers while the screen data is softly visible. Professional office "
            "environment, moody ambient lighting with blue and teal accents."
        ),
        "first_frame": str(ART / "semrush_v3_anchors/shot2_first.png"),
        "last_frame": str(ART / "semrush_v3_anchors/shot2_last.png"),
        "camera": "zoom in",
        "voiceover": str(ART / "semrush_v3_vo2.mp3"),
    },
    {
        "id": 3,
        "prompt": (
            "Medium shot of a confident, well-dressed young professional man in a navy "
            "blazer sitting in a modern bright co-working space. He faces the camera "
            "directly with a warm genuine smile. Natural daylight streams through large "
            "windows behind him. Clean white walls, indoor plants, and modern furniture "
            "in the background. He gestures towards the camera invitingly with one hand. "
            "Professional headshot quality lighting, warm skin tones, sharp focus on face."
        ),
        "first_frame": str(ART / "semrush_v3_anchors/shot3_first.png"),
        "last_frame": str(ART / "semrush_v3_anchors/shot3_last.png"),
        "camera": "static",
        "voiceover": str(ART / "semrush_v3_vo3.mp3"),
    },
]

STYLE_PROMPT = (
    "photorealistic, 8k resolution, cinematic lighting, highly detailed, "
    "professional cinematography, shallow depth of field, color graded, "
    "commercial advertisement quality"
)
NEGATIVE_PROMPT = (
    "cartoon, animated, illustration, low quality, deformed, blurry, "
    "text overlay, watermark, logo, split screen, multiple scenes, slow motion"
)


def wait_for_comfyui(timeout=60):
    """Wait for ComfyUI to be ready."""
    import urllib.request
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen("http://127.0.0.1:8188/system_stats", timeout=3)
            return True
        except Exception:
            time.sleep(2)
    return False


def kill_comfyui():
    """Kill any running ComfyUI processes and clear GPU memory."""
    patterns = [
        str(COMFYUI_ROOT / "launch_ltx.sh"),
        str(COMFYUI_ROOT / "main.py"),
        "main.py.*--bf16-vae",
    ]
    for pattern in patterns:
        subprocess.run(["pkill", "-f", pattern], capture_output=True)
    time.sleep(3)
    # Force-clear GPU memory
    subprocess.run([
        str(VENV_PYTHON), "-c",
        "import torch; torch.cuda.empty_cache(); torch.cuda.synchronize(); print('✓ CUDA cache cleared')"
    ], capture_output=True)
    time.sleep(2)


def start_comfyui():
    """Start ComfyUI in the background."""
    proc = subprocess.Popen(
        [str(COMFYUI_ROOT / "launch_ltx.sh")],
        cwd=str(COMFYUI_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print("  Starting ComfyUI...", end=" ", flush=True)
    if wait_for_comfyui(timeout=90):
        print("✓ Ready")
        return proc
    else:
        print("✗ Timeout")
        proc.kill()
        return None


def generate_single_scene(scene: dict, output_dir: Path) -> Path | None:
    """
    Generate a single scene as a standalone YAML storyboard,
    run it through the story agent, and return the clip path.
    """
    scene_dir = output_dir / f"scene_{scene['id']:02d}"
    scene_dir.mkdir(parents=True, exist_ok=True)

    # Write a single-shot storyboard YAML
    storyboard = {
        "title": f"Semrush Scene {scene['id']}",
        "style": "photorealistic",
        "fps": 24,
        "style_prompt": STYLE_PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "output_dir": "./output/story",
        "shots": [{
            "shot_id": 1,
            "prompt": scene["prompt"],
            "duration_frames": 65,  # 2.7s at 24fps — safe for 16GB VRAM
            "camera": scene["camera"],
            "transition_in": "cut",
            "transition_duration": 0.5,
            "first_frame_path": scene["first_frame"],
            "last_frame_path": scene["last_frame"],
            "seed": -1,
            "width": 768,
            "height": 512,
            "style_override": None,
            "skill": None,
            "expected_text": None,
        }],
    }

    yaml_path = scene_dir / "storyboard.yaml"
    import yaml
    with open(yaml_path, "w") as f:
        yaml.dump(storyboard, f, default_flow_style=False, allow_unicode=True)

    clips_dir = scene_dir / "clips"

    # Run the story agent for this single shot
    cmd = [
        str(VENV_PYTHON), "-m", "comfyui_story_agent", "generate",
        "-s", str(yaml_path),
        "--output", str(scene_dir),
    ]
    print(f"  Generating scene {scene['id']}...", flush=True)
    result = subprocess.run(
        cmd, cwd=str(COMFYUI_ROOT),
        capture_output=True, text=True, timeout=7200
    )

    if result.returncode != 0:
        print("  ✗ Generation failed:")
        print(result.stdout[-1000:])
        print(result.stderr[-1000:])
        return None

    # Find the output clip (latest webp)
    webps = sorted(clips_dir.glob("shot_001_*.webp"))
    if not webps:
        print(f"  ✗ No clips found in {clips_dir}")
        return None

    clip = webps[-1]  # latest attempt

    # Convert webp frames to mp4 at native 24fps
    import re
    frames = sorted(clips_dir.glob("shot_001_frames_*.png"))
    if not frames:
        print(f"  ✗ No frame PNGs found")
        return None

    nums = [int(re.search(r'_frames_(\d+)_', f.name).group(1)) for f in frames]
    start_num = min(nums)
    count = len(nums)
    pattern = str(clips_dir / "shot_001_frames_%05d_.png")
    mp4_out = scene_dir / f"scene_{scene['id']:02d}.mp4"

    r = subprocess.run([
        "ffmpeg", "-y",
        "-framerate", "24",
        "-start_number", str(start_num),
        "-i", pattern,
        "-frames:v", str(count),
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        str(mp4_out)
    ], capture_output=True)

    if r.returncode != 0:
        print(f"  ✗ ffmpeg failed: {r.stderr.decode()[-200:]}")
        return None

    dur = float(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(mp4_out)
    ]).decode().strip())
    print(f"  ✓ Scene {scene['id']}: {mp4_out.name} ({dur:.1f}s, {count} frames)")
    return mp4_out


def add_voiceover(clip: Path, vo: str, output: Path) -> Path | None:
    """Mux voiceover with clip, trimming audio to clip duration."""
    dur = float(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(clip)
    ]).decode().strip())

    r = subprocess.run([
        "ffmpeg", "-y",
        "-i", str(clip),
        "-i", vo,
        "-filter_complex",
        f"[1:a]apad=pad_dur={dur},atrim=0:{dur},aresample=48000[a]",
        "-map", "0:v", "-map", "[a]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
        "-t", str(dur),
        str(output)
    ], capture_output=True)

    return output if r.returncode == 0 else None


def stitch_final(clips: list[Path], output: Path) -> bool:
    """Concatenate all voiced clips into the final commercial."""
    n = len(clips)
    inputs = []
    for c in clips:
        inputs += ["-i", str(c)]

    fp = []
    for i in range(n):
        fp.append(f"[{i}:v]scale=768:512,setsar=1,fps=24[v{i}]")
        fp.append(f"[{i}:a]aresample=48000[a{i}]")

    concat = "".join(f"[v{i}][a{i}]" for i in range(n))
    filt = "; ".join(fp) + f"; {concat}concat=n={n}:v=1:a=1[v][a]"

    r = subprocess.run([
        "ffmpeg", "-y", *inputs,
        "-filter_complex", filt,
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        str(output)
    ], capture_output=True)

    return r.returncode == 0


def main():
    output_dir = COMFYUI_ROOT / "comfyui_story_agent" / "examples" / "marketing_campaign" / "semrush_output_v3"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("╔═══════════════════════════════════════════════════╗")
    print("║  Producer Agent — Semrush Commercial v3          ║")
    print("║  Independent Scenes + Fresh VRAM per shot        ║")
    print("╚═══════════════════════════════════════════════════╝")
    print()

    scene_clips = []

    for scene in SCENES:
        print(f"\n{'═'*50}")
        print(f"  SCENE {scene['id']}")
        print(f"{'═'*50}")

        # Kill any existing ComfyUI → fresh VRAM
        print("  Clearing GPU memory...")
        kill_comfyui()

        # Start fresh ComfyUI
        proc = start_comfyui()
        if proc is None:
            print("  ✗ ComfyUI failed to start, skipping scene")
            continue

        # Generate the scene
        clip = generate_single_scene(scene, output_dir)

        # Kill ComfyUI to free VRAM before next scene
        kill_comfyui()

        if clip and clip.exists():
            scene_clips.append({"clip": clip, "scene": scene})
        else:
            print(f"  ✗ Scene {scene['id']} failed")

    if not scene_clips:
        print("\n✗ No scenes generated!")
        sys.exit(1)

    # ── Add voiceover to each scene ─────────────────────────────
    print(f"\n{'═'*50}")
    print("  VOICEOVER & ASSEMBLY")
    print(f"{'═'*50}")

    voiced_clips = []
    for item in scene_clips:
        clip = item["clip"]
        scene = item["scene"]
        voiced = output_dir / f"voiced_scene_{scene['id']:02d}.mp4"
        print(f"  Adding VO to scene {scene['id']}...", end=" ", flush=True)
        result = add_voiceover(clip, scene["voiceover"], voiced)
        if result:
            print("✓")
            voiced_clips.append(voiced)
        else:
            print("✗ (using original without VO)")
            voiced_clips.append(clip)

    # ── Stitch final commercial ─────────────────────────────────
    final = ART / "semrush_commercial_v3.mp4"
    print(f"\n  Stitching {len(voiced_clips)} scenes...", end=" ", flush=True)
    if stitch_final(voiced_clips, final):
        dur = float(subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(final)
        ]).decode().strip())
        size = final.stat().st_size // (1024 * 1024)
        print(f"✓\n")
        print(f"╔═══════════════════════════════════════════════════╗")
        print(f"║  ✓ FINAL: semrush_commercial_v3.mp4              ║")
        print(f"║    Duration: {dur:.1f}s  Size: {size}MB               ║")
        print(f"╚═══════════════════════════════════════════════════╝")

        # Copy to easy location
        dest = COMFYUI_ROOT / "semrush_commercial_v3.mp4"
        shutil.copy2(final, dest)
        print(f"  Copied → {dest}")
    else:
        print("✗ Stitch failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
