#!/usr/bin/env python3
"""
Producer Agent — Eight Sleep Scene Generation & Assembly (v13.1)
Generates each scene independently, applies homography UI overlays for crisp 
legible phone screens (using SIFT tracking with fallback mechanisms), and verifies
onscreen text legibility using a transcription verification step.
"""

import os
import sys
import time
import shutil
import subprocess
import yaml
from pathlib import Path

COMFYUI_ROOT = Path("/home/mikeyb/Documents/AI/ComfyUI")
sys.path.append(str(COMFYUI_ROOT))
VENV_PYTHON = COMFYUI_ROOT / "venv" / "bin" / "python"
ART = Path("/home/mikeyb/.gemini/antigravity/brain/ed14af04-07db-4537-881e-4d8d1aa3f71b/artifacts")

# Load Gemini API Key from env or .env file
def load_gemini_key():
    key = os.environ.get("GEMINI_API_KEY", "")
    if key:
        return key
    env_path = COMFYUI_ROOT / "pixelle-mcp" / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("GEMINI_API_KEY="):
                    return stripped.split("=", 1)[1].strip().strip('"').strip("'")
    return ""

GEMINI_API_KEY = load_gemini_key()
if GEMINI_API_KEY:
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

SCENES = [
    {
        "id": 1,
        "prompt": "A cinematic, slow-motion shot of a person falling onto a luxurious, softly lit bed at night, moody blue and warm amber lighting.",
        "first_frame": "/home/mikeyb/Documents/AI/ComfyUI/comfyui_story_agent/examples/marketing_campaign/eight_sleep_anchors/shot1_start.png",
        "last_frame": None,
        "camera": "pan right",
        "duration_frames": 81,
        "width": 1280,
        "height": 704,
    },
    {
        "id": 2,
        "prompt": "Extreme close-up of the Eight Sleep cooling grid technology and modern app interface glowing in the dark, shallow depth of field.",
        "first_frame": "/home/mikeyb/Documents/AI/ComfyUI/comfyui_story_agent/examples/marketing_campaign/eight_sleep_anchors/shot2_start.png",
        "last_frame": None,
        "camera": "zoom in",
        "duration_frames": 81,
        "width": 1280,
        "height": 704,
        "composite_ui": True,
        "expected_text": "Eight Sleep",
    },
    {
        "id": 3,
        "prompt": "A cinematic, high-detail macro shot of a modern mattress with a glowing blue digital grid pattern representing cooling technology, soft dark ambient lighting, premium, clean, high-tech bed cover, commercial quality.",
        "first_frame": "/home/mikeyb/Documents/AI/ComfyUI/comfyui_story_agent/examples/marketing_campaign/eight_sleep_anchors/shot6_start.png",
        "last_frame": None,
        "camera": "pan right",
        "duration_frames": 81,
        "width": 1280,
        "height": 704,
    },
    {
        "id": 4,
        "prompt": "A cinematic, photorealistic shot of a woman sleeping peacefully and deeply in a luxurious bed at night, moody blue and warm ambient lighting, comfortable, quiet night, shallow depth of field, high detail commercial quality.",
        "first_frame": "/home/mikeyb/Documents/AI/ComfyUI/comfyui_story_agent/examples/marketing_campaign/eight_sleep_anchors/shot5_start.png",
        "last_frame": None,
        "camera": "static",
        "duration_frames": 81,
        "width": 1280,
        "height": 704,
    },
    {
        "id": 5,
        "prompt": "The person waking up looking incredibly refreshed, sunlight softly filtering through the blinds, peaceful and mindful morning.",
        "first_frame": "/home/mikeyb/Documents/AI/ComfyUI/comfyui_story_agent/examples/marketing_campaign/eight_sleep_anchors/shot3_start.png",
        "last_frame": None,
        "camera": "static",
        "duration_frames": 81,
        "width": 1280,
        "height": 704,
    },
    {
        "id": 6,
        "prompt": "A confident, well-rested professional sitting on the edge of the luxurious bed, looking directly at the camera and speaking warmly to the viewer. Sunlight filtering through the room.",
        "first_frame": None, # Set dynamically from scene 5 output
        "last_frame": None,
        "camera": "static",
        "duration_frames": 65,
        "width": 1280,
        "height": 704,
    }
]

STYLE_PROMPT = "photorealistic, 8k resolution, cinematic lighting, highly detailed, professional cinematography, color graded, commercial advertisement quality"
NEGATIVE_PROMPT = "cartoon, animated, illustration, low quality, deformed, blurry, watermark, logo, text overlay, split screen, multiple scenes"

def wait_for_comfyui(timeout=120):
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
    patterns = [
        str(COMFYUI_ROOT / "launch_ltx.sh"),
        str(COMFYUI_ROOT / "main.py"),
        "main.py.*--bf16-vae",
    ]
    for pattern in patterns:
        subprocess.run(["pkill", "-f", pattern], capture_output=True)
    time.sleep(3)
    # Clear CUDA memory
    subprocess.run([
        str(VENV_PYTHON), "-c",
        "import torch; torch.cuda.empty_cache(); torch.cuda.synchronize(); print('CUDA cache cleared')"
    ], capture_output=True)
    time.sleep(2)

def start_comfyui():
    proc = subprocess.Popen(
        [str(COMFYUI_ROOT / "launch_ltx.sh")],
        cwd=str(COMFYUI_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print("  Starting ComfyUI...", end=" ", flush=True)
    if wait_for_comfyui():
        print("✓ Ready")
        return proc
    else:
        print("✗ Timeout starting ComfyUI")
        proc.kill()
        return None

def is_valid_homography(M, w, h):
    """Validate that the transformation matrix doesn't cause degenerate warps."""
    import cv2
    import numpy as np

    if M is None:
        return False
        
    # Calculate rotation angle from the affine component of M
    a = M[0, 0]
    c = M[1, 0]
    theta = np.arctan2(c, a)
    rot_deg = np.abs(np.degrees(theta))
    if rot_deg > 15.0:
        return False

    # Corners of the base screen image
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    try:
        warped = cv2.perspectiveTransform(corners, M).reshape(-1, 2)
    except Exception:
        return False

    # Check 1: Must be a convex quad
    if not cv2.isContourConvex(np.array(warped, dtype=np.float32)):
        return False

    # Check 2: Area must be within reasonable bounds (allowing zoom-in up to 8x original size)
    area = cv2.contourArea(np.array(warped, dtype=np.float32))
    if area < 0.1 * w * h or area > 8.0 * w * h:
        return False

    # Check 3: Coordinate bounds check (allow points to go off-screen during zoom-in)
    for pt in warped:
        if pt[0] < -2.0 * w or pt[0] > 3.0 * w or pt[1] < -2.0 * h or pt[1] > 3.0 * h:
            return False

    return True

def apply_hybrid_ui_overlay(scene_dir: Path):
    """Overlay clean UI on Shot 2 using robust affine mapping with temporal smoothing validation."""
    import cv2
    import numpy as np
    import glob

    base_image_path = "/home/mikeyb/Documents/AI/ComfyUI/comfyui_story_agent/examples/marketing_campaign/eight_sleep_anchors/shot2_start.png"
    clips_dir = scene_dir / "clips"
    frame_files = sorted(glob.glob(str(clips_dir / "shot_001_frames_*.png")))
    
    if not frame_files:
        print("    ⚠ No frame PNGs found for Shot 2 composite!")
        return False

    print(f"    Composite: Smoothing and overlaying clean UI on {len(frame_files)} frames...")

    base_img = cv2.imread(base_image_path)
    if base_img is None:
        print("    ✗ Base image shot2_start.png not found!")
        return False
        
    first_frame = cv2.imread(frame_files[0])
    th, tw = first_frame.shape[:2]
    base_img = cv2.resize(base_img, (tw, th))
    h, w = base_img.shape[:2]

    # Precise polygon corners representing the glowing phone screen area
    pts = np.array([[455, 140], [700, 165], [820, 620], [570, 655]], dtype=np.int32)

    # UI mask with tight blur to avoid background bleeding
    ui_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(ui_mask, [pts], 255)
    ui_mask_blurred = cv2.GaussianBlur(ui_mask, (15, 15), 0) / 255.0
    ui_mask_blurred = np.stack([ui_mask_blurred]*3, axis=2)

    # Restrict SIFT matching purely to the phone bezel and hands (ignores background bedsheets grid)
    feature_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(feature_mask, (380, 80), (880, 700), 255, -1)
    cv2.fillPoly(feature_mask, [pts], 0)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(base_img, mask=feature_mask)
    matcher = cv2.BFMatcher()

    last_M = None
    fallback_count = 0
    estimated_matrices = []

    # Pass 1: Estimate raw transformation matrices frame-by-frame
    for i, fpath in enumerate(frame_files):
        frame = cv2.imread(fpath)
        if frame is None:
            estimated_matrices.append(None)
            continue

        M = None
        kp2, des2 = sift.detectAndCompute(frame, mask=feature_mask)

        if des2 is not None and len(des2) > 0:
            matches = matcher.knnMatch(des1, des2, k=2)
            good = []
            for match_pts in matches:
                if len(match_pts) == 2:
                    m, n = match_pts
                    if m.distance < 0.75 * n.distance:
                        good.append(m)

            # Ensure unique keypoints in the target frame to prevent many-to-one collapse
            good = sorted(good, key=lambda x: x.distance)
            seen_dst = set()
            unique_good = []
            for m in good:
                if m.trainIdx not in seen_dst:
                    seen_dst.add(m.trainIdx)
                    unique_good.append(m)
            good = unique_good

            if len(good) > 5:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                A, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
                if A is not None:
                    temp_M = np.eye(3)
                    temp_M[:2, :] = A
                    if is_valid_homography(temp_M, w, h):
                        M = temp_M
                        last_M = M

        # Fallback to last valid transform if current estimation is degenerate
        if M is None and last_M is not None:
            M = last_M
            fallback_count += 1

        estimated_matrices.append(M)

    # Pass 2: Smooth matrices temporally using a moving window (size 7)
    smoothed_matrices = []
    window_size = 7
    half_w = window_size // 2

    for i in range(len(estimated_matrices)):
        valid_window_matrices = []
        for offset in range(-half_w, half_w + 1):
            idx = i + offset
            if 0 <= idx < len(estimated_matrices) and estimated_matrices[idx] is not None:
                valid_window_matrices.append(estimated_matrices[idx])

        if valid_window_matrices:
            smoothed_M = np.mean(valid_window_matrices, axis=0)
            smoothed_matrices.append(smoothed_M)
        else:
            smoothed_matrices.append(estimated_matrices[i])

    # Pass 3: Render and overwrite frame images with smoothed matrices
    applied_count = 0
    for i, fpath in enumerate(frame_files):
        frame = cv2.imread(fpath)
        if frame is None:
            continue

        M = smoothed_matrices[i]
        if M is not None:
            warped_base = cv2.warpPerspective(base_img, M, (w, h))
            warped_mask = cv2.warpPerspective(ui_mask_blurred, M, (w, h))
            frame = frame * (1 - warped_mask) + warped_base * warped_mask
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            applied_count += 1

        cv2.imwrite(fpath, frame)

    print(f"    ✓ Composite UI tracking overlay complete (applied on {applied_count}/{len(frame_files)} frames, fallback={fallback_count}).")
    return True


def generate_single_scene(scene: dict, output_dir: Path) -> Path | None:
    scene_dir = output_dir / f"scene_{scene['id']:02d}"
    scene_dir.mkdir(parents=True, exist_ok=True)

    mp4_out = scene_dir / f"scene_{scene['id']:02d}.mp4"
    if mp4_out.exists():
        print(f"  ✓ Found existing MP4 for Scene {scene['id']}. Running verification check...")
        if scene.get("expected_text"):
            try:
                from comfyui_story_agent.text_agent import TextDirector
                td = TextDirector(
                    api_key=os.environ.get("GEMINI_API_KEY"),
                    evidence_dir=str(scene_dir / "evidence_transcription")
                )
                evaluation = td.review_clip(
                    clip_path=str(mp4_out),
                    expected_text=scene["expected_text"],
                    shot_id=scene["id"]
                )
                if evaluation.passed:
                    print(f"  ✓ Existing clip PASSED transcription check. Reusing existing scene!")
                    return mp4_out
                else:
                    print(f"  ✗ Existing clip FAILED transcription check: {evaluation.summary}. Regenerating...")
            except Exception as e:
                print(f"  ⚠️ Error running transcription check on existing clip: {e}. Regenerating...")
        else:
            print(f"  ✓ Reusing existing scene!")
            return mp4_out

    storyboard = {
        "title": f"Eight Sleep Shot {scene['id']}",
        "style": "photorealistic",
        "fps": 24,
        "style_prompt": STYLE_PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "output_dir": "./output/story",
        "shots": [{
            "shot_id": 1,
            "prompt": scene["prompt"],
            "duration_frames": scene["duration_frames"],
            "camera": scene["camera"],
            "transition_in": "cut",
            "transition_duration": 0.5,
            "first_frame_path": scene["first_frame"],
            "last_frame_path": scene["last_frame"],
            "seed": -1,
            "width": scene["width"],
            "height": scene["height"],
            "style_override": None,
            "skill": None,
            "expected_text": scene.get("expected_text"),
        }],
    }

    yaml_path = scene_dir / "storyboard.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(storyboard, f, default_flow_style=False, allow_unicode=True)

    clips_dir = scene_dir / "clips"
    # Clean clips directory to prevent stale/duplicate frames from previous runs/attempts
    shutil.rmtree(str(clips_dir), ignore_errors=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    # Generate
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

    # Apply composite screen UI if flagged
    if scene.get("composite_ui"):
        apply_hybrid_ui_overlay(scene_dir)

    # Find the frame PNGs
    import re
    frames = sorted(clips_dir.glob("shot_001_frames_*.png"))
    if not frames:
        print(f"  ✗ No frame PNGs found in {clips_dir}")
        return None

    nums = [int(re.search(r'_frames_(\d+)_', f.name).group(1)) for f in frames]
    start_num = min(nums)
    count = len(nums)
    pattern = str(clips_dir / "shot_001_frames_%05d_.png")
    mp4_out = scene_dir / f"scene_{scene['id']:02d}.mp4"

    # Encode to MP4 at 24fps
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
        print(f"  ✗ ffmpeg encoding failed: {r.stderr.decode()[-200:]}")
        return None

    # Verify legibility of onscreen text via transcription verification process
    if scene.get("expected_text"):
        print(f"  📝 Verifying legibility of onscreen text (expected: '{scene['expected_text']}') using TextDirector...")
        from comfyui_story_agent.text_agent import TextDirector
        td = TextDirector(
            api_key=os.environ.get("GEMINI_API_KEY"),
            evidence_dir=str(scene_dir / "evidence_transcription")
        )
        evaluation = td.review_clip(
            clip_path=str(mp4_out),
            expected_text=scene["expected_text"],
            shot_id=scene["id"]
        )
        if not evaluation.passed:
            print(f"  ✗ Text transcription check FAILED for Scene {scene['id']}: {evaluation.summary}")
            return None
        print(f"  ✓ Text transcription check PASSED: Onscreen text is clear and legible English.")

    dur = float(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(mp4_out)
    ]).decode().strip())
    print(f"  ✓ Scene {scene['id']}: {mp4_out.name} ({dur:.2f}s, {count} frames)")
    return mp4_out

def main():
    output_dir = COMFYUI_ROOT / "comfyui_story_agent" / "examples" / "marketing_campaign" / "eight_sleep_output_v13"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("╔═══════════════════════════════════════════════════╗")
    print("║  Producer Agent — Eight Sleep Commercial v13.1   ║")
    print("║  SIFT Poly Overlay + Transcription Verification    ║")
    print("╚═══════════════════════════════════════════════════╝")
    print()

    scene_clips = {}

    for scene in SCENES:
        print(f"\n{'═'*50}")
        print(f"  SCENE {scene['id']}")
        print(f"{'═'*50}")

        passed = False
        for attempt in range(1, 4):
            print(f"\n  --- Attempt {attempt} for Scene {scene['id']} ---")

            # Set first frame dynamically from previous scene if required
            if scene["id"] == 6:
                prev_last_frame = output_dir / "scene_05" / "frames" / "shot_001_last.png"
                if prev_last_frame.exists():
                    scene["first_frame"] = str(prev_last_frame)
                    print(f"  Chaining Shot 5 last frame as Shot 6 first frame: {prev_last_frame}")
                else:
                    fallback = "/home/mikeyb/Documents/AI/ComfyUI/comfyui_story_agent/examples/marketing_campaign/eight_sleep_anchors/shot3_end.png"
                    scene["first_frame"] = fallback
                    print(f"  ⚠ Scene 5 last frame not found. Using fallback anchor: {fallback}")

            # Restart ComfyUI for clean memory
            print("  Clearing GPU memory...")
            kill_comfyui()

            proc = start_comfyui()
            if proc is None:
                print("  ✗ ComfyUI failed to start, skipping attempt")
                continue

            clip = generate_single_scene(scene, output_dir)
            kill_comfyui()

            if clip and clip.exists():
                scene_clips[scene["id"]] = clip
                passed = True
                break
            else:
                print(f"  ✗ Attempt {attempt} for Scene {scene['id']} failed")

        if not passed:
            print(f"\n✗ Scene {scene['id']} failed after all retry attempts.")
            sys.exit(1)

    print(f"\n{'═'*50}")
    print("  AUDIO MIXING & ASSEMBLY")
    print(f"{'═'*50}")

    # Generate/ensure audio files are generated with edge-tts
    edge_tts_bin = COMFYUI_ROOT / "venv" / "bin" / "edge-tts"
    
    print("  Generating voiceover segments...")
    
    text_part1 = (
        "Struggling to get a good night's rest? Meet the Eight Sleep Pod. "
        "Its advanced cooling grid and smart app let you customize temperature right from your phone. "
        "The Pod automatically adjusts throughout the night to keep you in deep, restorative sleep cycles. "
        "No more tossing and turning. Wake up feeling refreshed and ready to conquer your day."
    )
    text_part2 = "Upgrade your sleep today."

    audio_part1_female = ART / "audio_part1_female.mp3"
    audio_part1_male = ART / "audio_part1_male.mp3"
    audio_part2_female = ART / "audio_part2.mp3"
    audio_part2_male = ART / "audio_part2_male.mp3"

    # Female Part 1
    subprocess.run([
        str(edge_tts_bin), "--voice", "en-US-AriaNeural",
        "--text", text_part1, "--write-media", str(audio_part1_female)
    ], check=True)
    
    # Male Part 1
    subprocess.run([
        str(edge_tts_bin), "--voice", "en-US-ChristopherNeural",
        "--text", text_part1, "--write-media", str(audio_part1_male)
    ], check=True)

    # Female Part 2
    subprocess.run([
        str(edge_tts_bin), "--voice", "en-US-AriaNeural",
        "--text", text_part2, "--write-media", str(audio_part2_female)
    ], check=True)

    # Male Part 2
    subprocess.run([
        str(edge_tts_bin), "--voice", "en-US-ChristopherNeural",
        "--text", text_part2, "--write-media", str(audio_part2_male)
    ], check=True)

    # 1. Stitch scenes 1, 2, 3, 4, and 5
    clips_part1 = [scene_clips[1], scene_clips[2], scene_clips[3], scene_clips[4], scene_clips[5]]
    v_part1_silent = output_dir / "scenes_part1_silent.mp4"
    
    fp_filt = "; ".join(f"[{i}:v]scale=1280:704,setsar=1,fps=24[v{i}]" for i in range(5))
    concat_filt = "".join(f"[v{i}]" for i in range(5)) + "concat=n=5:v=1:a=0[v]"
    
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(clips_part1[0]), "-i", str(clips_part1[1]), "-i", str(clips_part1[2]),
        "-i", str(clips_part1[3]), "-i", str(clips_part1[4]),
        "-filter_complex", f"{fp_filt}; {concat_filt}",
        "-map", "[v]",
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        str(v_part1_silent)
    ], capture_output=True)

    # 2. Add main voiceover to scenes part 1 (padding audio with silence)
    v_part1_voiced_female = output_dir / "scenes_part1_voiced_female.mp4"
    v_part1_voiced_male = output_dir / "scenes_part1_voiced_male.mp4"
    
    # Dub Female Part 1
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(v_part1_silent),
        "-i", str(audio_part1_female),
        "-filter_complex", "[1:a]apad[a]",
        "-map", "0:v", "-map", "[a]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        str(v_part1_voiced_female)
    ], capture_output=True)

    # Dub Male Part 1
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(v_part1_silent),
        "-i", str(audio_part1_male),
        "-filter_complex", "[1:a]apad[a]",
        "-map", "0:v", "-map", "[a]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        str(v_part1_voiced_male)
    ], capture_output=True)

    # 3. Dub Shot 6 (Female and Male voiceovers, padding audio with silence)
    v6_voiced_female = output_dir / "scene_06_voiced_female.mp4"
    v6_voiced_male = output_dir / "scene_06_voiced_male.mp4"

    # Dub Female
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(scene_clips[6]),
        "-i", str(audio_part2_female),
        "-filter_complex", "[1:a]apad[a]",
        "-map", "0:v", "-map", "[a]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        str(v6_voiced_female)
    ], capture_output=True)

    # Dub Male
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(scene_clips[6]),
        "-i", str(audio_part2_male),
        "-filter_complex", "[1:a]apad[a]",
        "-map", "0:v", "-map", "[a]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        str(v6_voiced_male)
    ], capture_output=True)

    # 4. Concatenate scenes part 1 voiced with scene 6 voiced
    final_female = COMFYUI_ROOT / "eight_sleep_commercial_v13_female.mp4"
    final_male = COMFYUI_ROOT / "eight_sleep_commercial_v13_male.mp4"

    def stitch_final(v_part1: Path, v6: Path, out_path: Path):
        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(v_part1),
            "-i", str(v6),
            "-filter_complex",
            "[0:v]scale=1280:704,setsar=1,fps=24[v0]; [0:a]aresample=48000[a0]; "
            "[1:v]scale=1280:704,setsar=1,fps=24[v1]; [1:a]aresample=48000[a1]; "
            "[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]",
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            str(out_path)
        ], capture_output=True)

    print("  Stitching final commercial with Female voiceover...")
    stitch_final(v_part1_voiced_female, v6_voiced_female, final_female)
    print("  Stitching final commercial with Male voiceover...")
    stitch_final(v_part1_voiced_male, v6_voiced_male, final_male)

    # Copy files to the artifacts directory as well
    shutil.copy(str(final_female), str(ART / final_female.name))
    shutil.copy(str(final_male), str(ART / final_male.name))

    print()
    print("╔═══════════════════════════════════════════════════╗")
    print("║  ✓ SUCCESS: Eight Sleep Commercial Completed      ║")
    print(f"║    Female VO: {final_female.name:<24s}║")
    print(f"║    Male VO:   {final_male.name:<24s}║")
    print("╚═══════════════════════════════════════════════════╝")

if __name__ == "__main__":
    main()
