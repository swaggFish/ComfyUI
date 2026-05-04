"""
Frame Utilities — Image/video frame manipulation

Extract, resize, and validate frames for FLF2V shot chaining.
Uses subprocess calls to ffmpeg for video frame extraction,
with PIL fallback for image operations.
"""

import os
import subprocess
import shutil
import logging
from pathlib import Path

from . import RESOLUTION_STEP, FRAME_STEP

logger = logging.getLogger(__name__)

# Check for ffmpeg at module load
FFMPEG_PATH = shutil.which("ffmpeg")
FFPROBE_PATH = shutil.which("ffprobe")


class FrameError(Exception):
    """Raised when frame operations fail."""
    pass


# ── Validation ────────────────────────────────────────────────────────

def validate_resolution(width: int, height: int) -> tuple[bool, str]:
    """
    Check if resolution is valid for LTX 2.3.
    Width and height must be divisible by 32.
    """
    if width % RESOLUTION_STEP != 0:
        return False, (
            f"Width {width} not divisible by {RESOLUTION_STEP}. "
            f"Nearest valid: {(width // RESOLUTION_STEP) * RESOLUTION_STEP}"
        )
    if height % RESOLUTION_STEP != 0:
        return False, (
            f"Height {height} not divisible by {RESOLUTION_STEP}. "
            f"Nearest valid: {(height // RESOLUTION_STEP) * RESOLUTION_STEP}"
        )
    return True, "OK"


def validate_frame_count(length: int) -> tuple[bool, str]:
    """
    Check if frame count follows the 8n+1 formula.
    Valid values: 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, ...
    """
    if length < 9:
        return False, f"Frame count {length} too low. Minimum is 9."
    if (length - 1) % FRAME_STEP != 0:
        n = (length - 1) // FRAME_STEP
        lower = FRAME_STEP * n + 1
        upper = FRAME_STEP * (n + 1) + 1
        return False, (
            f"Frame count {length} doesn't match 8n+1 formula. "
            f"Nearest valid: {lower} or {upper}"
        )
    return True, "OK"


def snap_resolution(width: int, height: int) -> tuple[int, int]:
    """Snap width/height to nearest valid LTX resolution."""
    w = round(width / RESOLUTION_STEP) * RESOLUTION_STEP
    h = round(height / RESOLUTION_STEP) * RESOLUTION_STEP
    return max(w, RESOLUTION_STEP), max(h, RESOLUTION_STEP)


def snap_frame_count(length: int) -> int:
    """Snap frame count to nearest valid 8n+1 value."""
    if length < 9:
        return 9
    n = round((length - 1) / FRAME_STEP)
    return FRAME_STEP * n + 1


# ── Frame Extraction (FFmpeg) ─────────────────────────────────────────

def _require_ffmpeg():
    if FFMPEG_PATH is None:
        raise FrameError(
            "ffmpeg not found. Install with: sudo apt-get install -y ffmpeg"
        )


def extract_last_frame(video_path: str, output_path: str = None) -> str:
    """
    Extract the last frame from a video file.

    Args:
        video_path: Path to the video file
        output_path: Where to save the frame. Auto-generated if None.

    Returns:
        Path to the extracted frame (PNG)
    """
    _require_ffmpeg()
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if output_path is None:
        output_path = str(video_path.with_suffix("")) + "_last_frame.png"

    # Get total frame count
    frame_count = _get_frame_count(str(video_path))
    if frame_count <= 0:
        raise FrameError(f"Could not determine frame count for {video_path}")

    # Extract last frame using sseof (seek from end)
    cmd = [
        FFMPEG_PATH, "-y",
        "-sseof", "-0.1",  # Seek to 0.1s before end
        "-i", str(video_path),
        "-frames:v", "1",
        "-update", "1",
        output_path,
    ]
    _run_ffmpeg(cmd)
    logger.info(f"Extracted last frame → {output_path}")
    return output_path


def extract_first_frame(video_path: str, output_path: str = None) -> str:
    """Extract the first frame from a video file."""
    _require_ffmpeg()
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if output_path is None:
        output_path = str(video_path.with_suffix("")) + "_first_frame.png"

    cmd = [
        FFMPEG_PATH, "-y",
        "-i", str(video_path),
        "-frames:v", "1",
        output_path,
    ]
    _run_ffmpeg(cmd)
    logger.info(f"Extracted first frame → {output_path}")
    return output_path


def extract_frame_at(video_path: str, index: int,
                     output_path: str = None) -> str:
    """
    Extract a specific frame by index from a video file.

    Args:
        video_path: Path to the video
        index: 0-based frame index
        output_path: Where to save. Auto-generated if None.
    """
    _require_ffmpeg()
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if output_path is None:
        output_path = (
            str(video_path.with_suffix("")) + f"_frame_{index:04d}.png"
        )

    cmd = [
        FFMPEG_PATH, "-y",
        "-i", str(video_path),
        "-vf", f"select=eq(n\\,{index})",
        "-frames:v", "1",
        "-vsync", "vfr",
        output_path,
    ]
    _run_ffmpeg(cmd)
    logger.info(f"Extracted frame {index} → {output_path}")
    return output_path


def get_video_info(video_path: str) -> dict:
    """
    Get video metadata (resolution, fps, duration, frame count).

    Returns:
        dict with keys: width, height, fps, duration, frame_count
    """
    if FFPROBE_PATH is None:
        raise FrameError("ffprobe not found.")

    cmd = [
        FFPROBE_PATH, "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise FrameError(f"ffprobe failed: {result.stderr}")

    import json
    data = json.loads(result.stdout)
    video_stream = next(
        (s for s in data.get("streams", []) if s["codec_type"] == "video"),
        None,
    )
    if video_stream is None:
        raise FrameError(f"No video stream in {video_path}")

    # Parse FPS from r_frame_rate (e.g., "25/1")
    fps_parts = video_stream.get("r_frame_rate", "25/1").split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 25.0

    return {
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "fps": fps,
        "duration": float(data.get("format", {}).get("duration", 0)),
        "frame_count": int(video_stream.get("nb_frames", 0)),
    }


# ── Image Manipulation ────────────────────────────────────────────────

def resize_to_ltx(image_path: str, width: int, height: int,
                  output_path: str = None) -> str:
    """
    Resize an image to LTX-valid resolution, maintaining aspect ratio
    and padding/cropping to exact dimensions.

    Args:
        image_path: Source image path
        width: Target width (must be divisible by 32)
        height: Target height (must be divisible by 32)
        output_path: Where to save. Auto-generated if None.
    """
    _require_ffmpeg()
    width, height = snap_resolution(width, height)

    if output_path is None:
        p = Path(image_path)
        output_path = str(p.with_stem(f"{p.stem}_ltx_{width}x{height}"))

    # Scale to fill, then center crop to exact size
    cmd = [
        FFMPEG_PATH, "-y",
        "-i", str(image_path),
        "-vf", (
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height}"
        ),
        "-frames:v", "1",
        output_path,
    ]
    _run_ffmpeg(cmd)
    logger.info(f"Resized to {width}x{height} → {output_path}")
    return output_path


# ── WebP Frame Extraction ─────────────────────────────────────────────
# ComfyUI SaveAnimatedWEBP outputs .webp files. These need special
# handling to extract individual frames.

def extract_last_frame_from_webp(webp_path: str,
                                  output_path: str = None) -> str:
    """Extract the last frame from an animated WebP file using Pillow."""
    if output_path is None:
        output_path = str(Path(webp_path).with_suffix("")) + "_last.png"

    try:
        from PIL import Image
        with Image.open(webp_path) as im:
            im.seek(im.n_frames - 1)
            im.save(output_path, "PNG")
        logger.info(f"Extracted last frame from WebP using PIL → {output_path}")
    except Exception as e:
        raise FrameError(f"PIL could not extract frame from {webp_path}: {e}")

    return output_path


# ── Internal Helpers ──────────────────────────────────────────────────

def _get_frame_count(path: str) -> int:
    """Get total frame count of a video/animation."""
    if FFPROBE_PATH is None:
        return 0
    try:
        cmd = [
            FFPROBE_PATH, "-v", "quiet",
            "-count_frames",
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames",
            "-of", "csv=p=0",
            str(path),
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60
        )
        return int(result.stdout.strip()) if result.stdout.strip() else 0
    except (subprocess.TimeoutExpired, ValueError):
        return 0


def _run_ffmpeg(cmd: list[str]):
    """Run an ffmpeg command, raising FrameError on failure."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            raise FrameError(
                f"ffmpeg failed (exit {result.returncode}): {result.stderr}"
            )
    except subprocess.TimeoutExpired:
        raise FrameError("ffmpeg timed out after 120s")
