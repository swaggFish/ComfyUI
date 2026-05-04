"""
Transitions — FFmpeg-based video transition library and assembly

Provides post-generation clip assembly with professional transitions
using FFmpeg's xfade filter. Handles concatenation, crossfades,
dissolves, wipes, and audio mixing.
"""

import subprocess
import shutil
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

FFMPEG_PATH = shutil.which("ffmpeg")

# ── Transition Map ────────────────────────────────────────────────────
# Maps friendly names to FFmpeg xfade transition names.

TRANSITIONS = {
    "cut": None,                    # Direct concatenation, no blend
    "crossfade": "fade",            # Smooth opacity blend
    "dissolve": "dissolve",         # Granular pixel dissolve
    "wipe_left": "wipeleft",        # Directional wipe →
    "wipe_right": "wiperight",      # Directional wipe ←
    "wipe_up": "wipeup",           # Directional wipe ↑
    "wipe_down": "wipedown",       # Directional wipe ↓
    "slide_left": "slideleft",     # Push transition →
    "slide_right": "slideright",   # Push transition ←
    "zoom_in": "smoothup",         # Zoom-style transition
    "fade_black": "fadeblack",     # Fade through black
    "fade_white": "fadewhite",     # Fade through white
    "circle_open": "circleopen",   # Iris open
    "circle_close": "circleclose", # Iris close
    "radial": "radial",            # Radial wipe
    "smoothleft": "smoothleft",    # Smooth directional
    "smoothright": "smoothright",  # Smooth directional
    "pixelize": "pixelize",        # Pixelation transition
    "diagtl": "diagtl",            # Diagonal top-left
    "diagtr": "diagtr",            # Diagonal top-right
}


class TransitionError(Exception):
    """Raised when transition operations fail."""
    pass


def _require_ffmpeg():
    if FFMPEG_PATH is None:
        raise TransitionError(
            "ffmpeg not found. Install with: sudo apt-get install -y ffmpeg"
        )


# ── Single Transition ─────────────────────────────────────────────────

def apply_transition(
    clip_a: str,
    clip_b: str,
    output_path: str,
    transition: str = "crossfade",
    duration: float = 0.5,
) -> str:
    """
    Merge two clips with a transition effect.

    Args:
        clip_a: Path to the first clip
        clip_b: Path to the second clip
        output_path: Where to save the merged result
        transition: Transition type (key from TRANSITIONS)
        duration: Transition duration in seconds

    Returns:
        Path to the output file
    """
    _require_ffmpeg()

    if transition == "cut" or transition not in TRANSITIONS:
        return _concat_cut([clip_a, clip_b], output_path)

    xfade_name = TRANSITIONS[transition]

    # Get duration of clip_a to calculate offset
    clip_a_duration = _get_duration(clip_a)
    offset = max(0, clip_a_duration - duration)

    cmd = [
        FFMPEG_PATH, "-y",
        "-i", str(clip_a),
        "-i", str(clip_b),
        "-filter_complex",
        f"xfade=transition={xfade_name}:duration={duration}:offset={offset}",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]

    _run_ffmpeg(cmd)
    logger.info(
        f"Applied {transition} ({duration}s) → {output_path}"
    )
    return output_path


# ── Full Sequence Assembly ────────────────────────────────────────────

def assemble_sequence(
    clips: list[dict],
    output_path: str,
    fps: int = 25,
) -> str:
    """
    Assemble multiple clips into a final video with transitions.

    Args:
        clips: List of dicts with keys:
            - "path": clip file path
            - "transition": transition type (default "cut")
            - "transition_duration": duration in seconds (default 0.5)
        output_path: Final output video path
        fps: Target frame rate

    Returns:
        Path to the assembled video
    """
    _require_ffmpeg()

    if not clips:
        raise TransitionError("No clips to assemble")

    if len(clips) == 1:
        # Single clip — just convert/copy
        return _convert_single(clips[0]["path"], output_path, fps)

    # First, convert all clips to a common format (mp4 h264)
    temp_dir = (Path(output_path).parent / "_assembly_temp").resolve()
    temp_dir.mkdir(exist_ok=True)

    try:
        normalized = []
        for i, clip_info in enumerate(clips):
            temp_path = str(temp_dir / f"norm_{i:03d}.mp4")
            _normalize_clip(clip_info["path"], temp_path, fps)
            normalized.append({
                **clip_info,
                "path": temp_path,
            })

        # Check if all transitions are cuts (simple concat)
        all_cuts = all(
            c.get("transition", "cut") == "cut" for c in normalized[1:]
        )

        if all_cuts:
            result = _concat_cut(
                [c["path"] for c in normalized], output_path
            )
        else:
            result = _assemble_with_transitions(
                normalized, output_path
            )

        logger.info(f"✓ Assembled {len(clips)} clips → {output_path}")
        return result

    finally:
        # Cleanup temp files
        shutil.rmtree(temp_dir, ignore_errors=True)


def _assemble_with_transitions(clips: list[dict],
                                output_path: str) -> str:
    """Assemble clips sequentially, applying transitions between pairs."""
    if len(clips) < 2:
        return clips[0]["path"]

    temp_dir = (Path(output_path).parent / "_xfade_temp").resolve()
    temp_dir.mkdir(exist_ok=True)

    try:
        current = clips[0]["path"]

        for i in range(1, len(clips)):
            clip = clips[i]
            transition = clip.get("transition", "cut")
            duration = clip.get("transition_duration", 0.5)

            if i < len(clips) - 1:
                # Intermediate result
                temp_out = str((temp_dir / f"merged_{i:03d}.mp4").resolve())
            else:
                # Final output
                temp_out = str(Path(output_path).resolve())

            if transition == "cut":
                current = _concat_cut([current, clip["path"]], temp_out)
            else:
                current = apply_transition(
                    current, clip["path"], temp_out,
                    transition=transition, duration=duration,
                )

        return current

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ── Audio Mixing ──────────────────────────────────────────────────────

def add_soundtrack(
    video_path: str,
    audio_path: str,
    output_path: str,
    audio_volume: float = 0.8,
    loop_audio: bool = True,
) -> str:
    """
    Mix an audio track onto a video.

    Args:
        video_path: Input video path
        audio_path: Audio file path (mp3, wav, etc.)
        output_path: Output video path
        audio_volume: Audio volume (0.0-1.0)
        loop_audio: Whether to loop audio to match video length

    Returns:
        Path to the output file
    """
    _require_ffmpeg()

    input_flags = ["-i", str(video_path)]
    if loop_audio:
        input_flags.extend(["-stream_loop", "-1"])
    input_flags.extend(["-i", str(audio_path)])

    cmd = [
        FFMPEG_PATH, "-y",
        *input_flags,
        "-filter_complex",
        f"[1:a]volume={audio_volume}[a]",
        "-map", "0:v",
        "-map", "[a]",
        "-shortest",
        "-c:v", "copy",
        "-c:a", "aac",
        str(output_path),
    ]

    _run_ffmpeg(cmd)
    logger.info(f"Added soundtrack → {output_path}")
    return output_path


# ── WebP to MP4 Conversion ───────────────────────────────────────────

def webp_to_mp4(webp_path: str, output_path: str,
                fps: int = 25) -> str:
    """Convert an animated WebP to MP4 for assembly compatibility."""
    _require_ffmpeg()

    try:
        from PIL import Image
        import tempfile
        
        with tempfile.TemporaryDirectory() as td:
            with Image.open(webp_path) as im:
                n_frames = im.n_frames
                for i in range(n_frames):
                    im.seek(i)
                    frame_path = Path(td) / f"frame_{i:04d}.png"
                    im.save(frame_path, "PNG")
            
            # Apply "The Crunch": Downsample to 256p then upscale with neighbor for sharp pixels
            vf = "scale=256:-1:flags=neighbor,scale=1024:-1:flags=neighbor"
            
            cmd = [
                FFMPEG_PATH, "-y",
                "-framerate", str(fps),
                "-i", str(Path(td) / "frame_%04d.png"),
                "-vf", vf,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p10le",
                "-crf", "10",
                "-preset", "slow",
                "-an",
                str(output_path),
            ]
            _run_ffmpeg(cmd)
        logger.info(f"Converted WebP → MP4: {output_path}")
        return output_path
    except Exception as e:
        raise TransitionError(f"Failed to convert WebP to MP4 using PIL: {e}")


# ── Internal Helpers ──────────────────────────────────────────────────

def _normalize_clip(input_path: str, output_path: str,
                     fps: int = 25) -> str:
    """Convert a clip to a standard mp4 format for assembly."""
    if input_path.lower().endswith(".webp"):
        return webp_to_mp4(input_path, output_path, fps)

    # Apply "The Crunch" for sharp pixel art edges
    vf = "scale=256:-1:flags=neighbor,scale=1024:-1:flags=neighbor"
    
    cmd = [
        FFMPEG_PATH, "-y",
        "-i", str(input_path),
        "-vf", vf,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p10le",
        "-r", str(fps),
        "-crf", "10",
        "-preset", "slow",
        "-an",  # Strip audio (will be mixed separately)
        str(output_path),
    ]
    _run_ffmpeg(cmd)
    return output_path


def _concat_cut(clip_paths: list[str], output_path: str) -> str:
    """Concatenate clips with hard cuts (no transition)."""
    output_path = str(Path(output_path).resolve())
    # Create concat list file
    list_path = Path(output_path).with_suffix(".txt").resolve()
    with open(list_path, "w") as f:
        for path in clip_paths:
            # Use absolute paths to avoid ffmpeg resolving them relative to the .txt file
            abs_path = str(Path(path).resolve())
            f.write(f"file '{abs_path}'\n")

    cmd = [
        FFMPEG_PATH, "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_path),
        "-c", "copy",
        str(output_path),
    ]

    try:
        _run_ffmpeg(cmd)
    finally:
        list_path.unlink(missing_ok=True)

    return output_path


def _convert_single(input_path: str, output_path: str,
                     fps: int = 25) -> str:
    """Convert a single clip to mp4."""
    return _normalize_clip(input_path, output_path, fps)


def _get_duration(path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return 4.0  # Fallback estimate

    cmd = [
        ffprobe, "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        str(path),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )
        return float(result.stdout.strip()) if result.stdout.strip() else 4.0
    except (subprocess.TimeoutExpired, ValueError):
        return 4.0


def _run_ffmpeg(cmd: list[str]):
    """Run ffmpeg, raising TransitionError on failure."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            raise TransitionError(
                f"ffmpeg failed (exit {result.returncode}): "
                f"{result.stderr[-500:]}"
            )
    except subprocess.TimeoutExpired:
        raise TransitionError("ffmpeg timed out after 300s")


def list_transitions() -> list[str]:
    """Return all available transition names."""
    return list(TRANSITIONS.keys())
