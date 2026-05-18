"""
Shot Planner — StoryBoard and Shot data structures

Defines the shot list format for planning video sequences.
Supports YAML and CSV import/export for easy editing.
"""

import os
import csv
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from . import (
    DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_FRAMES,
    DEFAULT_FPS, DEFAULT_SEED, RESOLUTION_STEP, FRAME_STEP,
)

logger = logging.getLogger(__name__)

# Try importing yaml; provide helpful error if missing
try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class Shot:
    """A single video clip in a sequence."""

    shot_id: int
    prompt: str
    duration_frames: int = DEFAULT_FRAMES  # Must be 8n+1
    camera: str = "static"
    transition_in: str = "cut"  # cut, crossfade, dissolve, etc.
    transition_duration: float = 0.5  # seconds
    first_frame_path: Optional[str] = None  # None = chain from previous
    last_frame_path: Optional[str] = None   # None = free generation
    seed: int = DEFAULT_SEED
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    style_override: Optional[str] = None  # Override global style
    skill: Optional[str] = None  # Use a custom ComfyUI skill (JSON workflow)
    expected_text: Optional[str] = None  # Text that MUST appear legibly in the shot

    def validate(self) -> list[str]:
        """Validate this shot's parameters. Returns list of error strings."""
        errors = []
        if self.width % RESOLUTION_STEP != 0:
            errors.append(
                f"Shot {self.shot_id}: width {self.width} not divisible "
                f"by {RESOLUTION_STEP}"
            )
        if self.height % RESOLUTION_STEP != 0:
            errors.append(
                f"Shot {self.shot_id}: height {self.height} not divisible "
                f"by {RESOLUTION_STEP}"
            )
        if (self.duration_frames - 1) % FRAME_STEP != 0:
            errors.append(
                f"Shot {self.shot_id}: frame count {self.duration_frames} "
                f"doesn't match 8n+1 formula. Use one of: "
                f"{', '.join(str(8*n+1) for n in range(1, 21))}"
            )
        if self.first_frame_path and not os.path.exists(self.first_frame_path):
            errors.append(
                f"Shot {self.shot_id}: first_frame_path not found: "
                f"{self.first_frame_path}"
            )
        if self.last_frame_path and not os.path.exists(self.last_frame_path):
            errors.append(
                f"Shot {self.shot_id}: last_frame_path not found: "
                f"{self.last_frame_path}"
            )
        valid_transitions = {
            "cut", "crossfade", "dissolve", "wipe_left", "wipe_right",
            "slide_left", "slide_right", "zoom_in", "fade_black",
            "fade_white",
        }
        if self.transition_in not in valid_transitions:
            errors.append(
                f"Shot {self.shot_id}: unknown transition '{self.transition_in}'. "
                f"Valid: {', '.join(sorted(valid_transitions))}"
            )
        return errors

    def duration_seconds(self, fps: int = DEFAULT_FPS) -> float:
        """Duration of this clip in seconds."""
        return self.duration_frames / fps

    def to_dict(self) -> dict:
        """Convert to a serializable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Shot":
        """Create a Shot from a dict."""
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})


@dataclass
class StoryBoard:
    """A complete sequence of shots telling a story."""

    title: str = "Untitled Story"
    style: str = "comic"  # Global style name (watercolor, anime, etc.)
    shots: list[Shot] = field(default_factory=list)
    fps: int = DEFAULT_FPS
    style_prompt: str = ""  # Global style prefix for all shots
    negative_prompt: str = ""  # Global negative prompt
    output_dir: str = "./output/story"

    def validate(self) -> list[str]:
        """Validate all shots. Returns list of error strings."""
        errors = []
        if not self.shots:
            errors.append("StoryBoard has no shots")
        for shot in self.shots:
            errors.extend(shot.validate())
        # Check for duplicate shot IDs
        ids = [s.shot_id for s in self.shots]
        if len(ids) != len(set(ids)):
            errors.append("Duplicate shot_id values found")
        return errors

    def total_duration_seconds(self) -> float:
        """Total duration accounting for transitions."""
        if not self.shots:
            return 0.0
        total = sum(s.duration_seconds(self.fps) for s in self.shots)
        # Subtract transition overlaps (except first shot)
        for s in self.shots[1:]:
            if s.transition_in != "cut":
                total -= s.transition_duration
        return total

    def add_shot(self, prompt: str, **kwargs) -> Shot:
        """Add a new shot with auto-incrementing ID."""
        shot_id = max((s.shot_id for s in self.shots), default=0) + 1
        shot = Shot(shot_id=shot_id, prompt=prompt, **kwargs)
        self.shots.append(shot)
        return shot

    # ── YAML I/O ──────────────────────────────────────────────────────

    def to_yaml(self, path: str):
        """Export storyboard to YAML file."""
        if yaml is None:
            raise ImportError("PyYAML required: pip install pyyaml")
        data = {
            "title": self.title,
            "style": self.style,
            "fps": self.fps,
            "style_prompt": self.style_prompt,
            "negative_prompt": self.negative_prompt,
            "output_dir": self.output_dir,
            "shots": [s.to_dict() for s in self.shots],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved storyboard to {path}")

    @classmethod
    def from_yaml(cls, path: str) -> "StoryBoard":
        """Load storyboard from YAML file."""
        if yaml is None:
            raise ImportError("PyYAML required: pip install pyyaml")
        with open(path) as f:
            data = yaml.safe_load(f)
        shots = [Shot.from_dict(s) for s in data.get("shots", [])]
        return cls(
            title=data.get("title", "Untitled"),
            style=data.get("style", "comic"),
            shots=shots,
            fps=data.get("fps", DEFAULT_FPS),
            style_prompt=data.get("style_prompt", ""),
            negative_prompt=data.get("negative_prompt", ""),
            output_dir=data.get("output_dir", "./output/story"),
        )

    # ── CSV I/O ───────────────────────────────────────────────────────

    def to_csv(self, path: str):
        """Export shots to CSV (storyboard metadata in header comment)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fields = list(Shot.__dataclass_fields__.keys())
        with open(path, "w", newline="") as f:
            # Header comment with storyboard metadata
            f.write(f"# title: {self.title}\n")
            f.write(f"# fps: {self.fps}\n")
            f.write(f"# style_prompt: {self.style_prompt}\n")
            f.write(f"# negative_prompt: {self.negative_prompt}\n")
            f.write(f"# output_dir: {self.output_dir}\n")

            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for shot in self.shots:
                writer.writerow(shot.to_dict())
        logger.info(f"Saved shot list CSV to {path}")

    @classmethod
    def from_csv(cls, path: str) -> "StoryBoard":
        """Load storyboard from CSV file."""
        metadata = {}
        data_lines = []
        with open(path) as f:
            for line in f:
                if line.startswith("#"):
                    # Parse metadata from comments
                    key_val = line[1:].strip().split(":", 1)
                    if len(key_val) == 2:
                        metadata[key_val[0].strip()] = key_val[1].strip()
                else:
                    data_lines.append(line)

        import io
        reader = csv.DictReader(io.StringIO("".join(data_lines)))
        shots = []
        for row in reader:
            # Type coercion
            for int_field in ["shot_id", "duration_frames", "seed",
                              "width", "height"]:
                if int_field in row and row[int_field]:
                    row[int_field] = int(row[int_field])
            for float_field in ["transition_duration"]:
                if float_field in row and row[float_field]:
                    row[float_field] = float(row[float_field])
            # Handle None strings from CSV
            for nullable in ["first_frame_path", "last_frame_path",
                             "style_override", "skill"]:
                if nullable in row and row[nullable] in ("", "None"):
                    row[nullable] = None
            shots.append(Shot.from_dict(row))

        return cls(
            title=metadata.get("title", "Untitled"),
            style=metadata.get("style", "comic"),
            shots=shots,
            fps=int(metadata.get("fps", DEFAULT_FPS)),
            style_prompt=metadata.get("style_prompt", ""),
            negative_prompt=metadata.get("negative_prompt", ""),
            output_dir=metadata.get("output_dir", "./output/story"),
        )

    # ── Template Generator ────────────────────────────────────────────

    @classmethod
    def create_template(cls, n_shots: int = 3,
                        style: str = "anime",
                        title: str = "My Story") -> "StoryBoard":
        """Create a starter storyboard template with sensible defaults."""
        from .choreography import ANIMATION_STYLES, build_negative_prompt

        style_prompt = ANIMATION_STYLES.get(style, "")
        negative = build_negative_prompt(style)

        shots = []
        cameras = ["static", "pan_right", "zoom_in", "dolly_forward",
                    "tracking", "crane_up"]
        transitions = ["cut", "crossfade", "dissolve"]
        prompts = [
            "Establishing shot of a vast landscape at golden hour",
            "A character walks along a winding path through the scene",
            "Close-up of the character looking toward the horizon",
            "The character reaches a mysterious doorway",
            "Through the doorway, a new world unfolds",
            "Wide shot of the character stepping into the new world",
        ]

        for i in range(n_shots):
            shots.append(Shot(
                shot_id=i + 1,
                prompt=prompts[i % len(prompts)],
                duration_frames=33,
                camera=cameras[i % len(cameras)],
                transition_in=transitions[i % len(transitions)],
                transition_duration=0.5,
                seed=-1,
            ))

        return cls(
            title=title,
            shots=shots,
            style_prompt=style_prompt,
            negative_prompt=negative,
        )
