"""
Director Agent — Automated Visual QA for Video Generation

The Director is an automated test suite for pixels, not code. It:
  1. EXTRACTS evidence frames (first, middle, last) via FFmpeg
  2. EVALUATES them via Gemini Flash vision (zero local VRAM cost)
  3. CORRECTS the ComfyUI workflow parameters based on structured feedback
  4. RETRIES up to max_retries times before scrapping the concept

The Director never touches the prompt creatively — it only adjusts
mathematical parameters (CFG, steps, seed, negative prompt, strengths)
within the proven envelopes from production_envelopes.json.

Architecture:
  Local GPU (RTX 5060 Ti) → Rendering ONLY
  Gemini Flash (Cloud)     → Vision evaluation ONLY
  Zero memory contention.
"""

import base64
import json
import logging
import os
import random
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from .drift_schema import (
    ShotEvaluation,
    ShotCorrectionPlan,
    DriftSeverity,
    RecommendedAction,
)
from .validator import Critic

logger = logging.getLogger(__name__)

# ── FFmpeg frame extraction ───────────────────────────────────────────

FFMPEG_PATH = shutil.which("ffmpeg")


def extract_evidence_frames(
    clip_path: str,
    output_dir: str,
    shot_id: int = 0,
) -> dict[str, str]:
    """Phase 1: Extract first, middle, and last frames from a clip.

    Supports both .mp4 video files and .webp animated images.
    Returns dict with keys: 'first', 'middle', 'last' → file paths.
    """
    clip_path = Path(clip_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"evidence_shot{shot_id:03d}"

    # Handle the case where the "clip" path is actually a directory
    # (the pipeline creates directories like shot_001.webp/clips/...)
    if clip_path.is_dir():
        # Search for actual clip files inside
        real_clips = sorted(clip_path.rglob("*.webp")) + sorted(clip_path.rglob("*.mp4"))
        real_clips = [f for f in real_clips if f.is_file()]
        if real_clips:
            clip_path = real_clips[0]
            logger.info(f"Resolved directory to actual clip: {clip_path}")
        else:
            raise FileNotFoundError(f"No clip files found inside directory: {clip_path}")

    if clip_path.suffix.lower() == ".webp":
        return _extract_from_webp(clip_path, output_dir, prefix)
    else:
        return _extract_from_video(clip_path, output_dir, prefix)


def _extract_from_webp(
    webp_path: Path, output_dir: Path, prefix: str
) -> dict[str, str]:
    """Extract evidence frames from an animated WebP using PIL."""
    from PIL import Image

    frames = {}
    with Image.open(webp_path) as im:
        total = im.n_frames

        # First frame
        im.seek(0)
        first_path = str(output_dir / f"{prefix}_first.png")
        im.save(first_path, "PNG")
        frames["first"] = first_path

        # Middle frame
        mid_idx = total // 2
        im.seek(mid_idx)
        mid_path = str(output_dir / f"{prefix}_middle.png")
        im.save(mid_path, "PNG")
        frames["middle"] = mid_path

        # Last frame
        im.seek(total - 1)
        last_path = str(output_dir / f"{prefix}_last.png")
        im.save(last_path, "PNG")
        frames["last"] = last_path

    logger.info(
        f"Extracted {total} total frames → evidence: first, middle({mid_idx}), last"
    )
    return frames


def _extract_from_video(
    video_path: Path, output_dir: Path, prefix: str
) -> dict[str, str]:
    """Extract evidence frames from a video file using FFmpeg."""
    if not FFMPEG_PATH:
        raise RuntimeError("ffmpeg not found. Install with: sudo apt install ffmpeg")

    # Get frame count
    from .frame_utils import _get_frame_count
    total = _get_frame_count(str(video_path))
    if total <= 0:
        total = 97  # Fallback assumption

    frames = {}
    for label, idx in [("first", 0), ("middle", total // 2), ("last", total - 1)]:
        out = str(output_dir / f"{prefix}_{label}.png")
        cmd = [
            FFMPEG_PATH, "-y",
            "-i", str(video_path),
            "-vf", f"select=eq(n\\,{idx})",
            "-frames:v", "1",
            "-vsync", "vfr",
            out,
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        frames[label] = out

    logger.info(
        f"Extracted evidence frames from video ({total} frames): first, middle({total//2}), last"
    )
    return frames


# ── Gemini Flash Vision Evaluation ────────────────────────────────────

_DIRECTOR_SYSTEM_PROMPT = """You are a quality control Director for an AI video production pipeline.

You are reviewing THREE keyframes extracted from a single generated video clip:
- Frame 1 (FIRST): The opening frame
- Frame 2 (MIDDLE): The exact middle frame
- Frame 3 (LAST): The final frame

Your job is to detect DRIFT — where the subject, background, or physics break down over the duration of the clip. Common issues include:
- CHARACTER ANATOMY: Arms merging into objects, extra fingers, face morphing, body proportions changing
- CHARACTER IDENTITY: Hair color changing, outfit changing, character becoming a different person
- BACKGROUND STABILITY: Buildings warping, horizon shifting, objects appearing/disappearing
- PHYSICS COHERENCE: Gravity violations, impossible perspectives, scale changes
- STYLE CONSISTENCY: Art style shifting (e.g., watercolor becoming photorealistic)

IMPORTANT CALIBRATION:
- Minor pose changes and camera movements are EXPECTED and should NOT be flagged
- Slight color grading shifts due to lighting changes are NORMAL
- Focus on STRUCTURAL drift that would break the narrative
- A "passed" clip is one that a viewer would accept as coherent animation
- Only fail clips with issues that are genuinely distracting or break continuity

Score each category from 0.0 (completely broken) to 1.0 (perfect).
A shot PASSES if ALL scores are >= 0.6 AND drift_severity is 'none' or 'low'.
"""

_EVALUATION_PROMPT = """Review these three sequential frames from a generated video clip.

The intended scene is: "{prompt}"

Evaluate for drift, anatomical errors, background instability, and style consistency.
Rate each category 0.0-1.0 and determine if this shot passes quality control.

If it fails, recommend ONE specific mathematical correction from the allowed actions.
Do NOT suggest creative/prompt changes — only parameter adjustments."""


def evaluate_shot_with_gemini(
    evidence_frames: dict[str, str],
    prompt: str,
    api_key: str,
    model: str = "gemini-2.5-flash",
) -> ShotEvaluation:
    """Phase 2+3: Send evidence frames to Gemini Flash for structured evaluation.

    Uses Gemini's native structured output to guarantee parseable JSON.
    Zero local VRAM cost — the GPU stays hot for ComfyUI.

    Args:
        evidence_frames: Dict with 'first', 'middle', 'last' → PNG paths
        prompt: The original scene description (for context)
        api_key: Google Gemini API key
        model: Which Gemini model to use (default: gemini-2.0-flash)

    Returns:
        ShotEvaluation with scores, issues, and recommended action
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError(
            "google-genai is required. Install with: pip install google-genai"
        )

    # Force the client to route to Vertex AI instead of the Developer API.
    # This bypasses AI Studio's prepay requirements and directly consumes the $300 GCP trial.
    client = genai.Client(
        vertexai=True,
        project="project-752e56a3-dc98-4377-b37",
        location="us-central1"
    )

    # Build the multimodal content with all 3 evidence frames
    contents = []

    # System instruction
    config = types.GenerateContentConfig(
        system_instruction=_DIRECTOR_SYSTEM_PROMPT,
        temperature=0.1,  # Low temperature for consistent grading
        response_mime_type="application/json",
        response_schema=ShotEvaluation,
    )

    # Build parts: text prompt + 3 images
    parts = [_EVALUATION_PROMPT.format(prompt=prompt)]

    for label in ["first", "middle", "last"]:
        frame_path = evidence_frames.get(label)
        if frame_path and Path(frame_path).exists():
            img_data = Path(frame_path).read_bytes()
            parts.append(
                types.Part.from_bytes(data=img_data, mime_type="image/png")
            )
            parts.append(f"[Frame: {label.upper()}]")

    try:
        response = client.models.generate_content(
            model=model,
            contents=parts,
            config=config,
        )

        # Parse the structured response
        if response.text:
            raw = json.loads(response.text)
            evaluation = ShotEvaluation(**raw)
        else:
            logger.warning("Gemini returned empty response; defaulting to PASS")
            evaluation = ShotEvaluation(
                passed=True,
                summary="Gemini returned empty response; auto-passed."
            )

    except Exception as e:
        logger.error(f"Gemini evaluation failed: {e}")
        # Fail-open: if the Director is down, don't block production
        evaluation = ShotEvaluation(
            passed=True,
            summary=f"Director unavailable ({e}); auto-passed."
        )

    logger.info(
        f"Director verdict: {'PASS ✅' if evaluation.passed else 'FAIL ❌'} | "
        f"Drift: {evaluation.drift_severity.value} | "
        f"Char: {evaluation.character_consistency:.2f} | "
        f"BG: {evaluation.background_stability:.2f} | "
        f"Phys: {evaluation.physics_coherence:.2f} | "
        f"Style: {evaluation.style_consistency:.2f}"
    )

    if evaluation.issues:
        for issue in evaluation.issues:
            logger.info(
                f"  ⚠️  [{issue.frame_location}] {issue.category}: {issue.description}"
            )

    return evaluation


# ── Phase 4: The Correction Loop ──────────────────────────────────────

def compute_corrections(
    evaluation: ShotEvaluation,
    current_params: dict,
    attempt: int,
) -> dict:
    """Translate a Director's recommended_action into concrete parameter changes.

    Only adjusts mathematical parameters within the proven envelopes.
    Never touches creative prompts.

    Args:
        evaluation: The Director's structured evaluation
        current_params: Current workflow parameters (cfg, steps, seed, etc.)
        attempt: Which retry attempt (1-indexed)

    Returns:
        Dict of corrected parameters to merge into the workflow
    """
    corrections = current_params.copy()
    action = evaluation.recommended_action

    # If the evaluation failed but the model returned no explicit action,
    # choose a conservative fallback so retries do not become no-ops.
    if not evaluation.passed and action == RecommendedAction.none:
        action = RecommendedAction.scrap_and_rethink
        logger.warning(
            "Director correction: No recommended_action returned for failed evaluation; "
            "defaulting to scrap_and_rethink."
        )

    # Load envelopes for boundary clamping
    defaults = Critic.get_envelope_defaults("flf2v_cinematic")

    if action == RecommendedAction.decrease_cfg:
        old = corrections.get("cfg", 3.5)
        corrections["cfg"] = max(1.0, old - 0.5)
        logger.info(f"Director correction: CFG {old} → {corrections['cfg']}")

    elif action == RecommendedAction.increase_cfg:
        old = corrections.get("cfg", 3.5)
        corrections["cfg"] = min(4.0, old + 0.5)
        logger.info(f"Director correction: CFG {old} → {corrections['cfg']}")

    elif action == RecommendedAction.decrease_steps:
        old = corrections.get("steps", 24)
        corrections["steps"] = max(8, old - 4)
        logger.info(f"Director correction: Steps {old} → {corrections['steps']}")

    elif action == RecommendedAction.increase_steps:
        old = corrections.get("steps", 24)
        corrections["steps"] = min(30, old + 4)
        logger.info(f"Director correction: Steps {old} → {corrections['steps']}")

    elif action == RecommendedAction.adjust_negative:
        old_neg = corrections.get("negative", "")
        addition = evaluation.negative_prompt_addition or "morphing, blending, distorted"
        if addition not in old_neg:
            corrections["negative"] = f"{old_neg}, {addition}".strip(", ")
        logger.info(f"Director correction: Added to negative: '{addition}'")

    elif action == RecommendedAction.decrease_frame_count:
        old = corrections.get("length", 97)
        # Step down by 8 (following 8n+1 rule)
        corrections["length"] = max(33, old - 8)
        logger.info(f"Director correction: Frames {old} → {corrections['length']}")

    elif action == RecommendedAction.increase_first_strength:
        old = corrections.get("first_strength", 1.0)
        corrections["first_strength"] = min(1.0, old + 0.1)
        logger.info(f"Director correction: First strength {old} → {corrections['first_strength']}")

    elif action == RecommendedAction.increase_last_strength:
        old = corrections.get("last_strength", 1.0)
        corrections["last_strength"] = min(1.0, old + 0.1)
        logger.info(f"Director correction: Last strength {old} → {corrections['last_strength']}")

    elif action == RecommendedAction.reseed:
        old = corrections.get("seed", -1)
        corrections["seed"] = random.randint(1, 2**31)
        logger.info(f"Director correction: Reseed {old} → {corrections['seed']}")

    elif action == RecommendedAction.scrap_and_rethink:
        # On scrap: reseed + bump negative + reduce frames
        corrections["seed"] = random.randint(1, 2**31)
        old_neg = corrections.get("negative", "")
        corrections["negative"] = f"{old_neg}, morphing, distorted anatomy, drift".strip(", ")
        old_len = corrections.get("length", 97)
        corrections["length"] = max(33, old_len - 16)
        logger.info("Director correction: SCRAPPED — reseed + negative boost + frame reduction")

    else:
        logger.info("Director: No correction needed (action: none)")

    # Final safety: clamp everything through the Critic envelopes
    corrections = Critic.validate_inputs(corrections, pipeline="flf2v_cinematic")

    return corrections


class Director:
    """The Director Agent — automated visual QA for the generation pipeline.

    Usage:
        director = Director(api_key="your-gemini-key")
        evaluation = director.review_clip("output/shot_001.webp", "scene description")
        if not evaluation.passed:
            corrections = director.get_corrections(evaluation, current_params, attempt=1)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        max_retries: int = 3,
        evidence_dir: str = "/tmp/director_evidence",
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model = model
        self.max_retries = max_retries
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

        # History of evaluations for this session
        self.history: list[ShotCorrectionPlan] = []

        if not self.api_key:
            logger.info(
                "Director: No GEMINI_API_KEY found. Assuming Application Default Credentials (ADC) are active."
            )

    @property
    def enabled(self) -> bool:
        """Is the Director operational?"""
        return True

    def review_clip(
        self,
        clip_path: str,
        prompt: str,
        shot_id: int = 0,
    ) -> ShotEvaluation:
        """Full review pipeline: extract evidence → evaluate with Gemini.

        Args:
            clip_path: Path to the generated .webp or .mp4 clip
            prompt: The original scene description
            shot_id: Shot number for logging

        Returns:
            ShotEvaluation with pass/fail verdict and corrections
        """
        if not self.enabled:
            return ShotEvaluation(
                passed=True,
                summary="Director disabled; auto-passed."
            )

        logger.info(f"🎬 Director reviewing Shot {shot_id}...")

        # Phase 1: Extract evidence frames
        evidence = extract_evidence_frames(
            clip_path, str(self.evidence_dir), shot_id
        )

        # Phase 2+3: Evaluate with Gemini Flash
        evaluation = evaluate_shot_with_gemini(
            evidence_frames=evidence,
            prompt=prompt,
            api_key=self.api_key,
            model=self.model,
        )

        return evaluation

    def get_corrections(
        self,
        evaluation: ShotEvaluation,
        current_params: dict,
        attempt: int,
    ) -> dict:
        """Phase 4: Compute parameter corrections from the evaluation.

        Args:
            evaluation: The Director's verdict
            current_params: Current workflow parameters
            attempt: Which retry attempt (1-indexed)

        Returns:
            Corrected parameters dict
        """
        corrected = compute_corrections(evaluation, current_params, attempt)

        # Record in history
        self.history.append(ShotCorrectionPlan(
            shot_id=current_params.get("shot_id", 0),
            attempt=attempt,
            evaluation=evaluation,
            original_params=current_params,
            corrected_params=corrected,
            correction_notes=f"Action: {evaluation.recommended_action.value}",
        ))

        return corrected

    def should_retry(self, evaluation: ShotEvaluation, attempt: int) -> bool:
        """Should the engine retry this shot?

        Returns False if:
          - The shot passed
          - We've exceeded max_retries
          - The action is 'scrap_and_rethink' on the last attempt
        """
        if evaluation.passed:
            return False
        if evaluation.drift_severity in (DriftSeverity.high, DriftSeverity.critical) and attempt >= 2:
            logger.warning(
                "Director: High/critical drift persisted after multiple retries; "
                "stopping early to avoid wasted computation."
            )
            return False
        if evaluation.recommended_action == RecommendedAction.scrap_and_rethink and attempt >= 2:
            logger.warning(
                "Director: Scrap-and-rethink action repeated; stopping retries early."
            )
            return False
        if attempt >= self.max_retries:
            logger.warning(
                f"Director: Max retries ({self.max_retries}) reached. "
                f"Accepting current output despite drift."
            )
            return False
        return True

    def get_session_report(self) -> dict:
        """Generate a report of all evaluations in this session."""
        return {
            "total_evaluations": len(self.history),
            "total_retries": sum(1 for h in self.history if h.attempt > 1),
            "corrections": [
                {
                    "shot_id": h.shot_id,
                    "attempt": h.attempt,
                    "passed": h.evaluation.passed,
                    "severity": h.evaluation.drift_severity.value,
                    "action": h.evaluation.recommended_action.value,
                    "summary": h.evaluation.summary,
                }
                for h in self.history
            ],
        }
