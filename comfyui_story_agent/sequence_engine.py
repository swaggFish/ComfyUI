"""Sequence Engine — Shot chaining orchestration

The core brain that generates a sequence of video clips with continuity:
1. For Shot 1: T2V or I2V depending on whether a first_frame exists
2. For Shot N>1: Extract last frame from Shot N-1, use as first frame
3. If a shot has last_frame_path: use FLF2V (both anchors)
4. Otherwise: I2V (first frame only, free ending)

Each clip is saved with metadata for reproducibility.

Director Integration:
  After each shot renders, the Director agent extracts evidence frames
  (first, middle, last), sends them to Gemini Flash for drift analysis,
  and if the shot fails QA, automatically corrects parameters and retries
  up to max_retries times.
"""

import json
import os
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .comfyui_client import ComfyUIClient, ComfyUIError
from .shot_planner import Shot, StoryBoard
from .workflow_templates import (
    build_text_to_video, build_image_to_video, build_flf2v,
    build_hybrid_i2v, build_hybrid_flf2v,
)
from .choreography import compose_prompt, build_negative_prompt
from .frame_utils import (
    extract_last_frame, extract_last_frame_from_webp,
    validate_resolution, validate_frame_count,
)
from .skills import SkillManager
from .validator import Critic
from .director import Director
from .text_agent import TextDirector
from . import rife_interpolator

logger = logging.getLogger(__name__)


@dataclass
class ClipResult:
    """Result of generating a single video clip."""
    shot_id: int
    video_path: str = ""
    frames_dir: str = ""
    last_frame_path: str = ""
    first_frame_path: str = ""
    prompt_used: str = ""
    seed_used: int = -1
    width: int = 0
    height: int = 0
    frame_count: int = 0
    generation_time: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "shot_id": self.shot_id,
            "video_path": self.video_path,
            "last_frame_path": self.last_frame_path,
            "first_frame_path": self.first_frame_path,
            "prompt_used": self.prompt_used,
            "seed_used": self.seed_used,
            "width": self.width,
            "height": self.height,
            "frame_count": self.frame_count,
            "generation_time": self.generation_time,
        }


class SequenceEngine:
    """Orchestrates sequential video clip generation with continuity.

    Now includes the Director agent for automated visual QA.
    After each shot renders, the Director evaluates drift and
    automatically corrects parameters for retry if needed.
    """

    def __init__(
        self,
        client: ComfyUIClient,
        output_dir: str,
        gemini_api_key: Optional[str] = None,
        enable_director: bool = True,
        director_max_retries: int = 3,
    ):
        """
        Args:
            client: Connected ComfyUIClient instance
            output_dir: Directory to save all generated clips
            gemini_api_key: Optional API key for hybrid mode + Director
            enable_director: Whether to enable the Director QA agent
            director_max_retries: Max retry attempts per shot (default 3)
        """
        self.client = client
        self.gemini_api_key = gemini_api_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir = self.output_dir / "clips"
        self.clips_dir.mkdir(exist_ok=True)
        self.frames_dir = self.output_dir / "frames"
        self.frames_dir.mkdir(exist_ok=True)

        # Initialize Skill Manager (pointing to ComfyUI root)
        self.skill_manager = SkillManager(workspace_root="/home/mikeyb/Documents/AI/ComfyUI")

        # Initialize Director Agent
        director_key = gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
        if enable_director:
            self.director = Director(
                api_key=gemini_api_key,
                max_retries=director_max_retries,
                evidence_dir=os.path.join(output_dir, "evidence")
            )
            self.text_director = TextDirector(
                api_key=gemini_api_key,
                max_retries=director_max_retries,
                evidence_dir=os.path.join(output_dir, "evidence_text")
            )
            logger.info(
                f"Director Agent ENABLED (max_retries={director_max_retries})"
            )
        else:
            self.director = None
            logger.info("Director Agent DISABLED (explicitly disabled)")

    def generate_sequence(
        self,
        storyboard: StoryBoard,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
    ) -> list[ClipResult]:
        """
        Generate all clips in sequence, chaining last→first frames.

        Args:
            storyboard: The StoryBoard defining the sequence
            on_progress: Callback(message, current_shot, total_shots)

        Returns:
            List of ClipResult objects for each generated clip
        """
        # Validate
        errors = storyboard.validate()
        if errors:
            raise ValueError(
                f"StoryBoard validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        # Check ComfyUI connectivity
        if not self.client.is_alive():
            raise ComfyUIError(
                "ComfyUI is not running. Start it with: ./launch_ltx.sh"
            )

        results = []
        previous_last_frame = None

        for i, shot in enumerate(storyboard.shots):
            if on_progress:
                on_progress(
                    f"Generating shot {shot.shot_id}: {shot.prompt[:50]}...",
                    i + 1,
                    len(storyboard.shots),
                )

            logger.info(
                f"═══ Shot {shot.shot_id} ({i+1}/{len(storyboard.shots)}) ═══"
            )

            result = self._generate_with_director_loop(
                shot=shot,
                storyboard=storyboard,
                previous_last_frame=previous_last_frame,
            )
            results.append(result)

            # Chain: this shot's last frame → next shot's first frame
            previous_last_frame = result.last_frame_path
            logger.info(
                f"Shot {shot.shot_id} done in {result.generation_time:.1f}s"
            )

        # Save sequence metadata + Director report
        self._save_sequence_metadata(storyboard, results)

        if self.director:
            report = self.director.get_session_report()
            report_path = self.output_dir / "director_report.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(
                f"Director Report: {report['total_evaluations']} evaluations, "
                f"{report['total_retries']} retries → {report_path}"
            )

        logger.info(
            f"✓ Sequence complete: {len(results)} clips generated "
            f"in {sum(r.generation_time for r in results):.1f}s total"
        )

        # ── RIFE Interpolation Pass ──────────────────────────────────
        # If storyboard was generated at a low fps (≤8) for VRAM savings,
        # automatically upsample all clips to 24fps using TensorRT RIFE.
        source_fps = getattr(storyboard, 'fps', 24)
        if source_fps <= 8:
            target_fps = 24
            multiplier = target_fps // source_fps
            logger.info(
                f"🎞️  Low-fps storyboard detected ({source_fps}fps). "
                f"Running RIFE {multiplier}x interpolation → {target_fps}fps..."
            )
            rife_out_dir = self.output_dir / "clips_24fps"
            rife_out_dir.mkdir(exist_ok=True)
            for result in results:
                if not result.video_path:
                    continue
                clip_in = Path(result.video_path)
                if not clip_in.exists():
                    logger.warning(f"Clip not found for RIFE: {clip_in}")
                    continue
                clip_out = rife_out_dir / f"{clip_in.stem}_24fps.mp4"
                logger.info(f"  RIFE: {clip_in.name} → {clip_out.name}")
                ok = rife_interpolator.interpolate_clip(
                    input_path=str(clip_in),
                    output_path=str(clip_out),
                    multiplier=multiplier,
                    comfyui_host=self.client.host,
                    comfyui_port=self.client.port,
                )
                if ok:
                    result.metadata["rife_24fps_path"] = str(clip_out)
                    logger.info(f"  ✓ RIFE done → {clip_out.name}")
                else:
                    logger.warning(f"  ✗ RIFE failed for {clip_in.name} — original kept")

        return results

    def _generate_with_director_loop(
        self,
        shot: Shot,
        storyboard: Optional[StoryBoard] = None,
        previous_last_frame: Optional[str] = None,
    ) -> ClipResult:
        """Generate a shot with the Director's review-correct-retry loop.

        1. Generate the clip
        2. Director extracts evidence frames (first, middle, last)
        3. Director sends frames to Gemini Flash for evaluation
        4. If FAIL: apply corrections and retry (up to max_retries)
        5. If PASS or max_retries exceeded: return the result
        """
        current_params = {
            "cfg": getattr(shot, "cfg", 3.5),
            "steps": getattr(shot, "steps", 24),
            "seed": shot.seed,
            "length": shot.duration_frames,
            "negative": "",
            "first_strength": getattr(shot, "first_strength", 1.0),
            "last_strength": getattr(shot, "last_strength", 1.0),
        }

        for attempt in range(1, (self.director.max_retries if self.director else 1) + 1):
            if attempt > 1:
                logger.info(
                    f"🔄 Director retry #{attempt} for Shot {shot.shot_id}"
                )
                # Apply corrections to the shot object for retry
                shot.seed = current_params.get("seed", shot.seed)
                shot.duration_frames = current_params.get("length", shot.duration_frames)
                if "steps" in current_params:
                    shot.steps = current_params["steps"]
                if "cfg" in current_params:
                    shot.cfg = current_params["cfg"]
                if "negative" in current_params and current_params["negative"]:
                    shot.negative = current_params["negative"]
                if "first_strength" in current_params:
                    shot.first_strength = current_params["first_strength"]
                if "last_strength" in current_params:
                    shot.last_strength = current_params["last_strength"]

            result = self.generate_single_shot(
                shot=shot,
                storyboard=storyboard,
                previous_last_frame=previous_last_frame,
                on_comfyui_progress=None,
            )

            # Skip Director review if disabled or no clip was produced
            if not self.director or not result.video_path:
                return result

            # Director reviews the clip
            evaluation = self.director.review_clip(
                clip_path=result.video_path,
                prompt=shot.prompt,
                shot_id=shot.shot_id,
            )
            
            text_evaluation = None
            if hasattr(shot, 'expected_text') and shot.expected_text:
                text_evaluation = self.text_director.review_clip(
                    clip_path=result.video_path,
                    expected_text=shot.expected_text,
                    shot_id=shot.shot_id,
                )
                if not text_evaluation.passed:
                    evaluation.passed = False  # Fail the main evaluation if text fails
                    evaluation.recommended_action = text_evaluation.recommended_action
                    evaluation.summary = f"TextDirector failed: {text_evaluation.summary} | " + evaluation.summary
                    result.metadata["text_director_evaluation"] = text_evaluation.dict()

            # Store evaluation in result metadata
            result.metadata["director_evaluation"] = {
                "attempt": attempt,
                "passed": evaluation.passed,
                "drift_severity": evaluation.drift_severity.value,
                "character_consistency": evaluation.character_consistency,
                "background_stability": evaluation.background_stability,
                "physics_coherence": evaluation.physics_coherence,
                "style_consistency": evaluation.style_consistency,
                "summary": evaluation.summary,
            }

            if evaluation.passed:
                logger.info(
                    f"✅ Director APPROVED Shot {shot.shot_id} "
                    f"(attempt {attempt})"
                )
                return result

            # Should we retry?
            if text_evaluation and not self.text_director.should_retry(text_evaluation, attempt):
                logger.warning(
                    f"⚠️  TextDirector: Shot {shot.shot_id} failed but "
                    f"retry policy says stop after attempt {attempt}."
                )
                return result

            if not self.director.should_retry(evaluation, attempt):
                logger.warning(
                    f"⚠️  Director: Shot {shot.shot_id} failed but "
                    f"max retries reached. Accepting."
                )
                return result

            # Compute corrections
            current_params = self.director.get_corrections(
                evaluation, current_params, attempt
            )
            shot.seed = current_params.get("seed", shot.seed)
            shot.duration_frames = current_params.get("length", shot.duration_frames)
            if "steps" in current_params:
                shot.steps = current_params["steps"]
            if "cfg" in current_params:
                shot.cfg = current_params["cfg"]
            if "negative" in current_params and current_params["negative"]:
                shot.negative = current_params["negative"]
            if "first_strength" in current_params:
                shot.first_strength = current_params["first_strength"]
            if "last_strength" in current_params:
                shot.last_strength = current_params["last_strength"]
            logger.info(
                f"Director corrections for retry: {evaluation.recommended_action.value}"
            )

        return result  # Shouldn't reach here, but safety return

    def generate_single_shot(
        self,
        shot: Shot,
        storyboard: Optional[StoryBoard] = None,
        previous_last_frame: Optional[str] = None,
        on_comfyui_progress: Optional[Callable] = None,
    ) -> ClipResult:
        """
        Generate a single video clip.

        Logic:
        - If first_frame and last_frame → FLF2V
        - If first_frame only → I2V
        - If no frames → T2V
        """
        # Determine the style and compose the prompt
        style = storyboard.style if storyboard else "comic"
        style_prompt = ""
        negative = ""

        if storyboard:
            style_prompt = storyboard.style_prompt
            negative = storyboard.negative_prompt
        if hasattr(shot, 'negative') and shot.negative:
            negative = shot.negative
        if not negative:
            negative = build_negative_prompt(style)

        full_prompt = compose_prompt(
            scene_description=shot.prompt,
            camera=shot.camera,
            style=style,
            style_prompt=style_prompt if style_prompt else "",
        )

        # Determine first frame source
        first_frame = shot.first_frame_path or previous_last_frame
        last_frame = shot.last_frame_path

        # Determine generation mode
        if first_frame and last_frame:
            mode = "flf2v"
        elif first_frame:
            mode = "i2v"
        else:
            mode = "t2v"

        logger.info(f"Mode: {mode} | Prompt: {full_prompt[:80]}...")

        filename_prefix = f"shot_{shot.shot_id:03d}"

        # Upload images to ComfyUI if needed
        first_image_name = None
        last_image_name = None

        if first_frame:
            upload_result = self.client.upload_image(first_frame)
            first_image_name = upload_result.get(
                "name", Path(first_frame).name
            )
            logger.info(f"Uploaded first frame: {first_image_name}")

        if last_frame:
            upload_result = self.client.upload_image(last_frame)
            last_image_name = upload_result.get(
                "name", Path(last_frame).name
            )
            logger.info(f"Uploaded last frame: {last_image_name}")

        # Apply Guardrail 2: Parameter Envelopes
        source_fps = getattr(storyboard, 'fps', 24) if storyboard else 24
        validated_inputs = Critic.validate_inputs({
            "prompt": full_prompt,
            "negative": negative,
            "width": shot.width,
            "height": shot.height,
            "seed": shot.seed,
            "length": shot.duration_frames,
            "cfg": shot.cfg if hasattr(shot, 'cfg') else 3.5,
            "steps": shot.steps if hasattr(shot, 'steps') else 20,
        }, source_fps=source_fps)


        # Build workflow
        if hasattr(shot, 'skill') and shot.skill:
            logger.info(f"Using GOLDEN SKILL: {shot.skill}")
            workflow = self.skill_manager.load_skill(
                name=shot.skill,
                overrides={
                    **validated_inputs,
                    "image_name": first_image_name
                }
            )
        elif self.gemini_api_key:
            logger.info("Using HYBRID MODE: Gemini (Keyframes) + LTX 2.3 (Interpolation)")
            workflow = build_hybrid_flf2v(
                prompt=full_prompt,
                api_key=self.gemini_api_key,
                negative=negative,
                width=shot.width,
                height=shot.height,
                length=shot.duration_frames,
                seed=shot.seed,
                filename_prefix=filename_prefix,
            )
        elif mode == "flf2v":
            workflow = build_flf2v(
                prompt=full_prompt,
                first_image_name=first_image_name,
                last_image_name=last_image_name,
                negative=negative,
                width=shot.width,
                height=shot.height,
                length=shot.duration_frames,
                seed=shot.seed,
                filename_prefix=filename_prefix,
            )
        elif mode == "i2v":
            workflow = build_image_to_video(
                prompt=full_prompt,
                image_name=first_image_name,
                negative=negative,
                width=shot.width,
                height=shot.height,
                length=shot.duration_frames,
                seed=shot.seed,
                filename_prefix=filename_prefix,
            )
        else:  # t2v
            workflow = build_text_to_video(
                prompt=full_prompt,
                negative=negative,
                width=shot.width,
                height=shot.height,
                length=shot.duration_frames,
                seed=shot.seed,
                filename_prefix=filename_prefix,
            )
        # Apply Guardrail 3: Workflow Validation (Critic Node)
        if not Critic.validate_workflow(workflow):
            raise ComfyUIError("Critic rejected the proposed workflow architecture.")

        # Execute workflow
        start_time = time.time()
        try:
            outputs = self.client.execute_and_wait(
                workflow, on_progress=on_comfyui_progress
            )
        except ComfyUIError as e:
            logger.error(f"Shot {shot.shot_id} failed: {e}")
            raise

        generation_time = time.time() - start_time

        # Download outputs
        downloaded = self.client.download_all_outputs(
            outputs, str(self.clips_dir)
        )

        # Find the video and frame outputs
        video_path = ""
        last_frame_extracted = ""

        for node_id, files in downloaded.items():
            for f in files:
                if f.endswith(".webp"):
                    video_path = f
                elif f.endswith(".png") and "_frames" in f:
                    # The SaveImage node saves all frames; we want the last
                    pass

        # Extract last frame for chaining
        if video_path:
            try:
                last_frame_extracted = extract_last_frame_from_webp(
                    video_path,
                    str(self.frames_dir / f"shot_{shot.shot_id:03d}_last.png"),
                )
            except Exception as e:
                logger.warning(f"Could not extract last frame: {e}")
                # Fallback: try finding the last saved frame from SaveImage
                last_frame_extracted = self._find_last_saved_frame(
                    downloaded, shot.shot_id
                )

        result = ClipResult(
            shot_id=shot.shot_id,
            video_path=video_path,
            last_frame_path=last_frame_extracted,
            first_frame_path=first_frame or "",
            prompt_used=full_prompt,
            seed_used=workflow.get("50", {}).get("inputs", {}).get("seed", -1),
            width=shot.width,
            height=shot.height,
            frame_count=shot.duration_frames,
            generation_time=generation_time,
        )

        # Save per-clip metadata
        meta_path = self.clips_dir / f"shot_{shot.shot_id:03d}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        return result

    def _find_last_saved_frame(self, downloaded: dict,
                                shot_id: int) -> str:
        """Find the last PNG frame saved by the SaveImage node."""
        all_pngs = []
        for files in downloaded.values():
            for f in files:
                if f.endswith(".png"):
                    all_pngs.append(f)
        if all_pngs:
            all_pngs.sort()
            return all_pngs[-1]
        return ""

    def _save_sequence_metadata(self, storyboard: StoryBoard,
                                 results: list[ClipResult]):
        """Save complete sequence metadata for reproducibility."""
        meta = {
            "title": storyboard.title,
            "fps": storyboard.fps,
            "style_prompt": storyboard.style_prompt,
            "negative_prompt": storyboard.negative_prompt,
            "total_clips": len(results),
            "total_generation_time": sum(r.generation_time for r in results),
            "clips": [r.to_dict() for r in results],
        }
        meta_path = self.output_dir / "sequence_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Saved sequence metadata → {meta_path}")
