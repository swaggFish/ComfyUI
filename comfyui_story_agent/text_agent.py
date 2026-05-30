import json
import logging
import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

from .drift_schema import RecommendedAction, ShotCorrectionPlan, ShotEvaluation, DriftSeverity
from .director import compute_corrections, extract_evidence_frames

logger = logging.getLogger(__name__)

class TextIssue(BaseModel):
    """A single detected text error in a frame."""
    frame_location: str = Field(description="Which frame the issue appears in: 'first', 'middle', or 'last'")
    expected_text: str = Field(description="The text that was expected.")
    actual_transcription: str = Field(description="The gibberish or text that was actually transcribed.")
    description: str = Field(description="Description of the issue (e.g. 'Text morphed into illegible symbols').")

class TextEvaluation(BaseModel):
    """The TextDirector's complete evaluation of a generated shot."""
    passed: bool = Field(description="True if the text is perfectly legible and matches expectations.")
    confidence: float = Field(default=1.0, description="Confidence in the transcription.")
    issues: list[TextIssue] = Field(default_factory=list)
    recommended_action: RecommendedAction = Field(default=RecommendedAction.none)
    negative_prompt_addition: Optional[str] = Field(default=None)
    summary: str = Field(default="")

_TEXT_DIRECTOR_SYSTEM_PROMPT = """You are a quality control OCR Director for an AI video production pipeline.
Your job is to read the text in the provided video frames (first, middle, last) and verify if it is highly legible and matches the expected text.

AI video models often scramble, morph, or warp text over time.
If the text is gibberish, morphing, unreadable, missing, or changes between frames, the shot FAILS.
If the text matches the expected text clearly and remains stable across the frames, the shot PASSES.

Return a structured JSON response with:
- passed: true/false
- confidence: 0.0-1.0
- issues: list of frame-level transcription issues
- recommended_action: one of 'reseed', 'scrap_and_rethink', 'increase_first_strength', 'decrease_cfg', or 'none'
- negative_prompt_addition: optional text to add to the negative prompt
- summary: concise explanation of the decision

If the shot fails due to illegible or incorrect text, prefer 'scrap_and_rethink'.
If the shot fails because the text is slightly distorted but still mostly legible, prefer 'reseed'.
Do NOT suggest any creative prompt rewrites; only choose one mathematical or stability-oriented correction action.
"""

_TEXT_EVALUATION_PROMPT = """Review these frames from a generated video clip.

Expected Text to find on screen: "{expected_text}"

Please transcribe all text visible in the frames and compare it to the expected text.
If the text is mangled, missing, or morphs into gibberish, set passed=false and describe the issue.
Return only valid JSON matching the TextEvaluation schema.
"""

def evaluate_text_with_gemini(
    evidence_frames: dict[str, str],
    expected_text: str,
    api_key: str,
    model: str = "gemini-2.5-flash",
) -> TextEvaluation:
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError("google-genai is required.")

    client = genai.Client(
        vertexai=True,
        project="project-752e56a3-dc98-4377-b37",
        location="us-central1"
    )

    config = types.GenerateContentConfig(
        system_instruction=_TEXT_DIRECTOR_SYSTEM_PROMPT,
        temperature=0.1,
        response_mime_type="application/json",
        response_schema=TextEvaluation,
    )

    parts = [_TEXT_EVALUATION_PROMPT.format(expected_text=expected_text)]

    for label in ["first", "middle", "last"]:
        frame_path = evidence_frames.get(label)
        if frame_path and Path(frame_path).exists():
            img_data = Path(frame_path).read_bytes()
            parts.append(types.Part.from_bytes(data=img_data, mime_type="image/png"))
            parts.append(f"[Frame: {label.upper()}]")

    try:
        response = client.models.generate_content(
            model=model,
            contents=parts,
            config=config,
        )
        if response.text:
            raw = json.loads(response.text)
            evaluation = TextEvaluation(**raw)
        else:
            evaluation = TextEvaluation(passed=True, summary="Auto-passed (empty response).")

        if not evaluation.passed and evaluation.recommended_action == RecommendedAction.none:
            evaluation.recommended_action = RecommendedAction.scrap_and_rethink
            evaluation.summary = (
                evaluation.summary
                or "Text failed with no recommended action; defaulting to scrap_and_rethink."
            )
    except Exception as e:
        logger.error(f"Text evaluation failed: {e}")
        evaluation = TextEvaluation(passed=True, summary=f"Error: {e}")
    return evaluation

class TextDirector:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        max_retries: int = 3,
        evidence_dir: str = "/tmp/text_director_evidence",
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model = model
        self.max_retries = max_retries
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[ShotCorrectionPlan] = []

    @property
    def enabled(self) -> bool:
        return True

    def review_clip(
        self,
        clip_path: str,
        expected_text: str,
        shot_id: int = 0,
    ) -> TextEvaluation:
        if not self.enabled:
            return TextEvaluation(passed=True, summary="TextDirector disabled; auto-passed.")

        logger.info(f"📝 TextDirector reviewing Shot {shot_id} for text: '{expected_text}'")
        evidence = extract_evidence_frames(clip_path, str(self.evidence_dir), shot_id)
        return evaluate_text_with_gemini(evidence, expected_text, self.api_key, self.model)

    def get_corrections(self, evaluation: TextEvaluation, current_params: dict, attempt: int) -> dict:
        mock_eval = ShotEvaluation(
            passed=evaluation.passed,
            drift_severity=DriftSeverity.high if not evaluation.passed else DriftSeverity.none,
            recommended_action=evaluation.recommended_action,
            negative_prompt_addition=evaluation.negative_prompt_addition,
            summary=evaluation.summary
        )
        
        corrected = compute_corrections(mock_eval, current_params, attempt)
        
        self.history.append(ShotCorrectionPlan(
            shot_id=current_params.get("shot_id", 0),
            attempt=attempt,
            evaluation=mock_eval,
            original_params=current_params,
            corrected_params=corrected,
            correction_notes=f"Text Action: {evaluation.recommended_action.value}",
        ))
        
        return corrected

    def should_retry(self, evaluation: TextEvaluation, attempt: int) -> bool:
        if evaluation.passed:
            return False
        if attempt >= self.max_retries:
            logger.warning(f"TextDirector: Max retries ({self.max_retries}) reached. Accepting current text despite mangling.")
            return False
        return True
