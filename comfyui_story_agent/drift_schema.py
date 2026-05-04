"""
Drift Detection Schema — Pydantic Models for Structured Director Output

These models define the exact JSON contract between the Gemini Flash VLM
and the OpenClaw correction loop. Gemini is forced to return this schema,
eliminating parsing failures from hallucinated markdown or freeform text.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class DriftSeverity(str, Enum):
    """How bad is the drift?"""
    none = "none"
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class RecommendedAction(str, Enum):
    """What correction should the agent apply?"""
    none = "none"
    decrease_cfg = "decrease_cfg"
    increase_cfg = "increase_cfg"
    decrease_steps = "decrease_steps"
    increase_steps = "increase_steps"
    adjust_negative = "adjust_negative"
    decrease_frame_count = "decrease_frame_count"
    increase_first_strength = "increase_first_strength"
    increase_last_strength = "increase_last_strength"
    reseed = "reseed"
    scrap_and_rethink = "scrap_and_rethink"


class DriftIssue(BaseModel):
    """A single detected drift issue in a frame."""
    frame_location: str = Field(
        description="Which frame the issue appears in: 'first', 'middle', or 'last'"
    )
    category: str = Field(
        description="Category: 'anatomy', 'background', 'physics', 'style', 'character_identity', 'lighting', 'object_persistence'"
    )
    description: str = Field(
        description="Specific description of the issue, e.g. 'Subject's left arm merges with desk'"
    )


class ShotEvaluation(BaseModel):
    """The Director's complete evaluation of a generated shot.

    This is the structured output Gemini Flash is forced to return.
    The correction loop reads this to decide: pass, retry, or scrap.
    """
    passed: bool = Field(
        description="True if the shot is production-ready, False if it needs correction"
    )
    drift_severity: DriftSeverity = Field(
        default=DriftSeverity.none,
        description="Overall severity of detected drift"
    )
    character_consistency: float = Field(
        default=1.0,
        ge=0.0, le=1.0,
        description="Score 0.0-1.0: how consistent is the main character across frames"
    )
    background_stability: float = Field(
        default=1.0,
        ge=0.0, le=1.0,
        description="Score 0.0-1.0: how stable is the background/environment"
    )
    physics_coherence: float = Field(
        default=1.0,
        ge=0.0, le=1.0,
        description="Score 0.0-1.0: do objects follow physical rules (gravity, perspective)"
    )
    style_consistency: float = Field(
        default=1.0,
        ge=0.0, le=1.0,
        description="Score 0.0-1.0: does the art style remain consistent"
    )
    issues: list[DriftIssue] = Field(
        default_factory=list,
        description="List of specific drift issues detected"
    )
    recommended_action: RecommendedAction = Field(
        default=RecommendedAction.none,
        description="The primary correction action to apply"
    )
    negative_prompt_addition: Optional[str] = Field(
        default=None,
        description="If action is 'adjust_negative', what text to add to the negative prompt"
    )
    summary: str = Field(
        default="",
        description="One-sentence summary of the evaluation"
    )


class ShotCorrectionPlan(BaseModel):
    """The correction plan that the engine applies to retry a failed shot."""
    shot_id: int
    attempt: int
    evaluation: ShotEvaluation
    original_params: dict = Field(default_factory=dict)
    corrected_params: dict = Field(default_factory=dict)
    correction_notes: str = ""
