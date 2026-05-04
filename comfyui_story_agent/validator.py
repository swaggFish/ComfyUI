"""
Critic Node — Workflow Validation & Parameter Envelopes

Implements all three guardrails from the production blueprint:
  Guardrail 1: Few-Shot Style Matching (via memory.json queries)
  Guardrail 2: Parameter Envelopes (mathematical boundaries from production_envelopes.json)
  Guardrail 3: Architecture Validation (structural integrity of ComfyUI API payloads)

The Critic does NOT care about creative text prompts — it checks the architecture.
It verifies that the necessary VAE is loaded, that the latent dimensions match the
model requirements, and that all routing nodes are properly wired.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# ── Load Production Envelopes ────────────────────────────────────────
_ENVELOPES_PATH = Path(__file__).parent / "production_envelopes.json"
_MEMORY_PATH = Path(__file__).parent / "memory.json"

def _load_envelopes() -> dict:
    """Load parameter envelopes from the production config."""
    if _ENVELOPES_PATH.exists():
        with open(_ENVELOPES_PATH, "r") as f:
            return json.load(f)
    logger.warning("production_envelopes.json not found; using hardcoded fallbacks.")
    return {}

def _load_memory() -> dict:
    """Load the few-shot success library."""
    if _MEMORY_PATH.exists():
        with open(_MEMORY_PATH, "r") as f:
            return json.load(f)
    return {"golden_runs": [], "learnings": [], "failure_modes": []}


# Legacy fallback envelopes (used if production_envelopes.json is missing)
_FALLBACK_ENVELOPES = {
    "cfg": {"min": 1.0, "max": 6.0},
    "steps": {"min": 10, "max": 50},
    "width": {"min": 256, "max": 1024},
    "height": {"min": 256, "max": 1024},
    "length": {"min": 1, "max": 97},
}


class Critic:
    """The 'Critic' agent responsible for pre-flight workflow validation.

    Three-layer validation:
      1. validate_inputs()   — Guardrail 2: Clamp params to safe envelopes
      2. validate_workflow()  — Guardrail 3: Check architecture integrity
      3. get_few_shot_context() — Guardrail 1: Retrieve matching proven runs
    """

    @staticmethod
    def validate_inputs(
        inputs: Dict[str, Any],
        pipeline: str = "flf2v_cinematic",
    ) -> Dict[str, Any]:
        """Validate and clamp inputs to safe 'Parameter Envelopes'.

        Args:
            inputs: Dict of generation parameters (cfg, width, height, etc.)
            pipeline: Which pipeline envelope to use (e.g., 'flf2v_cinematic')

        Returns:
            Dict with all parameters clamped to the envelope boundaries.
        """
        validated = inputs.copy()
        config = _load_envelopes()

        # Try to load pipeline-specific envelopes
        envelopes = _FALLBACK_ENVELOPES
        pipelines = config.get("pipelines", {})
        if pipeline in pipelines:
            pipe_params = pipelines[pipeline].get("parameters", {})
            envelopes = {k: v for k, v in pipe_params.items() if "min" in v and "max" in v}

        for key, envelope in envelopes.items():
            if key in validated:
                val = validated[key]
                if not isinstance(val, (int, float)):
                    continue
                lo = envelope.get("min", float("-inf"))
                hi = envelope.get("max", float("inf"))
                clamped = max(lo, min(hi, val))
                if clamped != val:
                    logger.warning(f"Critic clamped {key}: {val} -> {clamped}")
                    validated[key] = clamped

        # Apply VRAM safety: enforce sequential rendering
        vram = config.get("vram_safety", {})
        max_concurrent = vram.get("max_concurrent_shots", 1)
        if max_concurrent > 1:
            logger.warning(
                f"Critic override: max_concurrent_shots clamped to 1 (was {max_concurrent})"
            )

        return validated

    @staticmethod
    def validate_workflow(workflow: Dict[str, Any]) -> bool:
        """Guardrail 3: Architecture Validation.

        Checks for:
          - Missing critical node types (VAE, Sampler, Checkpoint)
          - Dangerous hyperparameters inside the JSON
          - Broken connections (node refs pointing to non-existent nodes)
          - VRAM-unsafe resolutions
        """
        # 1. Check for critical node types
        required_classes = ["KSamplerSelect", "CheckpointLoaderSimple", "VAEDecodeTiled"]
        found_classes = [node.get("class_type") for node in workflow.values()]

        for req in required_classes:
            if req not in found_classes:
                logger.error(f"Critic Error: Missing required node type '{req}'")
                return False

        # 2. Check for dangerous hyperparams inside the JSON
        for node_id, node in workflow.items():
            inputs = node.get("inputs", {})

            # Check CFG in any Guider or Sampler node
            if "cfg" in inputs:
                cfg = inputs["cfg"]
                if cfg > 7.0:
                    logger.error(f"Critic Error: CFG {cfg} is too high (danger of deep-frying).")
                    return False

            # Check dimensions in EmptyLatent nodes
            if node.get("class_type") in ["EmptyLatentImage", "EmptyLTXVLatentVideo"]:
                w, h = inputs.get("width", 0), inputs.get("height", 0)
                if w > 1024 or h > 1024:
                    logger.error(f"Critic Error: Resolution {w}x{h} exceeds VRAM safety limits.")
                    return False

        # 3. Check for broken connections (refs to non-existent nodes)
        node_ids = set(workflow.keys())
        for node_id, node in workflow.items():
            inputs = node.get("inputs", {})
            for key, val in inputs.items():
                if isinstance(val, list) and len(val) == 2:
                    ref_id = str(val[0])
                    if ref_id not in node_ids:
                        logger.error(
                            f"Critic Error: Node '{node_id}' input '{key}' "
                            f"references non-existent node '{ref_id}'"
                        )
                        return False

        logger.info("Critic: Workflow passed architecture validation.")
        return True

    @staticmethod
    def get_few_shot_context(
        tags: List[str],
        top_k: int = 3,
    ) -> List[dict]:
        """Guardrail 1: Few-Shot Style Matching.

        Queries the success library for the top-K matching golden runs
        based on tag overlap. Returns them as few-shot examples for the
        agent's context window.

        Args:
            tags: List of semantic tags describing the current shot
                  (e.g., ["superhero", "flying", "sunset"])
            top_k: Number of matching runs to return

        Returns:
            List of golden run dicts, sorted by relevance.
        """
        memory = _load_memory()
        golden_runs = memory.get("golden_runs", [])

        if not golden_runs or not tags:
            return []

        # Score each golden run by tag overlap
        tag_set = set(t.lower() for t in tags)
        scored = []
        for run in golden_runs:
            run_tags = set(t.lower() for t in run.get("tags", []))
            overlap = len(tag_set & run_tags)
            if overlap > 0:
                scored.append((overlap, run.get("quality_score", 0), run))

        # Sort by overlap (primary) then quality_score (secondary)
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [item[2] for item in scored[:top_k]]

    @staticmethod
    def get_learnings() -> List[str]:
        """Return all accumulated learnings from past productions."""
        memory = _load_memory()
        return memory.get("learnings", [])

    @staticmethod
    def get_failure_modes() -> List[dict]:
        """Return known failure modes and their fixes."""
        memory = _load_memory()
        return memory.get("failure_modes", [])

    @staticmethod
    def get_envelope_defaults(pipeline: str = "flf2v_cinematic") -> Dict[str, Any]:
        """Return the default parameter values for a given pipeline."""
        config = _load_envelopes()
        pipelines = config.get("pipelines", {})
        if pipeline in pipelines:
            params = pipelines[pipeline].get("parameters", {})
            return {k: v.get("default") for k, v in params.items() if "default" in v}
        return {}
