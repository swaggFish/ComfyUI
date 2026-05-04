#!/usr/bin/env python3
"""
Export Golden Skills — Production-Grade ComfyUI API Workflows

Exports the exact proven workflows as versioned JSON files that can be:
  1. Loaded back by the SkillManager for reproducible generation
  2. Imported into ComfyUI GUI via "Load (API Format)"
  3. Shared with other users as immutable templates

The agent fills in the blanks of these pre-approved templates;
it never builds the workflow graph from scratch.

Usage:
    cd /home/mikeyb/Documents/AI/ComfyUI
    ./venv/bin/python3 -m comfyui_story_agent.export_golden_skills
"""

import json
import os
from datetime import datetime, timezone
from comfyui_story_agent import workflow_templates

SKILLS_DIR = "/home/mikeyb/Documents/AI/ComfyUI/comfyui_story_agent/skills"
os.makedirs(SKILLS_DIR, exist_ok=True)

# ── Production Settings (from production_envelopes.json) ─────────────
PROD_WIDTH = 1024
PROD_HEIGHT = 576
PROD_FRAMES = 97      # 8*12+1 (max safe for 16GB VRAM)
PROD_FPS = 25
PROD_SEED = 12345      # Placeholder — agent overrides at runtime
PROD_CFG = 3.5
PROD_STEPS = 24

_VERSION = "2.0.0"
_TIMESTAMP = datetime.now(timezone.utc).isoformat()


def _add_meta(workflow: dict, name: str, description: str) -> dict:
    """Add production metadata to a workflow for traceability."""
    workflow["_meta"] = {
        "name": name,
        "version": _VERSION,
        "exported_at": _TIMESTAMP,
        "description": description,
        "envelope": "flf2v_cinematic",
        "resolution": f"{PROD_WIDTH}x{PROD_HEIGHT}",
        "frames": PROD_FRAMES,
        "fps": PROD_FPS,
        "notes": "Golden template — DO NOT modify node wiring. Only fill in placeholder values.",
    }
    return workflow


# ── 1. Golden T2V (Text to Video) ────────────────────────────────────
t2v = workflow_templates.build_text_to_video(
    prompt="GOLDEN_PROMPT_PLACEHOLDER",
    negative="GOLDEN_NEGATIVE_PLACEHOLDER",
    width=PROD_WIDTH,
    height=PROD_HEIGHT,
    length=PROD_FRAMES,
    fps=PROD_FPS,
    seed=PROD_SEED,
    cfg=PROD_CFG,
    steps=PROD_STEPS,
)
_add_meta(t2v, "ltx_golden_t2v", "Text-to-Video: 2-pass advanced upscale pipeline for LTX 2.3 22B")
with open(os.path.join(SKILLS_DIR, "ltx_golden_t2v_v2.json"), "w") as f:
    json.dump(t2v, f, indent=2)
print(f"✓ Exported ltx_golden_t2v_v2.json")


# ── 2. Golden I2V (Image to Video) ───────────────────────────────────
i2v = workflow_templates.build_image_to_video(
    prompt="GOLDEN_PROMPT_PLACEHOLDER",
    image_name="GOLDEN_IMAGE_PLACEHOLDER.png",
    negative="GOLDEN_NEGATIVE_PLACEHOLDER",
    width=PROD_WIDTH,
    height=PROD_HEIGHT,
    length=PROD_FRAMES,
    fps=PROD_FPS,
    seed=PROD_SEED,
    cfg=PROD_CFG,
    steps=PROD_STEPS,
)
_add_meta(i2v, "ltx_golden_i2v", "Image-to-Video: First-frame anchored 2-pass pipeline")
with open(os.path.join(SKILLS_DIR, "ltx_golden_i2v_v2.json"), "w") as f:
    json.dump(i2v, f, indent=2)
print(f"✓ Exported ltx_golden_i2v_v2.json")


# ── 3. Golden FLF2V (First/Last Frame to Video) — THE PROVEN ONE ─────
flf2v = workflow_templates.build_flf2v(
    prompt="GOLDEN_PROMPT_PLACEHOLDER",
    first_image_name="GOLDEN_FIRST_FRAME.png",
    last_image_name="GOLDEN_LAST_FRAME.png",
    negative="GOLDEN_NEGATIVE_PLACEHOLDER",
    width=PROD_WIDTH,
    height=PROD_HEIGHT,
    length=PROD_FRAMES,
    fps=PROD_FPS,
    seed=PROD_SEED,
    cfg=PROD_CFG,
    steps=PROD_STEPS,
    first_strength=1.0,
    last_strength=1.0,
)
_add_meta(
    flf2v, "ltx_golden_flf2v",
    "First/Last Frame to Video: Dual-anchor 2-pass pipeline. "
    "This is the PROVEN workflow used for Super Noah and Super Henrio Hybrid productions."
)
with open(os.path.join(SKILLS_DIR, "ltx_golden_flf2v_v2.json"), "w") as f:
    json.dump(flf2v, f, indent=2)
print(f"✓ Exported ltx_golden_flf2v_v2.json (THE PROVEN PIPELINE)")


print(f"\n✓ All golden skills exported to {SKILLS_DIR}")
print(f"  Version: {_VERSION}")
print(f"  Resolution: {PROD_WIDTH}x{PROD_HEIGHT}")
print(f"  Frames: {PROD_FRAMES}")
print(f"  Envelope: flf2v_cinematic")
