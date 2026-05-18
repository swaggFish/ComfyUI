"""
Skill Manager — Plugin system for external ComfyUI workflows

Allows the agent to load community-created workflows (JSON) from 
Pixelle MCP or a local skills/ directory and use them as 
generation engines for specific shots.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class SkillManager:
    """Manages loading and mapping of external ComfyUI workflow 'skills'."""

    def __init__(self, workspace_root: str):
        self.workspace_root = Path(workspace_root)
        self.skills_dirs = [
            self.workspace_root / "comfyui_story_agent" / "skills",
            self.workspace_root / "pixelle-mcp" / "workflows",
        ]
        
        # Ensure local skills dir exists
        self.skills_dirs[0].mkdir(parents=True, exist_ok=True)
        self.skills = self._index_skills()

    def _index_skills(self) -> Dict[str, Path]:
        """Index all available .json workflows in the skills directories."""
        index = {}
        for d in self.skills_dirs:
            if d.exists():
                for f in d.glob("*.json"):
                    # Use filename without extension as skill name
                    index[f.stem] = f
        logger.info(f"Indexed {len(index)} skills from {self.skills_dirs}")
        return index

    def get_skill_path(self, name: str) -> Optional[Path]:
        """Get the absolute path to a skill JSON by name."""
        return self.skills.get(name)

    def load_skill(self, name: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load a skill workflow and map standard inputs to it.
        
        Standard Input Keys to Map:
        - prompt
        - negative
        - seed
        - width
        - height
        - image (for I2V/Hybrid)
        """
        path = self.get_skill_path(name)
        if not path:
            raise ValueError(f"Skill '{name}' not found. Available: {list(self.skills.keys())}")

        with open(path, "r") as f:
            workflow = json.load(f)

        return self.apply_mappings(workflow, overrides)

    def apply_mappings(self, workflow: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Intelligently find nodes in the workflow to inject inputs.
        
        This mimics the 'OpenClaw' mapping logic:
        1. Find CLIPTextEncode nodes for prompt/negative.
        2. Find CheckpointLoader/Sampler nodes for seed/size.
        3. Find LoadImage nodes for input images.
        """
        # Mapping strategies
        for node_id, node in workflow.items():
            class_type = node.get("class_type", "")
            node_inputs = node.get("inputs", {})

            # 1. Prompt Mapping
            if class_type in ["CLIPTextEncode", "GemmaTextEncoder", "LTXAVTextEncoderLoader"]:
                if "text" in node_inputs and inputs.get("prompt"):
                    # Heuristic: if it looks like a positive prompt node
                    text_val = node_inputs.get("text", "")
                    text_str = text_val if isinstance(text_val, str) else ""
                    if "negative" not in node_id.lower() and "bad" not in text_str.lower():
                        node_inputs["text"] = inputs["prompt"]
                if "text" in node_inputs and inputs.get("negative"):
                    text_val = node_inputs.get("text", "")
                    text_str = text_val if isinstance(text_val, str) else ""
                    if "negative" in node_id.lower() or "bad" in text_str.lower():
                        node_inputs["text"] = inputs["negative"]

            # 2. Seed Mapping
            if "seed" in node_inputs and inputs.get("seed") is not None:
                if inputs["seed"] != -1:
                    node_inputs["seed"] = inputs["seed"]

            # 3. Latent Size Mapping
            if class_type in ["EmptyLatentImage", "EmptyLTXVLatentVideo"]:
                if inputs.get("width"): node_inputs["width"] = inputs["width"]
                if inputs.get("height"): node_inputs["height"] = inputs["height"]
                if inputs.get("length") and "length" in node_inputs: node_inputs["length"] = inputs["length"]

            # 4. Image Input Mapping
            if class_type == "LoadImage":
                if inputs.get("image_name") and "image" in node_inputs:
                    node_inputs["image"] = inputs["image_name"]
            
            if class_type == "LTXVPreprocess" and not isinstance(node_inputs.get("image"), list):
                # Only map if it's not already connected to another node
                if inputs.get("image_name") and "image" in node_inputs:
                    node_inputs["image"] = inputs["image_name"]

        return workflow
