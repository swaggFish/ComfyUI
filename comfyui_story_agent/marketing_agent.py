import os
import json
import logging
from pathlib import Path
from typing import Optional

from .shot_planner import StoryBoard, Shot
from .sequence_engine import SequenceEngine
from .comfyui_client import ComfyUIClient

logger = logging.getLogger(__name__)

class MarketingVideoAgent:
    """Agent for generating realistic marketing videos from product concepts.
    
    Maintains continuity and auto-correction (via Director) while prioritizing
    high-end realism, professional cinematography, and marketing value.
    """

    def __init__(self, client: ComfyUIClient, output_dir: str, gemini_api_key: Optional[str] = None):
        self.client = client
        self.output_dir = output_dir
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
        
        # We reuse the robust SequenceEngine which already integrates the Director
        # for continuity and automated quality assurance/auto-correction.
        self.engine = SequenceEngine(
            client=self.client, 
            output_dir=self.output_dir, 
            gemini_api_key=self.gemini_api_key,
            enable_director=True,
            director_max_retries=3
        )
        
        # Use Gemini to expand the concept into a structured storyboard
        if self.gemini_api_key:
            try:
                from google import genai
                from google.genai import types
                # Force client to route to Vertex AI as per Director's established pattern
                self.llm_client = genai.Client(
                    vertexai=True,
                    project="project-752e56a3-dc98-4377-b37",
                    location="us-central1"
                )
            except ImportError:
                logger.warning("google-genai not found. Falling back to template generation.")
                self.llm_client = None
        else:
            self.llm_client = None

    def conceptualize_video(self, product_concept: str, num_shots: int = 3) -> StoryBoard:
        """Takes a concept and creates a storyboard detailing functionality and value."""
        if not self.llm_client:
            logger.warning("No LLM client available. Falling back to a template marketing storyboard.")
            return self._fallback_storyboard(product_concept, num_shots)
            
        system_instruction = (
            "You are an expert Social Media Marketer and Professional Videographer. "
            "Your task is to take a product concept and design a highly engaging, realistic, "
            "cinematic marketing video storyboard. Emphasize product functionality and value. "
            "Structure the pacing to include a strong hook in the first shot. "
            "Use terms related to professional videography (lighting, composition, depth of field)."
        )
        
        prompt = (
            f"Create a {num_shots}-shot video storyboard for the following product concept: '{product_concept}'.\n"
            "Output valid JSON matching this exact structure: \n"
            '{\n'
            '  "title": "Video Title",\n'
            '  "shots": [\n'
            '    {\n'
            '      "prompt": "highly detailed visual description of the shot",\n'
            '      "camera": "pan right",\n'
            '      "duration_frames": 33\n'
            '    }\n'
            '  ]\n'
            '}'
        )
        
        try:
            from google.genai import types
            response = self.llm_client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[prompt],
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.7,
                    response_mime_type="application/json"
                )
            )
            
            # Parse the structured response
            if response.text:
                data = json.loads(response.text)
            else:
                raise ValueError("LLM returned empty response")
            
            shots = []
            for i, s in enumerate(data.get("shots", [])):
                shots.append(Shot(
                    shot_id=i + 1,
                    prompt=s.get("prompt", ""),
                    camera=s.get("camera", "static"),
                    duration_frames=s.get("duration_frames", 33),
                    width=1280,  # Ensure HD quality for marketing videos
                    height=704
                ))
                
            # Define specific style/negative prompts for marketing
            style_prompt = (
                "photorealistic, 8k resolution, cinematic lighting, highly detailed, "
                "professional cinematography, depth of field, product photography, "
                "color graded, sharp focus, premium aesthetic"
            )
            negative_prompt = (
                "cartoon, animated, illustration, low quality, deformed, blurry, "
                "noisy, ugly, amateur, bad lighting, text, watermark"
            )
            
            return StoryBoard(
                title=data.get("title", f"Marketing: {product_concept[:15]}"),
                fps=24,
                style_prompt=style_prompt,
                negative_prompt=negative_prompt,
                shots=shots
            )
        except Exception as e:
            logger.error(f"Failed to generate storyboard via LLM: {e}")
            return self._fallback_storyboard(product_concept, num_shots)

    def _fallback_storyboard(self, concept: str, num_shots: int) -> StoryBoard:
        """Provides a hardcoded storyboard if LLM generation fails."""
        shots = [
            Shot(
                shot_id=1, 
                prompt=f"Cinematic wide establishing shot of {concept}, dynamic lighting, catching the viewer's attention immediately.", 
                camera="pan right", 
                duration_frames=49, 
                width=1280, 
                height=704
            ),
            Shot(
                shot_id=2, 
                prompt=f"Extreme close up of {concept} detailing its high-quality material and core functionality, shallow depth of field.", 
                camera="zoom in", 
                duration_frames=33, 
                width=1280, 
                height=704
            ),
            Shot(
                shot_id=3, 
                prompt=f"Dynamic shot of {concept} in action displaying its value to the user, bright, professional and engaging.", 
                camera="static", 
                duration_frames=49, 
                width=1280, 
                height=704
            ),
        ]
        
        return StoryBoard(
            title=f"Marketing: {concept[:20]}",
            fps=24,
            style_prompt="photorealistic, 8k resolution, cinematic lighting, highly detailed, professional cinematography",
            negative_prompt="cartoon, animated, illustration, low quality, deformed, blurry",
            shots=shots[:num_shots]
        )

    def produce(self, product_concept: str, num_shots: int = 3, on_progress=None):
        """Orchestrates the entire creation process from concept to final sequence."""
        logger.info(f"Conceptualizing marketing video for: {product_concept}")
        storyboard = self.conceptualize_video(product_concept, num_shots)
        
        logger.info(f"Starting production for '{storyboard.title}' with {len(storyboard.shots)} shots.")
        # Uses SequenceEngine which utilizes Director for visual QA and continuity tracking
        results = self.engine.generate_sequence(storyboard, on_progress=on_progress)
        return results

    def save_storyboard(self, storyboard: StoryBoard, filename: str):
        """Helper to save the generated storyboard before producing."""
        filepath = Path(self.output_dir) / filename
        storyboard.to_yaml(str(filepath))
        logger.info(f"Storyboard saved to {filepath}")
