import json
import logging
import os
from typing import Any, Optional

from .shot_planner import Shot, StoryBoard

logger = logging.getLogger(__name__)


class WriterAgent:
    """A writing-focused companion for the marketing agent.

    Produces market opportunity research, storyboard drafts, and short-form ad copy
    based on the current ComfyUI marketing video production pipeline.
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-pro",
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.llm_client = llm_client or self._init_llm_client()

    def _init_llm_client(self) -> Optional[Any]:
        if not self.api_key:
            logger.warning("WriterAgent has no API key; falling back to heuristics.")
            return None
        try:
            from google import genai

            client = genai.Client(
                vertexai=True,
                project="project-752e56a3-dc98-4377-b37",
                location="us-central1",
            )
            return client
        except ImportError:
            logger.warning(
                "google-genai is not installed. WriterAgent will run fallback logic."
            )
            return None

    def _call_llm(self, system_instruction: str, prompt: str) -> dict:
        if not self.llm_client:
            raise RuntimeError("No LLM client available for WriterAgent.")

        from google.genai import types

        response = self.llm_client.models.generate_content(
            model=self.model,
            contents=[prompt],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.6,
                response_mime_type="application/json",
            ),
        )

        if not response.text:
            raise ValueError("LLM returned empty response")

        try:
            return json.loads(response.text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Failed to parse LLM JSON response: {exc}\n{response.text}"
            )

    def recommend_product_opportunities(
        self,
        category: str,
        platforms: Optional[list[str]] = None,
        num_ideas: int = 3,
    ) -> dict:
        """Recommend product categories, audiences, and revenue-oriented ad targets."""
        platforms = platforms or ["TikTok", "YouTube Shorts", "Instagram Reels"]
        if self.llm_client:
            system_instruction = (
                "You are a revenue-driven marketing strategist for short-form video ads. "
                "Recommend the highest-potential product categories, audiences, and "
                "monetization opportunities for social media traffic. Focus on current "
                "trends, CPM value, e-commerce/ad offer performance, and quick-win campaign ideas."
            )
            prompt = (
                f"Analyze the category '{category}' and the platforms {platforms}. "
                f"Return valid JSON with these keys: category, audience, platforms, "
                f"product_ideas (list of {{name, reason, traffic_edge, revenue_signal}}). "
                f"Give me {num_ideas} product ideas that are most likely to attract social media traffic "
                f"and generate paid conversions quickly."
            )
            try:
                return self._call_llm(system_instruction, prompt)
            except Exception as exc:
                logger.warning(
                    f"WriterAgent product research failed; using heuristic fallback: {exc}"
                )

        # Heuristic fallback if no LLM is available
        candidate_names = [
            f"{category.title()} Quick-Launch Bundle",
            f"{category.title()} Premium Experience",
            f"{category.title()} High-velocity Starter",
        ]
        return {
            "category": category,
            "audience": "early adopters, trend-sensitive buyers, impulse shoppers",
            "platforms": platforms,
            "product_ideas": [
                {
                    "name": candidate_names[i % len(candidate_names)],
                    "reason": (
                        "Strong visual identity and fast storytelling work well on short-form video. "
                        "Prioritize product demonstrations and before/after results."
                    ),
                    "traffic_edge": "High engagement from aspirational lifestyle seekers.",
                    "revenue_signal": (
                        "Products with simple purchase funnels and subscription add-ons "
                        "tend to convert faster on TikTok and YouTube Shorts."
                    ),
                }
                for i in range(num_ideas)
            ],
        }

    def draft_storyboard(
        self,
        product_concept: str,
        num_shots: int = 3,
        target_length_sec: int = 30,
        platforms: Optional[list[str]] = None,
    ) -> StoryBoard:
        """Generate a production-ready storyboard for a short-form ad."""
        platforms = platforms or ["TikTok", "YouTube Shorts", "Instagram Reels"]
        target_fps = self._choose_fps(target_length_sec)

        if self.llm_client:
            system_instruction = (
                "You are a writer and storyboard planner for cinematic social media ads. "
                "Create concise, sale-oriented shot descriptions that work for short-form video "
                "and support a 30s or 60s ad length when generated at low fps then interpolated."
            )
            prompt = (
                f"Create a {num_shots}-shot storyboard for this product concept: '{product_concept}'. "
                f"Target {target_length_sec} seconds of finished ad runtime on {platforms}. "
                "Return valid JSON with title, style_prompt, negative_prompt, fps, and shots. "
                "Each shot should include prompt, camera, duration_frames, transition_in, and expected_text if there is a CTA or headline."
            )
            try:
                raw = self._call_llm(system_instruction, prompt)
                shots = []
                valid_transitions = {
                    "crossfade",
                    "cut",
                    "dissolve",
                    "fade_black",
                    "fade_white",
                    "slide_left",
                    "slide_right",
                    "wipe_left",
                    "wipe_right",
                    "zoom_in",
                }
                shots = []
                for i, shot_data in enumerate(raw.get("shots", [])[:num_shots]):
                    duration_frames = int(shot_data.get("duration_frames", 33))
                    transition_in = shot_data.get("transition_in", "cut")
                    if transition_in not in valid_transitions:
                        transition_in = "cut"

                    shots.append(
                        Shot(
                            shot_id=i + 1,
                            prompt=shot_data.get("prompt", ""),
                            camera=shot_data.get("camera", "static"),
                            duration_frames=self._normalize_frames(duration_frames),
                            transition_in=transition_in,
                            expected_text=shot_data.get("expected_text"),
                            width=shot_data.get("width", 1280),
                            height=shot_data.get("height", 704),
                        )
                    )
                storyboard = StoryBoard(
                    title=raw.get("title", f"Marketing: {product_concept[:20]}"),
                    style_prompt=raw.get("style_prompt", "photorealistic, cinematic lighting, premium aesthetic"),
                    negative_prompt=raw.get("negative_prompt", "cartoon, illustration, low quality, blurry, noisy, watermark"),
                    shots=shots,
                    fps=self._choose_fps(target_length_sec) if raw.get("fps") is None else int(raw.get("fps", target_fps)),
                )
                if storyboard.shots:
                    return storyboard
            except Exception as exc:
                logger.warning(
                    f"WriterAgent storyboard generation failed; using fallback: {exc}"
                )

        shots = []
        base_frames = self._normalize_frames(
            int(round((target_length_sec * target_fps) / max(num_shots, 1)))
        )
        for i in range(num_shots):
            shots.append(
                Shot(
                    shot_id=i + 1,
                    prompt=(
                        f"Shot {i+1}: {product_concept} presented with strong visual drama, "
                        "clean product details, motion, and a persuasive CTA overlay."
                    ),
                    camera="dynamic" if i == 0 else "push in",
                    duration_frames=base_frames,
                    transition_in="cut" if i == 0 else "crossfade",
                    expected_text="Click now to learn more" if i == num_shots - 1 else None,
                    width=1280,
                    height=704,
                )
            )

        return StoryBoard(
            title=f"Marketing: {product_concept[:20]}",
            style_prompt=(
                "photorealistic, cinematic lighting, premium product photography, "
                "shallow depth of field, strong brand focus"
            ),
            negative_prompt=(
                "cartoon, illustration, low quality, blurry, noisy, watermark, text artifacts"
            ),
            shots=shots,
            fps=target_fps,
        )

    def write_ad_script(
        self,
        storyboard: StoryBoard,
        product_name: str,
        platform: str = "TikTok",
        tone: str = "urgent and cinematic",
    ) -> dict:
        """Draft a short-form ad script for the provided storyboard."""
        if self.llm_client:
            system_instruction = (
                "You are a creative copywriter for short-form advertising video. "
                "Write a story-driven script with a strong hook, product benefit narrative, "
                "and a clear CTA for the target platform."
            )
            prompt = (
                f"Create a short-form ad script for '{product_name}' on {platform}. "
                f"Use the following storyboard title: '{storyboard.title}'. "
                "Include a hook, per-shot voiceover or caption guidance, and a call-to-action. "
                "Return valid JSON with hook, script, cta, and tone."
            )
            try:
                raw = self._call_llm(system_instruction, prompt)
                return {
                    "hook": raw.get("hook", "Watch this now."),
                    "script": raw.get("script", raw.get("copy", "")),
                    "cta": raw.get("cta", "Learn more"),
                    "tone": raw.get("tone", tone),
                }
            except Exception as exc:
                logger.warning(
                    f"WriterAgent script drafting failed; falling back to simple copy: {exc}"
                )

        return {
            "hook": f"See why everyone is talking about {product_name}.",
            "script": (
                "Start with an emotional product moment, show the benefit clearly, "
                "and end with a crisp CTA."
            ),
            "cta": "Tap to shop now.",
            "tone": tone,
        }

    def _choose_fps(self, target_length_sec: int) -> int:
        if target_length_sec >= 60:
            return 6
        if target_length_sec >= 30:
            return 8
        if target_length_sec >= 15:
            return 12
        return 24

    def _normalize_frames(self, frames: int) -> int:
        if frames < 33:
            return 33
        return 8 * ((frames - 1) // 8) + 1
