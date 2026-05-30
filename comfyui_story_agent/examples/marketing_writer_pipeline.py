import json
import logging
import os
from pathlib import Path

from comfyui_story_agent.comfyui_client import ComfyUIClient
from comfyui_story_agent.marketing_agent import MarketingVideoAgent


def load_gemini_key_from_env_file() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    env_path = repo_root / "pixelle-mcp" / ".env"
    if not env_path.exists():
        return ""

    with env_path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("GEMINI_API_KEY="):
                _, value = stripped.split("=", 1)
                return value.strip().strip('"').strip("'").strip()
    return ""

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    output_dir = Path(__file__).resolve().parent / "marketing_writer_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    comfy_host = os.environ.get("COMFYUI_HOST", "127.0.0.1")
    comfy_port = int(os.environ.get("COMFYUI_PORT", 8188))
    gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_api_key:
        gemini_api_key = load_gemini_key_from_env_file()
        if gemini_api_key:
            os.environ["GEMINI_API_KEY"] = gemini_api_key

    client = ComfyUIClient(host=comfy_host, port=comfy_port)
    agent = MarketingVideoAgent(
        client=client,
        output_dir=str(output_dir),
        gemini_api_key=gemini_api_key,
    )

    category = "health and wellness gadgets"
    logging.info("🔎 Running product opportunity research...")
    research = agent.recommend_product_opportunities(
        category=category,
        platforms=["TikTok", "Instagram Reels", "YouTube Shorts"],
        num_ideas=3,
    )
    research_path = output_dir / "product_research.json"
    with research_path.open("w", encoding="utf-8") as f:
        import json

        json.dump(research, f, indent=2)
    logging.info(f"Saved product research → {research_path}")

    product_concept = (
        "A premium sleep optimization device that combines smart temperature control, "
        "gentle motion cues, and a sleek lifestyle design for people who want better rest."
    )
    logging.info("🎬 Drafting a short-form ad storyboard...")
    storyboard = agent.conceptualize_video(
        product_concept=product_concept,
        num_shots=4,
        target_length_sec=30,
        platforms=["TikTok", "YouTube Shorts"],
    )

    storyboard_path = output_dir / "marketing_storyboard.yaml"
    try:
        storyboard.to_yaml(str(storyboard_path))
        logging.info(f"Saved storyboard → {storyboard_path}")
    except ImportError:
        storyboard_path = output_dir / "marketing_storyboard.json"
        with storyboard_path.open("w", encoding="utf-8") as f:
            json.dump({
                "title": storyboard.title,
                "style": storyboard.style,
                "fps": storyboard.fps,
                "style_prompt": storyboard.style_prompt,
                "negative_prompt": storyboard.negative_prompt,
                "shots": [shot.to_dict() for shot in storyboard.shots],
            }, f, indent=2)
        logging.info(f"PyYAML not installed; saved storyboard JSON → {storyboard_path}")

    logging.info("✍️  Generating a short-form ad script...")
    script = agent.create_ad_script(
        storyboard=storyboard,
        product_name="RestFlow Sleep Pod",
        platform="TikTok",
        tone="urgent and cinematic",
    )
    script_path = output_dir / "ad_script.json"
    with script_path.open("w", encoding="utf-8") as f:
        json.dump(script, f, indent=2)
    logging.info(f"Saved ad script → {script_path}")

    logging.info("✅ Demo complete. Review the generated product research, storyboard, and ad script.")


if __name__ == "__main__":
    main()
