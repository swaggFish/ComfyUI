import json
import logging
import os
from pathlib import Path

from comfyui_story_agent.comfyui_client import ComfyUIClient
from comfyui_story_agent.marketing_agent import MarketingVideoAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


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


def main():
    output_dir = Path(__file__).resolve().parent / "marketing_writer_production"
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

    product_name = "RestFlow Sleep Pod"
    product_concept = (
        "A premium sleep optimization device that combines smart temperature control, "
        "gentle motion cues, and a sleek lifestyle design for people who want better rest."
    )

    logging.info("🔎 Generating product research and creative direction...")
    research = agent.recommend_product_opportunities(
        category="health and wellness gadgets",
        platforms=["TikTok", "YouTube Shorts", "Instagram Reels"],
        num_ideas=3,
    )
    research_path = output_dir / "product_research.json"
    with research_path.open("w", encoding="utf-8") as f:
        json.dump(research, f, indent=2)
    logging.info(f"Saved product research → {research_path}")

    logging.info("🎬 Generating storyboard for a 30-second ad...")
    storyboard = agent.conceptualize_video(
        product_concept=product_concept,
        num_shots=4,
        target_length_sec=30,
        platforms=["TikTok", "YouTube Shorts"],
    )
    storyboard_path = output_dir / "marketing_storyboard.yaml"
    storyboard.to_yaml(str(storyboard_path))
    logging.info(f"Saved storyboard → {storyboard_path}")

    logging.info("✍️  Writing the ad script copy...")
    script = agent.create_ad_script(
        storyboard=storyboard,
        product_name=product_name,
        platform="TikTok",
        tone="urgent and cinematic",
    )
    script_path = output_dir / "ad_script.json"
    with script_path.open("w", encoding="utf-8") as f:
        json.dump(script, f, indent=2)
    logging.info(f"Saved ad script → {script_path}")

    # Anchor the last shot with the CTA to help the TextDirector
    if storyboard.shots and script.get("cta"):
        cta_text = script["cta"]
        if isinstance(cta_text, dict):
            cta_text = cta_text.get("voice_over") or cta_text.get("text_on_screen") or cta_text.get("button_text")
        storyboard.shots[-1].expected_text = cta_text
        storyboard.to_yaml(str(storyboard_path))
        logging.info(f"Updated storyboard last shot expected text with CTA: {cta_text}")

    logging.info("🚀 Producing the 30-second advertisement with Director QA...")
    results = agent.produce_storyboard(
        storyboard,
        on_progress=lambda msg, cur, total: logging.info(f"  [{cur}/{total}] {msg}"),
    )

    results_path = output_dir / "production_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    logging.info(f"Saved production results → {results_path}")
    logging.info("✅ Finished production. Check output clips and the Director report.")


if __name__ == "__main__":
    main()
