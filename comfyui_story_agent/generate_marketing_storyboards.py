import os
import sys
import logging

# Add the parent directory to the path so we can import the module correctly
sys.path.append("/home/mikeyb/Documents/AI/ComfyUI")

from comfyui_story_agent.marketing_agent import MarketingVideoAgent
from comfyui_story_agent.comfyui_client import ComfyUIClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    output_dir = "/home/mikeyb/Documents/AI/ComfyUI/comfyui_story_agent/examples/marketing_campaign"
    os.makedirs(output_dir, exist_ok=True)
    
    # We can pass a dummy client if we only want to conceptualize (since conceptualize_video doesn't use the client)
    # But let's instantiate the real one just in case
    client = ComfyUIClient(host="127.0.0.1", port=8188)
    
    agent = MarketingVideoAgent(client=client, output_dir=output_dir)

    concepts = {
        "eight_sleep": "A 3-shot cozy aesthetic marketing video for Eight Sleep Pod. Shot 1: A cinematic, slow-motion shot of a person falling onto a luxurious, softly lit bed at night, moody blue and warm amber lighting. Shot 2: Extreme close-up of the Eight Sleep cooling grid technology and modern app interface glowing in the dark, shallow depth of field. Shot 3: The person waking up looking incredibly refreshed, sunlight softly filtering through the blinds, peaceful and mindful morning.",
        "semrush": "A 3-shot professional marketing video for Semrush. Shot 1: A dynamic, high-contrast shot of a sleek laptop screen showing massive traffic growth charts, professional office lighting. Shot 2: Close-up of hands typing rapidly on a backlit keyboard while a floating AR projection of the Semrush keyword dashboard appears in focus. Shot 3: A confident, well-dressed professional smiling directly at the camera in a modern co-working space, pointing to a Call-To-Action overlay.",
        "kinsta": "A 3-shot high-energy marketing video for Kinsta Cloud Hosting. Shot 1: A fast-paced, dramatic shot of a glowing server room with red warning lights, transitioning instantly to cool, stable blue lights. Shot 2: Extreme close up of a speedometer graphic maxing out next to the Kinsta logo on a sleek monitor, neon cyberpunk lighting. Shot 3: A tech creator looking mind-blown, taking off their glasses in disbelief while holding their phone displaying a 100% performance score."
    }

    for name, concept in concepts.items():
        logging.info(f"Generating storyboard for {name}...")
        storyboard = agent.conceptualize_video(concept, num_shots=3)
        filename = f"{name}_storyboard.yaml"
        agent.save_storyboard(storyboard, filename)
        logging.info(f"Successfully saved {filename}")

if __name__ == "__main__":
    main()
