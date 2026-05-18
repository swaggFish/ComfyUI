import sys
import logging

# Add ComfyUI to path
sys.path.append("/home/mikeyb/Documents/AI/ComfyUI")
from comfyui_story_agent.text_agent import TextDirector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    director = TextDirector()
    clip_path = "/home/mikeyb/Documents/AI/ComfyUI/comfyui_story_agent/examples/marketing_campaign/eight_sleep_output_v3/clips/shot_002_00020_.webp"
    expected = "Eight Sleep 68°"
    
    print(f"Running TextDirector on {clip_path}...")
    evaluation = director.review_clip(clip_path, expected, shot_id=2)
    
    print("\n--- DIRECTOR EVALUATION ---")
    print(f"Passed: {evaluation.passed}")
    print(f"Summary: {evaluation.summary}")
    print(f"Recommended Action: {evaluation.recommended_action.value}")
    
    if evaluation.issues:
        print("\nDetected Issues:")
        for issue in evaluation.issues:
            print(f"[{issue.frame_location}] Expected: '{issue.expected_text}' | Actual: '{issue.actual_transcription}'")
            print(f"  -> {issue.description}")

if __name__ == "__main__":
    main()
