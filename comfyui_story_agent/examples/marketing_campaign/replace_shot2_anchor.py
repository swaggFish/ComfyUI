import os
from PIL import Image

input_path = "/home/mikeyb/.gemini/antigravity/brain/38e67136-715f-4218-afb0-2345a89a1faa/eight_sleep_app_anchor_1778983209511.png"
out_path = "/home/mikeyb/Documents/AI/ComfyUI/comfyui_story_agent/examples/marketing_campaign/eight_sleep_anchors/shot2_start.png"

img = Image.open(input_path)

def resize_crop(img, target_w=1280, target_h=704):
    img_ratio = img.width / img.height
    target_ratio = target_w / target_h
    
    if img_ratio > target_ratio:
        # Image is wider
        new_h = target_h
        new_w = int(new_h * img_ratio)
    else:
        # Image is taller
        new_w = target_w
        new_h = int(new_w / img_ratio)
        
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    
    # Center crop
    left = (new_w - target_w) / 2
    top = (new_h - target_h) / 2
    right = (new_w + target_w) / 2
    bottom = (new_h + target_h) / 2
    
    return img_resized.crop((left, top, right, bottom))

new_img = resize_crop(img)
new_img.save(out_path)
print(f"Successfully replaced {out_path} with new legible Gemini-generated image.")
