import os
from PIL import Image

out_dir = "/home/mikeyb/Documents/AI/ComfyUI/comfyui_story_agent/examples/marketing_campaign/eight_sleep_anchors"
os.makedirs(out_dir, exist_ok=True)

img1_path = "/home/mikeyb/.gemini/antigravity/brain/38e67136-715f-4218-afb0-2345a89a1faa/media__1778974354720.jpg"
img2_path = "/home/mikeyb/.gemini/antigravity/brain/38e67136-715f-4218-afb0-2345a89a1faa/media__1778974354759.jpg"

if os.path.exists(img1_path):
    img1 = Image.open(img1_path)
    w, h = img1.size
    pw = w // 3
    
    panel1 = img1.crop((0, 0, pw, h))
    panel2 = img1.crop((pw, 0, 2*pw, h))
    panel3 = img1.crop((2*pw, 0, w, h))
    
    # We will resize them to 1280x704 to be safe. Since they are vertical, 
    # doing a center-crop resize is best to avoid stretching.
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
        
    p1_rc = resize_crop(panel1)
    p2_rc = resize_crop(panel2)
    p3_rc = resize_crop(panel3)
    
    p1_rc.save(os.path.join(out_dir, "shot1_start.png"))
    p2_rc.save(os.path.join(out_dir, "shot2_start.png"))
    p3_rc.save(os.path.join(out_dir, "shot3_start.png"))
    print("Processed img1 into 3 panels")

if os.path.exists(img2_path):
    img2 = Image.open(img2_path)
    p4_rc = resize_crop(img2)
    p4_rc.save(os.path.join(out_dir, "shot3_end.png"))
    print("Processed img2 as shot3_end")
