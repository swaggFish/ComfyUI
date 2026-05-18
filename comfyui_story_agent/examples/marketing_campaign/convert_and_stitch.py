import cv2
import numpy as np
from PIL import Image
import os
import subprocess

def webp_to_mp4(webp_path, mp4_path, fps=24):
    if not os.path.exists(webp_path): return False
    img = Image.open(webp_path)
    frames = []
    try:
        while True:
            frame = img.convert('RGB')
            frames.append(np.array(frame))
            img.seek(img.tell() + 1)
    except EOFError:
        pass
    
    if not frames: return False
    h, w, _ = frames[0].shape
    out = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for frame in frames: out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
    return True

base = "/home/mikeyb/Documents/AI/ComfyUI/comfyui_story_agent/examples/marketing_campaign/"

for w, m in [("semrush_output_final/clips/story_t2v_00001_.webp", "semrush_output_final/clips/shot1.mp4"),
             ("semrush_output_final/clips/story_t2v_00002_.webp", "semrush_output_final/clips/shot2.mp4"),
             ("semrush_output_final/clips/story_t2v_00003_.webp", "semrush_output_final/clips/shot3.mp4"),
             ("kinsta_output_final/clips/story_t2v_00004_.webp", "kinsta_output_final/clips/shot1.mp4"),
             ("kinsta_output_final/clips/story_t2v_00005_.webp", "kinsta_output_final/clips/shot2.mp4"),
             ("kinsta_output_final/clips/story_t2v_00006_.webp", "kinsta_output_final/clips/shot3.mp4")]:
    webp_to_mp4(base + w, base + m)

def stitch(name, input_dir):
    cmd = [
        "ffmpeg", "-y", 
        "-i", f"{base}{input_dir}/clips/shot1.mp4",
        "-i", f"{base}{input_dir}/clips/shot2.mp4",
        "-i", f"{base}{input_dir}/clips/shot3.mp4",
        "-filter_complex", "[0:v]scale=1280:704,setsar=1[v0]; [1:v]scale=1280:704,setsar=1[v1]; [2:v]scale=1280:704,setsar=1[v2]; [v0][v1][v2]concat=n=3:v=1:a=0[outv]",
        "-map", "[outv]", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        f"/home/mikeyb/.gemini/antigravity/brain/38e67136-715f-4218-afb0-2345a89a1faa/artifacts/{name}_commercial.mp4"
    ]
    subprocess.run(cmd)

stitch("semrush", "semrush_output_final")
stitch("kinsta", "kinsta_output_final")
