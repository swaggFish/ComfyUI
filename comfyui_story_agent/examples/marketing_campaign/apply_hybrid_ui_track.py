import cv2
import numpy as np
import glob
import os
import subprocess

# Paths
base_image_path = "/home/mikeyb/Documents/AI/ComfyUI/comfyui_story_agent/examples/marketing_campaign/eight_sleep_anchors/shot2_start.png"
frames_dir = "/home/mikeyb/Documents/AI/ComfyUI/comfyui_story_agent/examples/marketing_campaign/eight_sleep_output_v6/clips"
out_frames_dir = "/home/mikeyb/Documents/AI/ComfyUI/comfyui_story_agent/examples/marketing_campaign/eight_sleep_output_v6/clips_tracked"
out_video = "/home/mikeyb/Documents/AI/ComfyUI/comfyui_story_agent/examples/marketing_campaign/eight_sleep_output_v6/clips/shot_002_hybrid_ui.mp4"

os.makedirs(out_frames_dir, exist_ok=True)

frame_files = sorted(glob.glob(os.path.join(frames_dir, "story_i2v_frames_*.png")))
print(f"Found {len(frame_files)} frames to track and composite.")

base_img = cv2.imread(base_image_path)
if len(frame_files) > 0:
    first_frame = cv2.imread(frame_files[0])
    th, tw = first_frame.shape[:2]
    base_img = cv2.resize(base_img, (tw, th))

h, w = base_img.shape[:2]

# 1. Define the UI area (Phone Screen) to overlay.
# We keep the center of the phone screen where the text/UI is, avoiding the hands and edges.
ui_mask = np.zeros((h, w), dtype=np.uint8)
# Soft bounds inside the phone screen
cv2.rectangle(ui_mask, (w//2 - 130, h//2 - 180), (w//2 + 130, h//2 + 220), 255, -1)
# Blur the mask heavily for a seamless blend into the generated phone bezel
ui_mask_blurred = cv2.GaussianBlur(ui_mask, (41, 41), 0) / 255.0
ui_mask_blurred = np.stack([ui_mask_blurred]*3, axis=2)

# 2. Define the Feature Matching mask.
# We want OpenCV to track the movement of the HANDS and PHONE BEZEL,
# so we MUST ignore the screen area (warped text) AND the background (parallax).
feature_mask = np.zeros((h, w), dtype=np.uint8)
# Enable tracking just around the phone's expected bounds
cv2.rectangle(feature_mask, (w//2 - 250, h//2 - 350), (w//2 + 250, h//2 + 400), 255, -1)
# Disable tracking inside the actual screen where the bad text morphs
cv2.rectangle(feature_mask, (w//2 - 150, h//2 - 200), (w//2 + 150, h//2 + 240), 0, -1)

# Initialize SIFT detector for robust feature matching
sift = cv2.SIFT_create()

# Compute anchor features
kp1, des1 = sift.detectAndCompute(base_img, mask=feature_mask)
matcher = cv2.BFMatcher()

# Frame files already loaded above

for i, fpath in enumerate(frame_files):
    frame = cv2.imread(fpath)
    if frame is None:
        continue
        
    # Extract features from current video frame
    kp2, des2 = sift.detectAndCompute(frame, mask=feature_mask)
    
    # Match features
    if des2 is not None and len(des2) > 0:
        matches = matcher.knnMatch(des1, des2, k=2)
        good = []
        for match_pts in matches:
            if len(match_pts) == 2:
                m, n = match_pts
                # Lowe's ratio test
                if m.distance < 0.75 * n.distance:
                    good.append(m)
        
        # If we have enough good matches, calculate Homography (the camera movement/zoom)
        if len(good) > 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if M is not None:
                # Warp the perfect base UI to match the zoom/pan of the current frame!
                warped_base = cv2.warpPerspective(base_img, M, (w, h))
                warped_mask = cv2.warpPerspective(ui_mask_blurred, M, (w, h))
                
                # Composite the perfect UI over the generated video frame
                frame = frame * (1 - warped_mask) + warped_base * warped_mask
                frame = np.clip(frame, 0, 255).astype(np.uint8)
    
    out_path = os.path.join(out_frames_dir, f"frame_{i:04d}.png")
    cv2.imwrite(out_path, frame)
    
    if i % 10 == 0:
        print(f"Processed {i}/{len(frame_files)} frames")

print("Tracking and compositing complete. Encoding to MP4...")
subprocess.run([
    "ffmpeg", "-y", "-framerate", "24", "-i", os.path.join(out_frames_dir, "frame_%04d.png"),
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out_video
], check=True)

print(f"✅ Success! Hybrid video saved to {out_video}")
