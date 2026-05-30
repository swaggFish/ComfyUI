import cv2
import numpy as np
import glob
import os
from pathlib import Path

def is_valid_homography(M, w, h):
    if M is None:
        return False
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    try:
        warped = cv2.perspectiveTransform(corners, M).reshape(-1, 2)
    except Exception:
        return False

    if not cv2.isContourConvex(np.array(warped, dtype=np.float32)):
        return False

    area = cv2.contourArea(np.array(warped, dtype=np.float32))
    if area < 0.15 * w * h or area > 2.0 * w * h:
        return False

    for pt in warped:
        if pt[0] < -0.5 * w or pt[0] > 1.5 * w or pt[1] < -0.5 * h or pt[1] > 1.5 * h:
            return False

    return True

def test_klt():
    base_image_path = "/home/mikeyb/Documents/AI/ComfyUI/comfyui_story_agent/examples/marketing_campaign/eight_sleep_anchors/shot2_start.png"
    scene_dir = Path("/home/mikeyb/Documents/AI/ComfyUI/comfyui_story_agent/examples/marketing_campaign/eight_sleep_output_v13/scene_02")
    clips_dir = scene_dir / "clips"
    frame_files = sorted(glob.glob(str(clips_dir / "shot_001_frames_*.png")))
    
    if not frame_files:
        print("No frame files found!")
        return

    base_img = cv2.imread(base_image_path)
    first_frame = cv2.imread(frame_files[0])
    th, tw = first_frame.shape[:2]
    base_img = cv2.resize(base_img, (tw, th))
    h, w = base_img.shape[:2]

    # Precise polygon corners representing the glowing phone screen area
    pts = np.array([[455, 140], [700, 165], [820, 620], [570, 655]], dtype=np.int32)

    # Let's define the mask for features to track:
    tracking_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(tracking_mask, (380, 80), (850, 700), 255, -1)
    cv2.fillPoly(tracking_mask, [pts], 0)

    # Detect features in the first frame
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=8, mask=tracking_mask)

    if p0 is None or len(p0) < 10:
        print("Failed to detect features to track!")
        return

    # Keep track of initial coordinates
    initial_pts = p0.copy()
    current_pts = p0.copy()

    lk_params = dict(winSize=(21, 21),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    # Mask for UI
    ui_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(ui_mask, [pts], 255)
    ui_mask_blurred = cv2.GaussianBlur(ui_mask, (15, 15), 0) / 255.0
    ui_mask_blurred = np.stack([ui_mask_blurred]*3, axis=2)

    os.makedirs("/tmp/klt_test_bi", exist_ok=True)

    last_M = np.eye(3)
    
    for i in range(1, len(frame_files)):
        frame = cv2.imread(frame_files[i])
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Forward flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, current_pts, None, **lk_params)
        # Backward flow
        p0_reconstructed, st_back, err_back = cv2.calcOpticalFlowPyrLK(frame_gray, prev_gray, p1, None, **lk_params)

        # Check forward-backward distance
        d = np.linalg.norm((current_pts - p0_reconstructed).reshape(-1, 2), axis=1)
        valid = (st.reshape(-1) == 1) & (st_back.reshape(-1) == 1) & (d < 0.5)

        good_new = p1[valid]
        good_old_initial = initial_pts[valid]

        # Update tracking lists
        current_pts = good_new.reshape(-1, 1, 2)
        initial_pts = good_old_initial.reshape(-1, 1, 2)
        prev_gray = frame_gray.copy()

        # Compute Homography
        if len(good_new) > 10:
            M, inliers = cv2.findHomography(good_old_initial, good_new, cv2.RANSAC, 5.0)
            if is_valid_homography(M, w, h):
                last_M = M
            else:
                print(f"Frame {i}: degenerate homography detected!")

        # Warp
        warped_base = cv2.warpPerspective(base_img, last_M, (w, h))
        warped_mask = cv2.warpPerspective(ui_mask_blurred, last_M, (w, h))
        
        # Composite
        composite = frame * (1 - warped_mask) + warped_base * warped_mask
        composite = np.clip(composite, 0, 255).astype(np.uint8)
        
        if i in [1, 20, 50, 100, 150, 200, len(frame_files)-1]:
            cv2.imwrite(f"/tmp/klt_test_bi/frame_{i:03d}.png", composite)

    print("KLT tracking test completed. Check /tmp/klt_test_bi/ for frames.")

if __name__ == "__main__":
    test_klt()
