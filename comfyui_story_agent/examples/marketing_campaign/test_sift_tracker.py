import cv2
import numpy as np
import glob
import os
from pathlib import Path

def is_valid_homography(M, w, h):
    if M is None:
        return False
        
    # Calculate rotation angle from the affine component of M
    a = M[0, 0]
    c = M[1, 0]
    theta = np.arctan2(c, a)
    rot_deg = np.abs(np.degrees(theta))
    if rot_deg > 15.0:
        return False

    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    try:
        warped = cv2.perspectiveTransform(corners, M).reshape(-1, 2)
    except Exception:
        return False

    # Check 1: Must be a convex quad
    if not cv2.isContourConvex(np.array(warped, dtype=np.float32)):
        return False

    # Check 2: Area must be within reasonable bounds (allowing zoom-in up to 8x original size)
    area = cv2.contourArea(np.array(warped, dtype=np.float32))
    if area < 0.1 * w * h or area > 8.0 * w * h:
        return False

    # Check 3: Coordinate bounds check (allow points to go off-screen during zoom-in)
    for pt in warped:
        if pt[0] < -2.0 * w or pt[0] > 3.0 * w or pt[1] < -2.0 * h or pt[1] > 3.0 * h:
            return False

    return True

def test_sift():
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

    # UI mask with tight blur
    ui_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(ui_mask, [pts], 255)
    ui_mask_blurred = cv2.GaussianBlur(ui_mask, (15, 15), 0) / 255.0
    ui_mask_blurred = np.stack([ui_mask_blurred]*3, axis=2)

    # Restrict SIFT matching purely to phone bezel and hands
    feature_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(feature_mask, (380, 80), (880, 700), 255, -1)
    cv2.fillPoly(feature_mask, [pts], 0)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(base_img, mask=feature_mask)
    matcher = cv2.BFMatcher()

    os.makedirs("/tmp/sift_test", exist_ok=True)

    last_M = None
    fallback_count = 0
    estimated_matrices = []

    # Pass 1: Estimate raw transformation matrices
    for i, fpath in enumerate(frame_files):
        frame = cv2.imread(fpath)
        if frame is None:
            estimated_matrices.append(None)
            continue

        M = None
        # Detect SIFT keypoints on target frame (using the same mask)
        kp2, des2 = sift.detectAndCompute(frame, mask=feature_mask)

        if des2 is not None and len(des2) > 0:
            matches = matcher.knnMatch(des1, des2, k=2)
            good = []
            for match_pts in matches:
                if len(match_pts) == 2:
                    m, n = match_pts
                    if m.distance < 0.75 * n.distance:
                        good.append(m)

            # Ensure unique keypoints in the target frame to prevent many-to-one collapse
            good = sorted(good, key=lambda x: x.distance)
            seen_dst = set()
            unique_good = []
            for m in good:
                if m.trainIdx not in seen_dst:
                    seen_dst.add(m.trainIdx)
                    unique_good.append(m)
            good = unique_good

            if len(good) > 5:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                A, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
                if A is not None:
                    temp_M = np.eye(3)
                    temp_M[:2, :] = A
                    if is_valid_homography(temp_M, w, h):
                        M = temp_M
                        last_M = M

        if M is None and last_M is not None:
            M = last_M
            fallback_count += 1
        
        estimated_matrices.append(M)

    # Pass 2: Smooth matrices temporally using a moving window (Gaussian/Box filter)
    # Using window size of 7 for rock-solid stability
    smoothed_matrices = []
    window_size = 7
    half_w = window_size // 2

    for i in range(len(estimated_matrices)):
        valid_window_matrices = []
        for offset in range(-half_w, half_w + 1):
            idx = i + offset
            if 0 <= idx < len(estimated_matrices) and estimated_matrices[idx] is not None:
                valid_window_matrices.append(estimated_matrices[idx])

        if valid_window_matrices:
            smoothed_M = np.mean(valid_window_matrices, axis=0)
            smoothed_matrices.append(smoothed_M)
        else:
            smoothed_matrices.append(estimated_matrices[i])

    # Pass 3: Render composited frames with smoothed matrices
    applied_count = 0
    for i, fpath in enumerate(frame_files):
        frame = cv2.imread(fpath)
        if frame is None:
            continue

        M = smoothed_matrices[i]
        if M is not None:
            warped_base = cv2.warpPerspective(base_img, M, (w, h))
            warped_mask = cv2.warpPerspective(ui_mask_blurred, M, (w, h))
            frame = frame * (1 - warped_mask) + warped_base * warped_mask
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            applied_count += 1

        if i in [1, 20, 50, 100, 150, 200, len(frame_files)-1]:
            cv2.imwrite(f"/tmp/sift_test/frame_{i:03d}.png", frame)

    print(f"SIFT test completed: applied={applied_count}, fallback={fallback_count}")

if __name__ == "__main__":
    test_sift()
