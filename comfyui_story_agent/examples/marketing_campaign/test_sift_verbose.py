import cv2
import numpy as np
import glob
from pathlib import Path

def is_valid_homography_verbose(M, w, h):
    if M is None:
        return False, "M is None"
    
    # Calculate rotation angle from the affine component of M
    a = M[0, 0]
    c = M[1, 0]
    theta = np.arctan2(c, a)
    rot_deg = np.abs(np.degrees(theta))
    if rot_deg > 15.0:
        return False, f"Rotation too large: {rot_deg:.1f} degrees"

    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    try:
        warped = cv2.perspectiveTransform(corners, M).reshape(-1, 2)
    except Exception as e:
        return False, f"Exception: {e}"

    # Check 1: Must be a convex quad
    if not cv2.isContourConvex(np.array(warped, dtype=np.float32)):
        return False, f"Not convex: {warped.tolist()}"

    # Check 2: Area must be within reasonable bounds (allowing zoom-in up to 8x original size)
    area = cv2.contourArea(np.array(warped, dtype=np.float32))
    if area < 0.1 * w * h or area > 8.0 * w * h:
        return False, f"Area bounds: {area:.1f} (allowed {0.1*w*h:.1f} to {8.0*w*h:.1f})"

    # Check 3: Coordinate bounds check (allow points to go off-screen during zoom-in)
    for pt in warped:
        if pt[0] < -2.0 * w or pt[0] > 3.0 * w or pt[1] < -2.0 * h or pt[1] > 3.0 * h:
            return False, f"Coord bounds: {pt.tolist()}"

    return True, "Valid"

def test_sift_verbose():
    base_image_path = "/home/mikeyb/Documents/AI/ComfyUI/comfyui_story_agent/examples/marketing_campaign/eight_sleep_anchors/shot2_start.png"
    scene_dir = Path("/home/mikeyb/Documents/AI/ComfyUI/comfyui_story_agent/examples/marketing_campaign/eight_sleep_output_v13/scene_02")
    clips_dir = scene_dir / "clips"
    frame_files = sorted(glob.glob(str(clips_dir / "shot_001_frames_*.png")))
    
    base_img = cv2.imread(base_image_path)
    first_frame = cv2.imread(frame_files[0])
    th, tw = first_frame.shape[:2]
    base_img = cv2.resize(base_img, (tw, th))
    h, w = base_img.shape[:2]

    pts = np.array([[455, 140], [700, 165], [820, 620], [570, 655]], dtype=np.int32)

    feature_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(feature_mask, (380, 80), (880, 700), 255, -1)
    cv2.fillPoly(feature_mask, [pts], 0)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(base_img, mask=feature_mask)
    matcher = cv2.BFMatcher()

    print(f"Base keypoints: {len(kp1)}")

    for i in range(1, len(frame_files), 10): # Sample every 10 frames
        frame = cv2.imread(frame_files[i])
        kp2, des2 = sift.detectAndCompute(frame, mask=feature_mask)
        
        num_matches = 0
        status_str = "No keypoints/descriptors"
        
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
            
            num_matches = len(good)
            
            if len(good) > 5:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                A, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
                if A is not None:
                    temp_M = np.eye(3)
                    temp_M[:2, :] = A
                    valid, msg = is_valid_homography_verbose(temp_M, w, h)
                    status_str = f"Affine valid: {valid} ({msg})"
                else:
                    status_str = "Affine estimation failed"
            else:
                status_str = "Not enough good matches (< 5)"
                
        print(f"Frame {i:03d}: Keypoints={len(kp2) if kp2 else 0}, Good Matches={num_matches}, Status: {status_str}")

if __name__ == "__main__":
    test_sift_verbose()
