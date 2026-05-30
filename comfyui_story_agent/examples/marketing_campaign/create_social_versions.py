import os
import subprocess
import shutil
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

COMFYUI_ROOT = Path("/home/mikeyb/Documents/AI/ComfyUI")
ART = Path("/home/mikeyb/.gemini/antigravity/brain/ed14af04-07db-4537-881e-4d8d1aa3f71b/artifacts")
OUTPUT_DIR = ART / "socials"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FONT_PATH = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"

def generate_vertical_overlay(filename, platform, text_label):
    # 720x1280 transparent image
    img = Image.new("RGBA", (720, 1280), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    try:
        font_large = ImageFont.truetype(FONT_PATH, 28)
        font_small = ImageFont.truetype(FONT_PATH, 20)
        font_badge = ImageFont.truetype(FONT_PATH, 16)
    except Exception:
        font_large = font_small = font_badge = ImageFont.load_default()

    # Draw profile circle on the right side
    px, py = 660, 600
    pr = 30
    draw.ellipse([px - pr, py - pr, px + pr, py + pr], fill=(50, 50, 50, 220), outline=(255, 255, 255, 255), width=2)
    draw.text((px - 8, py - 18), "8", fill=(255, 255, 255, 255), font=font_large)
    
    # Plus button at bottom of profile
    draw.ellipse([px - 10, py + pr - 10, px + 10, py + pr + 10], fill=(254, 44, 85, 255))
    draw.text((px - 5, py + pr - 12), "+", fill=(255, 255, 255, 255), font=font_badge)

    # Like heart (TikTok is red, others are white/gray)
    hx, hy = 660, 720
    heart_color = (254, 44, 85, 255) if platform == "tiktok" else (255, 255, 255, 240)
    draw.ellipse([hx - 20, hy - 20, hx, hy], fill=heart_color)
    draw.ellipse([hx, hy - 20, hx + 20, hy], fill=heart_color)
    draw.polygon([hx - 20, hy - 10, hx + 20, hy - 10, hx, hy + 20], fill=heart_color)
    draw.text((hx - 25, hy + 25), "24.5K" if platform == "tiktok" else "12.8K", fill=(255, 255, 255, 255), font=font_badge)

    # Comment bubble
    cx, cy = 660, 840
    draw.rounded_rectangle([cx - 20, cy - 15, cx + 20, cy + 15], radius=6, fill=(255, 255, 255, 240))
    draw.polygon([cx - 12, cy + 12, cx - 4, cy + 12, cx - 8, cy + 20], fill=(255, 255, 255, 240))
    draw.line([cx - 10, cy - 4, cx + 10, cy - 4], fill=(0, 0, 0, 150), width=2)
    draw.line([cx - 10, cy + 4, cx + 10, cy + 4], fill=(0, 0, 0, 150), width=2)
    draw.text((cx - 20, cy + 22), "1,248" if platform == "tiktok" else "582", fill=(255, 255, 255, 255), font=font_badge)

    # Share arrow
    sx, sy = 660, 960
    draw.ellipse([sx - 22, sy - 22, sx + 22, sy + 22], fill=(255, 255, 255, 240))
    draw.polygon([sx - 4, sy - 12, sx + 12, sy, sx - 4, sy + 12], fill=(0, 0, 0, 220))
    draw.polygon([sx - 12, sy - 4, sx - 4, sy - 4, sx - 4, sy + 4, sx - 12, sy + 4], fill=(0, 0, 0, 220))
    draw.text((sx - 15, sy + 25), "15K" if platform == "tiktok" else "Share", fill=(255, 255, 255, 255), font=font_badge)

    # Sound disk
    dx, dy = 660, 1080
    draw.ellipse([dx - 22, dy - 22, dx + 22, dy + 22], fill=(20, 20, 20, 255))
    draw.ellipse([dx - 14, dy - 14, dx + 14, dy + 14], fill=(80, 80, 80, 255))
    draw.ellipse([dx - 6, dy - 6, dx + 6, dy + 6], fill=(255, 255, 255, 255))

    # Bottom Caption Box
    draw.text((40, 1130), "@eightsleep", fill=(255, 255, 255, 255), font=font_large)
    caption_text = f"Tired of tossing and turning? Optimize your body's recovery with the Eight Sleep Pod. {text_label}"
    
    # Simple word wrapper
    words = caption_text.split()
    lines = []
    current_line = []
    for word in words:
        current_line.append(word)
        line_w = draw.textlength(" ".join(current_line), font=font_small)
        if line_w > 550:
            current_line.pop()
            lines.append(" ".join(current_line))
            current_line = [word]
    if current_line:
        lines.append(" ".join(current_line))
        
    y_off = 1170
    for line in lines:
        draw.text((40, y_off), line, fill=(255, 255, 255, 230), font=font_small)
        y_off += 24

    img.save(filename)

def generate_shorts_overlay(filename):
    img = Image.new("RGBA", (720, 1280), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        font_large = ImageFont.truetype(FONT_PATH, 28)
        font_small = ImageFont.truetype(FONT_PATH, 20)
        font_badge = ImageFont.truetype(FONT_PATH, 16)
    except Exception:
        font_large = font_small = font_badge = ImageFont.load_default()

    # Thumbs up (Like)
    lx, ly = 660, 680
    draw.ellipse([lx - 22, ly - 22, lx + 22, ly + 22], fill=(40, 40, 40, 220))
    draw.text((lx - 8, ly - 14), "👍", fill=(255, 255, 255, 255), font=font_small)
    draw.text((lx - 12, ly + 25), "10K", fill=(255, 255, 255, 255), font=font_badge)

    # Thumbs down (Dislike)
    dx, dy = 660, 780
    draw.ellipse([dx - 22, dy - 22, dx + 22, dy + 22], fill=(40, 40, 40, 220))
    draw.text((dx - 8, dy - 12), "👎", fill=(255, 255, 255, 255), font=font_small)
    draw.text((dx - 22, dy + 25), "Dislike", fill=(255, 255, 255, 255), font=font_badge)

    # Comment bubble
    cx, cy = 660, 880
    draw.ellipse([cx - 22, cy - 22, cx + 22, cy + 22], fill=(40, 40, 40, 220))
    draw.text((cx - 8, cy - 14), "💬", fill=(255, 255, 255, 255), font=font_small)
    draw.text((cx - 12, cy + 25), "342", fill=(255, 255, 255, 255), font=font_badge)

    # Share arrow
    sx, sy = 660, 980
    draw.ellipse([sx - 22, sy - 22, sx + 22, sy + 22], fill=(40, 40, 40, 220))
    draw.text((sx - 8, sy - 14), "➡️", fill=(255, 255, 255, 255), font=font_small)
    draw.text((sx - 16, sy + 25), "Share", fill=(255, 255, 255, 255), font=font_badge)

    # Remix icon
    rx, ry = 660, 1080
    draw.rectangle([rx - 20, ry - 20, rx + 20, ry + 20], fill=(40, 40, 40, 220), outline=(255, 255, 255, 255), width=2)
    draw.text((rx - 8, ry - 14), "💿", fill=(255, 255, 255, 255), font=font_small)

    # Bottom left profile + Subscribe button
    draw.ellipse([40 - 15, 1130 - 15, 40 + 15, 1130 + 15], fill=(80, 80, 80, 255))
    draw.text((36, 1118), "8", fill=(255, 255, 255, 255), font=font_small)
    draw.text((70, 1115), "@eightsleep", fill=(255, 255, 255, 255), font=font_large)
    
    # Subscribe button
    draw.rounded_rectangle([250, 1115, 390, 1145], radius=15, fill=(255, 0, 0, 255))
    draw.text((265, 1120), "Subscribe", fill=(255, 255, 255, 255), font=font_small)

    # Caption
    caption_text = "The future of sleep technology is here. #shorts #eightsleep"
    draw.text((40, 1165), caption_text, fill=(255, 255, 255, 230), font=font_small)
    img.save(filename)

def generate_cta_card(filename, width, height):
    img = Image.new("RGB", (width, height), (10, 15, 28))
    draw = ImageDraw.Draw(img)
    
    try:
        font_title = ImageFont.truetype(FONT_PATH, int(height * 0.08))
        font_subtitle = ImageFont.truetype(FONT_PATH, int(height * 0.045))
        font_promo = ImageFont.truetype(FONT_PATH, int(height * 0.055))
        font_web = ImageFont.truetype(FONT_PATH, int(height * 0.04))
    except Exception:
        font_title = font_subtitle = font_promo = font_web = ImageFont.load_default()
        
    draw.rectangle([20, 20, width - 20, height - 20], outline=(255, 215, 0, 80), width=4)
    
    title_text = "EIGHT SLEEP POD"
    subtitle_text = "UNLEASH YOUR POTENTIAL"
    promo_text = "USE CODE: SLEEP20 FOR 20% OFF"
    web_text = "eightsleep.com"
    
    t_w = draw.textlength(title_text, font=font_title)
    draw.text(((width - t_w)//2, int(height * 0.25)), title_text, fill=(255, 255, 255), font=font_title)
    
    s_w = draw.textlength(subtitle_text, font=font_subtitle)
    draw.text(((width - s_w)//2, int(height * 0.4)), subtitle_text, fill=(180, 200, 255), font=font_subtitle)
    
    p_w = draw.textlength(promo_text, font=font_promo)
    box_padding = 20
    bx0 = (width - p_w)//2 - box_padding
    by0 = int(height * 0.55) - box_padding
    bx1 = (width - p_w)//2 + p_w + box_padding
    by1 = int(height * 0.55) + int(height * 0.075) + box_padding
    
    draw.rounded_rectangle([bx0, by0, bx1, by1], radius=8, fill=(254, 44, 85, 255))
    draw.text(((width - p_w)//2, int(height * 0.55) + 5), promo_text, fill=(255, 255, 255), font=font_promo)
    
    w_w = draw.textlength(web_text, font=font_web)
    draw.text(((width - w_w)//2, int(height * 0.8)), web_text, fill=(255, 215, 0), font=font_web)
    
    img.save(filename)

def main():
    print("Generating CTA cards...")
    cta_landscape_png = OUTPUT_DIR / "cta_landscape.png"
    cta_square_png = OUTPUT_DIR / "cta_square.png"

    generate_cta_card(str(cta_landscape_png), 1280, 704)
    generate_cta_card(str(cta_square_png), 1080, 1080)

    # Create 15.67s silent videos of the CTA cards
    print("Creating CTA card videos...")
    cta_landscape_mp4 = OUTPUT_DIR / "cta_landscape.mp4"
    cta_square_mp4 = OUTPUT_DIR / "cta_square.mp4"

    subprocess.run([
        "ffmpeg", "-y", "-loop", "1", "-i", str(cta_landscape_png),
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-t", "15.666", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
        str(cta_landscape_mp4)
    ], check=True, capture_output=True)
    time.sleep(1.0)

    subprocess.run([
        "ffmpeg", "-y", "-loop", "1", "-i", str(cta_square_png),
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-t", "15.666", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
        str(cta_square_mp4)
    ], check=True, capture_output=True)
    time.sleep(1.0)

    genders = ["female", "male"]
    
    for gender in genders:
        input_video = COMFYUI_ROOT / f"eight_sleep_commercial_v13_{gender}.mp4"
        if not input_video.exists():
            print(f"Base video not found: {input_video}")
            continue

        print(f"\nProcessing {gender.upper()} voiceover version...")

        # ----------------------------------------------------
        # 1. TikTok (9:16, 28.33s)
        # ----------------------------------------------------
        print("  Compiling TikTok version...")
        tiktok_out = OUTPUT_DIR / f"tiktok_{gender}.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(input_video),
            "-filter_complex",
            "[0:v]scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280,gblur=sigma=30[bg];"
            "[0:v]scale=720:-1[fg];"
            "[bg][fg]overlay=0:(1280-h)/2[final_v]",
            "-map", "[final_v]", "-map", "0:a",
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac",
            str(tiktok_out)
        ], check=True, capture_output=True)
        time.sleep(1.0)

        # ----------------------------------------------------
        # 2. Instagram Reels (9:16, 28.33s)
        # ----------------------------------------------------
        print("  Compiling Instagram Reels version...")
        reels_out = OUTPUT_DIR / f"instagram_reels_{gender}.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(input_video),
            "-filter_complex",
            "[0:v]scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280,gblur=sigma=30[bg];"
            "[0:v]scale=720:-1[fg];"
            "[bg][fg]overlay=0:(1280-h)/2[final_v]",
            "-map", "[final_v]", "-map", "0:a",
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac",
            str(reels_out)
        ], check=True, capture_output=True)
        time.sleep(1.0)

        # ----------------------------------------------------
        # 3. YouTube Shorts (9:16, 28.33s)
        # ----------------------------------------------------
        print("  Compiling YouTube Shorts version...")
        shorts_out = OUTPUT_DIR / f"youtube_shorts_{gender}.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(input_video),
            "-filter_complex",
            "[0:v]scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280,gblur=sigma=30[bg];"
            "[0:v]scale=720:-1[fg];"
            "[bg][fg]overlay=0:(1280-h)/2[final_v]",
            "-map", "[final_v]", "-map", "0:a",
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac",
            str(shorts_out)
        ], check=True, capture_output=True)
        time.sleep(1.0)

        # ----------------------------------------------------
        # 4. Facebook Reels (9:16, 28.33s)
        # ----------------------------------------------------
        print("  Compiling Facebook Reels version...")
        fb_out = OUTPUT_DIR / f"facebook_reels_{gender}.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(input_video),
            "-filter_complex",
            "[0:v]scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280,gblur=sigma=30[bg];"
            "[0:v]scale=720:-1[fg];"
            "[bg][fg]overlay=0:(1280-h)/2[final_v]",
            "-map", "[final_v]", "-map", "0:a",
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac",
            str(fb_out)
        ], check=True, capture_output=True)
        time.sleep(1.0)

        # ----------------------------------------------------
        # 5. X (formerly Twitter) (1:1 Square, 44.0s)
        # ----------------------------------------------------
        print("  Compiling X (Twitter) version...")
        base_square_mp4 = OUTPUT_DIR / f"base_square_{gender}.mp4"
        
        # Convert base video to 1:1 with top/bottom branding
        subprocess.run([
            "ffmpeg", "-y", "-i", str(input_video),
            "-filter_complex",
            "[0:v]scale=1080:1080:force_original_aspect_ratio=increase,crop=1080:1080,gblur=sigma=30[bg];"
            "[0:v]scale=1080:-1[fg];"
            "[bg][fg]overlay=0:(1080-h)/2[final_v]",
            "-map", "[final_v]", "-map", "0:a",
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac",
            str(base_square_mp4)
        ], check=True, capture_output=True)
        time.sleep(1.0)

        # Concatenate 28.33s square video + 15.67s square CTA video = 44s
        x_out = OUTPUT_DIR / f"x_twitter_{gender}.mp4"
        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(base_square_mp4),
            "-i", str(cta_square_mp4),
            "-filter_complex",
            "[0:v]scale=1080:1080,setsar=1,fps=24[v0]; [0:a]aresample=48000[a0]; "
            "[1:v]scale=1080:1080,setsar=1,fps=24[v1]; [1:a]aresample=48000[a1]; "
            "[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]",
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac",
            str(x_out)
        ], check=True, capture_output=True)
        time.sleep(1.0)
        
        # Clean up intermediate file
        if base_square_mp4.exists():
            base_square_mp4.unlink()

        # ----------------------------------------------------
        # 6. LinkedIn (16:9 Landscape, 44.0s)
        # ----------------------------------------------------
        print("  Compiling LinkedIn version...")
        base_linkedin_mp4 = OUTPUT_DIR / f"base_linkedin_{gender}.mp4"
        
        # Keep clean 16:9 base without text
        subprocess.run([
            "ffmpeg", "-y", "-i", str(input_video),
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac",
            str(base_linkedin_mp4)
        ], check=True, capture_output=True)
        time.sleep(1.0)

        # Concatenate 28.33s landscape video + 15.67s landscape CTA video = 44s
        linkedin_out = OUTPUT_DIR / f"linkedin_{gender}.mp4"
        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(base_linkedin_mp4),
            "-i", str(cta_landscape_mp4),
            "-filter_complex",
            "[0:v]scale=1280:704,setsar=1,fps=24[v0]; [0:a]aresample=48000[a0]; "
            "[1:v]scale=1280:704,setsar=1,fps=24[v1]; [1:a]aresample=48000[a1]; "
            "[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]",
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac",
            str(linkedin_out)
        ], check=True, capture_output=True)
        time.sleep(1.0)

        # Clean up intermediate file
        if base_linkedin_mp4.exists():
            base_linkedin_mp4.unlink()

    # Clean up static cta video files
    if cta_landscape_mp4.exists():
        cta_landscape_mp4.unlink()
    if cta_square_mp4.exists():
        cta_square_mp4.unlink()

    print("\n╔═══════════════════════════════════════════════════╗")
    print("║ ✓ SUCCESS: All Social Media Files Compiled        ║")
    print("╚═══════════════════════════════════════════════════╝")

if __name__ == "__main__":
    main()
