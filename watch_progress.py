#!/usr/bin/env python3
"""
ComfyUI Generation Progress Monitor
------------------------------------
Polls the ComfyUI REST API (zero GPU impact) and prints a live progress bar.
Usage:
    python watch_progress.py                  # defaults to localhost:8188
    python watch_progress.py --host 127.0.0.1 --port 8188
    python watch_progress.py --output /path/to/output/clips  # also show clip count
"""

import argparse
import sys
import time
import os
import json
import urllib.request
from pathlib import Path
from datetime import timedelta

HOST = "127.0.0.1"
PORT = 8188

CLEAR_LINE = "\033[2K\r"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
CYAN    = "\033[36m"
BOLD    = "\033[1m"
RESET   = "\033[0m"
GREY    = "\033[90m"

def fetch_json(url: str, timeout: int = 3) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None

def bar(value: int, maximum: int, width: int = 30) -> str:
    if maximum <= 0:
        return "[" + "?" * width + "]"
    filled = int(width * value / maximum)
    return "[" + "█" * filled + "░" * (width - filled) + "]"

def format_eta(elapsed: float, value: int, maximum: int) -> str:
    if value <= 0 or maximum <= 0:
        return "--:--"
    rate = value / elapsed if elapsed > 0 else 0
    remaining = (maximum - value) / rate if rate > 0 else 0
    return str(timedelta(seconds=int(remaining)))

def count_clips(clips_dir: Path) -> dict:
    if not clips_dir.exists():
        return {"webp": 0, "mp4": 0, "frames": 0}
    files = list(clips_dir.iterdir())
    return {
        "webp":   sum(1 for f in files if f.suffix == ".webp" and "frames" not in f.name),
        "mp4":    sum(1 for f in files if f.suffix == ".mp4"),
        "frames": sum(1 for f in files if f.suffix == ".png"),
    }

def watch(host: str, port: int, output_dir: Path | None, interval: float = 1.0):
    base = f"http://{host}:{port}"
    print(f"\n{BOLD}ComfyUI Progress Monitor{RESET}  {GREY}({base}){RESET}")
    print(f"{GREY}Press Ctrl+C to stop (generation will continue in background){RESET}\n")

    last_prompt_id = None
    node_history = {}
    gen_start = None
    idle_ticks = 0

    try:
        while True:
            queue    = fetch_json(f"{base}/queue")
            progress = fetch_json(f"{base}/api/comfy/progress") or {}
            system   = fetch_json(f"{base}/system_stats") or {}

            # ── GPU / RAM stats ───────────────────────────────────────────
            vram_used  = system.get("devices", [{}])[0].get("vram_used", 0) // (1024**2) if system.get("devices") else 0
            vram_total = system.get("devices", [{}])[0].get("vram_total", 0) // (1024**2) if system.get("devices") else 0
            ram_used   = system.get("ram", {}).get("used", 0) // (1024**3)
            ram_total  = system.get("ram", {}).get("total", 0) // (1024**3)

            vram_str = f"VRAM {vram_used}/{vram_total}MB" if vram_total else ""
            ram_str  = f"RAM {ram_used}/{ram_total}GB" if ram_total else ""

            # ── Queue state ───────────────────────────────────────────────
            running = queue.get("queue_running", []) if queue else []
            pending = queue.get("queue_pending", []) if queue else []
            n_pending = len(pending)

            # ── Step progress ─────────────────────────────────────────────
            step_val  = progress.get("value", 0)
            step_max  = progress.get("max", 0)
            node_name = progress.get("node", "")
            prompt_id = progress.get("prompt_id", "")

            if running and gen_start is None:
                gen_start = time.time()
            if not running and gen_start is not None:
                idle_ticks += 1
                if idle_ticks >= 3:
                    gen_start = None
                    idle_ticks = 0
            else:
                idle_ticks = 0

            elapsed = (time.time() - gen_start) if gen_start else 0

            # ── Clip count ────────────────────────────────────────────────
            clip_info = ""
            if output_dir:
                counts = count_clips(output_dir)
                clip_info = (
                    f"  │  {CYAN}Clips: {counts['webp']+counts['mp4']} done"
                    f"  Frames: {counts['frames']}{RESET}"
                )

            # ── Render output ─────────────────────────────────────────────
            sys.stdout.write(CLEAR_LINE)
            if running:
                status_line = (
                    f"  {GREEN}● GENERATING{RESET}  "
                    f"Elapsed {str(timedelta(seconds=int(elapsed)))}  │  "
                    f"Queue: {n_pending} pending{clip_info}"
                )
            else:
                status_line = (
                    f"  {YELLOW}◌ IDLE{RESET}  Queue: {n_pending} pending{clip_info}"
                )

            sys.stdout.write(status_line + "\n")

            if step_max > 0:
                eta = format_eta(elapsed, step_val, step_max)
                pct = int(100 * step_val / step_max)
                sys.stdout.write(
                    CLEAR_LINE +
                    f"  {bar(step_val, step_max)}  "
                    f"{BOLD}{pct:3d}%{RESET}  "
                    f"Step {step_val}/{step_max}  "
                    f"ETA {eta}"
                )
                if node_name:
                    sys.stdout.write(f"  {GREY}[{node_name}]{RESET}")
                sys.stdout.write("\n")
            else:
                sys.stdout.write(CLEAR_LINE + f"  {GREY}(waiting for sampler...){RESET}\n")

            if vram_str or ram_str:
                sys.stdout.write(
                    CLEAR_LINE +
                    f"  {GREY}{vram_str}  {ram_str}{RESET}\n"
                )

            sys.stdout.flush()
            # Move cursor back up to overwrite
            lines = 3 if (vram_str or ram_str) else 2
            sys.stdout.write(f"\033[{lines}A")
            time.sleep(interval)

    except KeyboardInterrupt:
        sys.stdout.write(f"\n\n{GREY}Monitor stopped. Generation continues in background.{RESET}\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Lightweight ComfyUI progress monitor")
    p.add_argument("--host", default=HOST)
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--output", default=None, help="Path to clips output dir for clip counting")
    p.add_argument("--interval", type=float, default=1.5, help="Poll interval in seconds (default: 1.5)")
    args = p.parse_args()

    clips_path = Path(args.output) if args.output else None
    watch(args.host, args.port, clips_path, args.interval)
