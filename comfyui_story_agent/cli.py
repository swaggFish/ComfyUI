"""
CLI — Command-line interface for the Video Story Agent

Commands:
  generate   — Generate a full story sequence from a storyboard YAML
  template   — Create a starter storyboard YAML template
  shot       — Generate a single test shot
  assemble   — Assemble existing clips with transitions
  extract    — Extract a frame from a video clip
  review     — Run the Director agent on existing clips for QA
  list       — List available cameras, styles, transitions, etc.
"""

import argparse
import logging
import sys
import json
import os
from pathlib import Path

from . import __version__, DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_FRAMES


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="comfyui_story_agent",
        description=(
            "ComfyUI Video Story Agent — LTX 2.3 FLF2V Sequence Generator\n"
            "Generate sequences of continuity-linked video clips."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="ComfyUI server host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=8188,
        help="ComfyUI server port (default: 8188)"
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ── generate ──────────────────────────────────────────────────────
    gen = sub.add_parser(
        "generate",
        help="Generate a full story from a storyboard file"
    )
    gen.add_argument(
        "--storyboard", "-s", required=True,
        help="Path to storyboard YAML or CSV file"
    )
    gen.add_argument(
        "--output", "-o", default="./output/story",
        help="Output directory (default: ./output/story)"
    )
    gen.add_argument(
        "--assemble", action="store_true",
        help="Also assemble final video with transitions"
    )
    gen.add_argument(
        "--hybrid", action="store_true",
        help="Enable Hybrid Mode (Gemini Veo 3 + LTX 2.3) for SOTA consistency"
    )
    gen.add_argument(
        "--shot", type=int, default=None,
        help="Generate only a specific shot ID from the storyboard"
    )
    gen.add_argument(
        "--director", action="store_true", default=True,
        help="Enable Director QA agent (default: enabled if GEMINI_API_KEY set)"
    )
    gen.add_argument(
        "--no-director", action="store_true",
        help="Disable Director QA agent"
    )
    gen.add_argument(
        "--max-retries", type=int, default=3,
        help="Max Director retries per shot (default: 3)"
    )

    # ── template ──────────────────────────────────────────────────────
    tpl = sub.add_parser(
        "template",
        help="Create a starter storyboard template"
    )
    tpl.add_argument(
        "--shots", "-n", type=int, default=3,
        help="Number of shots (default: 3)"
    )
    tpl.add_argument(
        "--style", default="comic",
        help="Animation style (default: comic)"
    )
    tpl.add_argument(
        "--title", default="My Story",
        help="Story title"
    )
    tpl.add_argument(
        "--output", "-o", default="storyboard.yaml",
        help="Output file path (default: storyboard.yaml)"
    )
    tpl.add_argument(
        "--format", choices=["yaml", "csv"], default="yaml",
        help="Output format (default: yaml)"
    )

    # ── shot ──────────────────────────────────────────────────────────
    sht = sub.add_parser(
        "shot",
        help="Generate a single test shot"
    )
    sht.add_argument(
        "--prompt", "-p", required=True,
        help="Scene description"
    )
    sht.add_argument(
        "--frames", type=int, default=DEFAULT_FRAMES,
        help=f"Frame count, must be 8n+1 (default: {DEFAULT_FRAMES})"
    )
    sht.add_argument(
        "--width", type=int, default=DEFAULT_WIDTH,
        help=f"Video width (default: {DEFAULT_WIDTH})"
    )
    sht.add_argument(
        "--height", type=int, default=DEFAULT_HEIGHT,
        help=f"Video height (default: {DEFAULT_HEIGHT})"
    )
    sht.add_argument(
        "--style", default="comic",
        help="Animation style (default: comic)"
    )
    sht.add_argument(
        "--camera", default="static",
        help="Camera movement (default: static)"
    )
    sht.add_argument(
        "--seed", type=int, default=-1,
        help="Random seed (-1 for random)"
    )
    sht.add_argument(
        "--first-frame", default=None,
        help="Path to first frame image (enables I2V mode)"
    )
    sht.add_argument(
        "--last-frame", default=None,
        help="Path to last frame image (enables FLF2V mode)"
    )
    sht.add_argument(
        "--output", "-o", default="./output/single_shot",
        help="Output directory"
    )
    sht.add_argument(
        "--hybrid", action="store_true",
        help="Enable Hybrid Mode (Gemini Veo 3 + LTX 2.3) for SOTA consistency"
    )
    sht.add_argument(
        "--director", action="store_true", default=False,
        help="Enable Director QA agent for this shot"
    )
    sht.add_argument(
        "--max-retries", type=int, default=3,
        help="Max Director retries (default: 3)"
    )

    # ── assemble ──────────────────────────────────────────────────────
    asm = sub.add_parser(
        "assemble",
        help="Assemble existing clips with transitions"
    )
    asm.add_argument(
        "--clips", "-c", required=True,
        help="Directory containing clip files, or comma-separated paths"
    )
    asm.add_argument(
        "--transition", "-t", default="crossfade",
        help="Transition type (default: crossfade)"
    )
    asm.add_argument(
        "--duration", "-d", type=float, default=0.5,
        help="Transition duration in seconds (default: 0.5)"
    )
    asm.add_argument(
        "--output", "-o", default="./output/assembled.mp4",
        help="Output video path"
    )
    asm.add_argument(
        "--fps", type=int, default=25,
        help="Target FPS (default: 25)"
    )

    # ── extract ───────────────────────────────────────────────────────
    ext = sub.add_parser(
        "extract",
        help="Extract a frame from a video clip"
    )
    ext.add_argument(
        "--video", "-i", required=True,
        help="Path to video file"
    )
    ext.add_argument(
        "--frame", default="last",
        help="Which frame: 'first', 'last', or a number (default: last)"
    )
    ext.add_argument(
        "--output", "-o", default=None,
        help="Output image path (auto-generated if omitted)"
    )

    # ── review ─────────────────────────────────────────────────────────
    rev = sub.add_parser(
        "review",
        help="Run the Director agent on existing clips for QA"
    )
    rev.add_argument(
        "--clips", "-c", required=True,
        help="Comma-separated list of clip paths, or directory containing clips"
    )
    rev.add_argument(
        "--prompt", "-p", default="",
        help="Scene description (for context in evaluation)"
    )
    rev.add_argument(
        "--output", "-o", default="./output/director_review",
        help="Directory to save evidence frames and report"
    )

    # ── list ──────────────────────────────────────────────────────────
    lst = sub.add_parser(
        "list",
        help="List available options"
    )
    lst.add_argument(
        "what", choices=["cameras", "styles", "transitions",
                         "shots", "motions", "skills"],
        help="What to list"
    )

    args = parser.parse_args(argv)

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command is None:
        parser.print_help()
        return 0

    # Dispatch commands
    try:
        if args.command == "generate":
            return _cmd_generate(args)
        elif args.command == "template":
            return _cmd_template(args)
        elif args.command == "shot":
            return _cmd_shot(args)
        elif args.command == "assemble":
            return _cmd_assemble(args)
        elif args.command == "extract":
            return _cmd_extract(args)
        elif args.command == "review":
            return _cmd_review(args)
        elif args.command == "list":
            return _cmd_list(args)
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
        return 130
    except Exception as e:
        logging.getLogger(__name__).error(f"Error: {e}", exc_info=True)
        return 1

    return 0


# ── Command Implementations ──────────────────────────────────────────

def _cmd_generate(args):
    from .comfyui_client import ComfyUIClient
    from .shot_planner import StoryBoard
    from .sequence_engine import SequenceEngine
    from .transitions import assemble_sequence

    # Load storyboard
    path = args.storyboard
    if path.endswith(".csv"):
        sb = StoryBoard.from_csv(path)
    else:
        sb = StoryBoard.from_yaml(path)

    sb.output_dir = args.output

    # Filter by shot ID if requested
    if args.shot is not None:
        target_shot = next((s for s in sb.shots if s.shot_id == args.shot), None)
        if not target_shot:
            print(f"✗ Shot ID {args.shot} not found in storyboard")
            return 1
        
        # Check if we need to resume from a previous shot's last frame
        if not target_shot.first_frame_path:
            shot_idx = sb.shots.index(target_shot)
            if shot_idx > 0:
                prev_shot = sb.shots[shot_idx-1]
                # Look for the last frame in the output directory
                prev_frame_path = Path(args.output) / "frames" / f"shot_{prev_shot.shot_id:03d}_last.png"
                if prev_frame_path.exists():
                    target_shot.first_frame_path = str(prev_frame_path)
                    print(f"✓ Resuming from previous shot's last frame: {prev_frame_path.name}")
                else:
                    print(f"⚠ Continuity warning: Previous shot's last frame not found at {prev_frame_path}")
        
        sb.shots = [target_shot]

    print(f"╔══════════════════════════════════════════════╗")
    print(f"║  Story Agent — {sb.title:<29s}║")
    print(f"║  Shots: {len(sb.shots):<5d} | FPS: {sb.fps:<5d}             ║")
    print(f"╚══════════════════════════════════════════════╝")

    # Validate
    errors = sb.validate()
    if errors:
        for e in errors:
            print(f"  ✗ {e}")
        return 1

    # Connect to ComfyUI
    with ComfyUIClient(host=args.host, port=args.port) as client:
        if not client.is_alive():
            print("✗ ComfyUI is not running. Start with: ./launch_ltx.sh")
            return 1
        print("✓ Connected to ComfyUI")

        gemini_key = None
        if args.hybrid:
            gemini_key = os.environ.get("GEMINI_API_KEY")
            if not gemini_key:
                # Try reading from .env if not in env
                try:
                    env_path = os.path.join(os.getcwd(), "pixelle-mcp", ".env")
                    if os.path.exists(env_path):
                        try:
                            from dotenv import load_dotenv
                            load_dotenv(env_path)
                            gemini_key = os.environ.get("GEMINI_API_KEY")
                        except ImportError:
                            # Manual fallback if dotenv is missing
                            with open(env_path, "r") as f:
                                for line in f:
                                    if "GEMINI_API_KEY=" in line:
                                        gemini_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                                        os.environ["GEMINI_API_KEY"] = gemini_key
                                        break
                except Exception as e:
                    print(f"  ⚠ Error loading .env: {e}")
            
            if not gemini_key:
                print("✗ Hybrid Mode requires GEMINI_API_KEY in environment or pixelle-mcp/.env")
                return 1
            print("✓ Hybrid Mode enabled (Gemini Veo 3 + LTX 2.3)")

        # Generate sequence
        engine = SequenceEngine(client, args.output, gemini_api_key=gemini_key)
        results = engine.generate_sequence(
            sb,
            on_progress=lambda msg, cur, total: print(
                f"  [{cur}/{total}] {msg}"
            ),
        )

    print(f"\n✓ Generated {len(results)} clips")
    for r in results:
        print(f"  Shot {r.shot_id}: {r.video_path} ({r.generation_time:.1f}s)")

    # Optionally assemble
    if args.assemble:
        print("\nAssembling final video...")
        clip_infos = []
        for i, (shot, result) in enumerate(zip(sb.shots, results)):
            clip_infos.append({
                "path": result.video_path,
                "transition": shot.transition_in if i > 0 else "cut",
                "transition_duration": shot.transition_duration,
            })

        output_video = str(Path(args.output) / "final.mp4")
        assemble_sequence(clip_infos, output_video, fps=sb.fps)
        print(f"✓ Final video: {output_video}")

    return 0


def _cmd_template(args):
    from .shot_planner import StoryBoard

    sb = StoryBoard.create_template(
        n_shots=args.shots,
        style=args.style,
        title=args.title,
    )

    if args.format == "csv":
        sb.to_csv(args.output)
    else:
        sb.to_yaml(args.output)

    print(f"✓ Created {args.shots}-shot template: {args.output}")
    print(f"  Style: {args.style}")
    print(f"  Edit the file, then run:")
    print(f"  python -m comfyui_story_agent generate -s {args.output}")
    return 0


def _cmd_shot(args):
    from .comfyui_client import ComfyUIClient
    from .shot_planner import Shot, StoryBoard
    from .sequence_engine import SequenceEngine
    from .choreography import ANIMATION_STYLES, build_negative_prompt

    shot = Shot(
        shot_id=1,
        prompt=args.prompt,
        duration_frames=args.frames,
        camera=args.camera,
        width=args.width,
        height=args.height,
        seed=args.seed,
        first_frame_path=args.first_frame,
        last_frame_path=args.last_frame,
    )

    sb = StoryBoard(
        title="Single Shot",
        shots=[shot],
        style_prompt=ANIMATION_STYLES.get(args.style, ""),
        negative_prompt=build_negative_prompt(args.style),
    )

    with ComfyUIClient(host=args.host, port=args.port) as client:
        if not client.is_alive():
            print("✗ ComfyUI is not running. Start with: ./launch_ltx.sh")
            return 1

        gemini_key = None
        if args.hybrid:
            gemini_key = os.environ.get("GEMINI_API_KEY")
            if not gemini_key:
                try:
                    env_path = os.path.join(os.getcwd(), "pixelle-mcp", ".env")
                    if os.path.exists(env_path):
                        try:
                            from dotenv import load_dotenv
                            load_dotenv(env_path)
                            gemini_key = os.environ.get("GEMINI_API_KEY")
                        except ImportError:
                            with open(env_path, "r") as f:
                                for line in f:
                                    if "GEMINI_API_KEY=" in line:
                                        gemini_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                                        os.environ["GEMINI_API_KEY"] = gemini_key
                                        break
                except Exception as e:
                    pass
            if not gemini_key:
                print("✗ Hybrid Mode requires GEMINI_API_KEY in environment or pixelle-mcp/.env")
                return 1
            print("✓ Hybrid Mode enabled (Gemini Veo 3 + LTX 2.3)")

        engine = SequenceEngine(client, args.output, gemini_api_key=gemini_key)
        results = engine.generate_sequence(sb)

    result = results[0]
    print(f"✓ Generated: {result.video_path}")
    print(f"  Seed: {result.seed_used}")
    print(f"  Time: {result.generation_time:.1f}s")
    print(f"  Last frame: {result.last_frame_path}")
    return 0


def _cmd_assemble(args):
    from .transitions import assemble_sequence

    # Gather clip paths
    clips_input = args.clips
    if Path(clips_input).is_dir():
        clip_dir = Path(clips_input)
        clip_files = sorted(
            f for f in clip_dir.iterdir()
            if f.suffix in (".mp4", ".webp", ".avi", ".mkv", ".mov")
        )
    else:
        clip_files = [Path(p.strip()) for p in clips_input.split(",")]

    if not clip_files:
        print("✗ No clips found")
        return 1

    clip_infos = []
    for i, path in enumerate(clip_files):
        clip_infos.append({
            "path": str(path),
            "transition": "cut" if i == 0 else args.transition,
            "transition_duration": args.duration,
        })

    print(f"Assembling {len(clip_infos)} clips with {args.transition}...")
    assemble_sequence(clip_infos, args.output, fps=args.fps)
    print(f"✓ Output: {args.output}")
    return 0


def _cmd_extract(args):
    from .frame_utils import (
        extract_first_frame, extract_last_frame,
        extract_frame_at, extract_last_frame_from_webp,
    )

    video = args.video
    is_webp = video.endswith(".webp")

    if args.frame == "last":
        if is_webp:
            result = extract_last_frame_from_webp(video, args.output)
        else:
            result = extract_last_frame(video, args.output)
    elif args.frame == "first":
        result = extract_first_frame(video, args.output)
    else:
        result = extract_frame_at(video, int(args.frame), args.output)

    print(f"✓ Extracted frame → {result}")
    return 0


def _cmd_list(args):
    from .choreography import (
        list_cameras, list_styles, list_shot_types, list_motions,
        CAMERA_MOVES, ANIMATION_STYLES, SHOT_TYPES, MOTION_DESCRIPTORS,
    )
    from .transitions import list_transitions, TRANSITIONS

    if args.what == "cameras":
        print("Available camera movements:")
        for k in list_cameras():
            desc = CAMERA_MOVES[k]
            print(f"  {k:<16s} {desc or '(no camera movement)'}")

    elif args.what == "styles":
        print("Available animation styles:")
        for k in list_styles():
            desc = ANIMATION_STYLES[k][:60]
            print(f"  {k:<16s} {desc}...")

    elif args.what == "transitions":
        print("Available transitions:")
        for k in list_transitions():
            ffmpeg_name = TRANSITIONS[k]
            print(f"  {k:<16s} → {ffmpeg_name or 'direct cut'}")

    elif args.what == "shots":
        print("Available shot types:")
        for k in list_shot_types():
            desc = SHOT_TYPES[k]
            print(f"  {k:<16s} {desc}")

    elif args.what == "motions":
        print("Available motion descriptors:")
        for k in list_motions():
            desc = MOTION_DESCRIPTORS[k]
            print(f"  {k:<16s} {desc}")

    elif args.what == "skills":
        from .skills import SkillManager
        sm = SkillManager(workspace_root="/home/mikeyb/Documents/AI/ComfyUI")
        print("Available community skills (workflows):")
        for name, path in sorted(sm.skills.items()):
            print(f"  {name:<32s} ({path.parent.name})")

    return 0


def _cmd_review(args):
    """Run the Director agent on existing clips for post-hoc QA."""
    import os
    from .director import Director, extract_evidence_frames

    api_key = os.environ.get("GEMINI_API_KEY", "")

    director = Director(
        api_key=api_key,
        evidence_dir=args.output,
    )

    # Resolve clip paths
    clips_arg = args.clips
    if os.path.isdir(clips_arg):
        clip_dir = Path(clips_arg)
        clip_paths = sorted(
            list(clip_dir.rglob("*.webp")) + list(clip_dir.rglob("*.mp4"))
        )
    else:
        clip_paths = [Path(p.strip()) for p in clips_arg.split(",")]

    if not clip_paths:
        print(f"✗ No clips found in: {clips_arg}")
        return 1

    print(f"🎬 Director reviewing {len(clip_paths)} clip(s)...")
    results = []

    import time
    for i, clip_path in enumerate(clip_paths):
        if not clip_path.exists():
            print(f"  ⏭️  Skipping (not found): {clip_path}")
            continue

        evaluation = director.review_clip(
            clip_path=str(clip_path),
            prompt=args.prompt,
            shot_id=i + 1,
        )

        status = "✅ PASS" if evaluation.passed else "❌ FAIL"
        print(
            f"  {status} | Shot {i+1} | "
            f"Drift: {evaluation.drift_severity.value} | "
            f"Char: {evaluation.character_consistency:.2f} | "
            f"BG: {evaluation.background_stability:.2f} | "
            f"Phys: {evaluation.physics_coherence:.2f} | "
            f"Style: {evaluation.style_consistency:.2f}"
        )

        if evaluation.issues:
            for issue in evaluation.issues:
                print(
                    f"      ⚠️  [{issue.frame_location}] "
                    f"{issue.category}: {issue.description}"
                )

        if not evaluation.passed:
            print(
                f"      💡 Recommended: {evaluation.recommended_action.value}"
            )

        results.append({
            "clip": str(clip_path),
            "passed": evaluation.passed,
            "drift_severity": evaluation.drift_severity.value,
            "character_consistency": evaluation.character_consistency,
            "background_stability": evaluation.background_stability,
            "physics_coherence": evaluation.physics_coherence,
            "style_consistency": evaluation.style_consistency,
            "issues": [i.model_dump() for i in evaluation.issues],
            "recommended_action": evaluation.recommended_action.value,
            "summary": evaluation.summary,
        })

        # Sleep to avoid hitting Gemini free-tier rate limits
        if i < len(clip_paths) - 1:
            print(f"  ⏳ Pausing 15s to respect Google free-tier limits...")
            time.sleep(15)

    # Save report
    import json
    report_path = Path(args.output) / "director_review_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed
    print(f"\n📊 Director Review: {passed} passed, {failed} failed")
    print(f"📄 Report saved: {report_path}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
