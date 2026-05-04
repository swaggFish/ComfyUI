# 🎬 ComfyUI Story Agent — Production Movie Pipeline

> **Generate reproducible, high-fidelity animated movies from YAML storyboards using LTX 2.3 22B on your RTX 5060 Ti.**

---

## 🚀 Three Ways to Use This System

### Method 1: YAML Storyboard (Full Automation) ⭐ RECOMMENDED

This is the "one command, full movie" approach. You write a YAML storyboard describing your shots, provide anchor keyframes, and the system generates and assembles everything automatically.

```bash
# Step 1: Make sure ComfyUI is running
./launch_ltx.sh

# Step 2: Generate all shots from the storyboard
./venv/bin/python3 -m comfyui_story_agent generate \
  -s comfyui_story_agent/examples/super_noah.yaml

# Step 3: Assemble the final movie with transitions
./venv/bin/python3 -m comfyui_story_agent assemble \
  --clips output/super_noah/ \
  --output output/super_noah/super_noah_FINAL.mp4 \
  --transition crossfade --duration 1.0
```

**Writing your own storyboard:**

```yaml
title: "My Movie"
style: comic
fps: 25

shots:
  - shot_id: 1
    prompt: >-
      Description of what happens in this shot. Be specific about
      characters, location, lighting, and action.
    duration_frames: 97      # 8n+1 rule (max 97 for 16GB VRAM)
    camera: push_in          # static, push_in, crane_up, orbit, etc.
    transition_in: crossfade
    transition_duration: 1.0
    seed: -1                 # -1 = random
    width: 1024
    height: 576
    first_frame_path: input/my_shot1_start.png   # Anchor keyframe
    last_frame_path: input/my_shot1_end.png      # Anchor keyframe
```

> **💡 Pro Tip:** Generate your anchor keyframes with Gemini, Grok, or any image generator. The more consistent your keyframes are, the better the interpolated animation will look.

---

### Method 2: Single Shot (Quick Test)

Test a single shot without writing a full YAML storyboard.

```bash
# Text-to-Video (no keyframes)
./venv/bin/python3 -m comfyui_story_agent shot \
  --prompt "A superhero flying over a city at sunset" \
  --frames 97 --width 1024 --height 576 \
  --output output/test_shot

# Image-to-Video (one keyframe)
./venv/bin/python3 -m comfyui_story_agent shot \
  --prompt "A boy discovers he can fly" \
  --first-frame input/my_start.png \
  --frames 97 --output output/test_shot

# First/Last Frame to Video (two keyframes) ⭐ BEST QUALITY
./venv/bin/python3 -m comfyui_story_agent shot \
  --prompt "A boy transforms into a superhero" \
  --first-frame input/start.png \
  --last-frame input/end.png \
  --frames 97 --output output/test_shot
```

---

### Method 3: Load Golden Workflows into ComfyUI GUI

If you prefer the visual ComfyUI interface for tweaking parameters:

1. Open ComfyUI in your browser: `http://localhost:8188`
2. Click the **⚙️ gear icon** → Enable **Dev Mode Options**
3. Click **Load** → Select one of these golden templates:

| Template | File | Best For |
|----------|------|----------|
| Text-to-Video | `skills/ltx_golden_t2v_v2.json` | Free-form generation |
| Image-to-Video | `skills/ltx_golden_i2v_v2.json` | Single anchor shots |
| **FLF2V** ⭐ | `skills/ltx_golden_flf2v_v2.json` | **Consistent movies** |

4. Replace the placeholder values:
   - `GOLDEN_PROMPT_PLACEHOLDER` → Your scene description
   - `GOLDEN_NEGATIVE_PLACEHOLDER` → Your negative prompt
   - `GOLDEN_FIRST_FRAME.png` → Upload your start keyframe
   - `GOLDEN_LAST_FRAME.png` → Upload your end keyframe
5. Click **Queue Prompt** ▶️

> **⚠️ Important:** Don't rewire any nodes! The golden templates have a proven architecture. Only change the text/image inputs and seed.

---

### Method 4: Via Pixelle MCP (AI Agent Integration)

The golden workflows are automatically indexed by Pixelle MCP, making them available as tools for any AI agent (Claude, Gemini, etc.):

```bash
# Make sure Pixelle is running
./pixelle-mcp/run_pixelle.sh

# The following skills are auto-discovered:
# - ltx_golden_t2v_v2
# - ltx_golden_i2v_v2
# - ltx_golden_flf2v_v2
```

---

## 📐 The Guardrail System

The pipeline has three safety layers that prevent bad output:

### Guardrail 1: Few-Shot Style Matching
Before generating, the system searches `memory.json` for proven past successes that match your shot's tags. These are injected as few-shot examples to maintain style consistency.

### Guardrail 2: Parameter Envelopes
Every parameter is clamped to safe ranges defined in `production_envelopes.json`:
- **CFG:** 1.0–4.0 (above 4.0 = "deep-fried" artifacts)
- **Frames:** 9–97 (must follow 8n+1 rule)
- **Resolution:** max 1024×576 (16GB VRAM limit)
- **Concurrent shots:** 1 (parallel = crash risk)

### Guardrail 3: Architecture Validation
Before any GPU execution, the Critic checks:
- ✅ Required nodes present (VAE, Sampler, Checkpoint)
- ✅ All node connections valid
- ✅ No dangerous hyperparameters

---

## 🕵️ The Director Agent (Automated Visual QA)

The pipeline includes a self-healing **Director Agent** that acts as an automated test suite for generated pixels.

When a shot finishes rendering, the Director:
1. **Extracts** 3 evidence frames (first, middle, last)
2. **Evaluates** them using Gemini Flash 2.0 (zero local VRAM cost) for:
   - Anatomy distortion
   - Background stability
   - Character identity drift
   - Physics coherence
3. **Corrects** mathematical parameters (CFG, seed, negative prompt) within safe envelopes if drift is detected
4. **Retries** the generation up to 3 times before accepting or scrapping

### Using the Director
The Director runs automatically during generation if `GEMINI_API_KEY` is set.

```bash
export GEMINI_API_KEY="your-google-ai-key-here"

# Generate with automated QA loop:
python -m comfyui_story_agent generate -s examples/my_movie.yaml

# Run post-hoc QA review on existing clips:
python -m comfyui_story_agent review -c output/my_movie/
```

---

## 🎥 The Movie-Making Workflow

Here's the proven process for creating a new movie:

```
1. WRITE the script (what happens in each shot)
         ↓
2. GENERATE anchor keyframes (Gemini, Grok, or any image gen)
         ↓
3. COPY keyframes to input/ directory
         ↓
4. WRITE the YAML storyboard (see examples/)
         ↓
5. RUN: python -m comfyui_story_agent generate -s my_movie.yaml
         ↓
6. REVIEW clips in output/ directory
         ↓
7. ASSEMBLE: python -m comfyui_story_agent assemble --clips output/my_movie/
         ↓
8. WATCH your movie! 🍿
```

### Character Consistency Tips
- **Always describe characters identically** in every shot prompt
  - ✅ "Noah, a 10-year-old boy with blonde hair in a blue suit"
  - ❌ "a boy" (too vague, will drift)
- **Use the same keyframe style** across all shots
- **Use FLF2V mode** (both start + end keyframes) for maximum control

---

## 📁 Available Commands

```bash
# Generate from storyboard
python -m comfyui_story_agent generate -s examples/super_noah.yaml

# Generate single shot
python -m comfyui_story_agent shot -p "description" --frames 97

# Assemble clips into movie
python -m comfyui_story_agent assemble --clips output/dir/ -o final.mp4

# Create a blank template
python -m comfyui_story_agent template --shots 6 -o my_movie.yaml

# Run Director Agent QA review on existing clips
python -m comfyui_story_agent review -c output/my_movie/

# List available transitions
python -m comfyui_story_agent list

# Export golden workflows
python -m comfyui_story_agent.export_golden_skills
```

---

## 🔧 Assembly Options

```bash
# Simple cut (no transitions)
--transition cut

# Smooth crossfade (recommended)
--transition crossfade --duration 1.0

# Dramatic fade through black
--transition fade_black --duration 1.5

# Pixel dissolve
--transition pixelize --duration 0.8

# Iris wipe (classic cartoon)
--transition circle_open --duration 0.5
```

Available transitions: `cut`, `crossfade`, `dissolve`, `wipe_left`, `wipe_right`,
`slide_left`, `slide_right`, `fade_black`, `fade_white`, `circle_open`,
`circle_close`, `radial`, `pixelize`, `diagtl`, `diagtr`

---

## ⚡ Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 12GB VRAM | **16GB VRAM (RTX 5060 Ti)** |
| RAM | 16GB | 32GB |
| Storage | 50GB free | 100GB free |
| CPU | 8 cores | 12+ cores |

---

## 🎬 Example Movies

| Movie | Shots | Runtime | Style |
|-------|-------|---------|-------|
| **Super Noah** | 12 | ~50s | 3D Cinematic |
| **Super Henrio** | 6 | ~30s | Pixel Art (Retro Crunch) |
| **Great Escape** | 5 | ~15s | Cartoon |
