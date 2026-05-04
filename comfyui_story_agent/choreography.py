"""
Choreography — Camera moves, animation styles, and prompt engineering

Provides structured prompt composition for LTX 2.3 video generation,
combining scene descriptions with camera movements and animation styles
into optimized prompts.
"""

# ── Camera Movements ──────────────────────────────────────────────────
# Each maps to a prompt fragment injected into the scene description.
# LTX 2.3 responds well to natural language camera direction.

CAMERA_MOVES = {
    "static": "",
    "pan_left": "smooth camera pan to the left",
    "pan_right": "smooth camera pan to the right",
    "zoom_in": "camera slowly zooms in",
    "zoom_out": "camera slowly zooms out",
    "dolly_forward": "camera moves forward through the scene",
    "dolly_back": "camera pulls back from the scene",
    "tilt_up": "camera tilts upward",
    "tilt_down": "camera tilts downward",
    "orbit_cw": "camera orbits clockwise around the subject",
    "orbit_ccw": "camera orbits counterclockwise around the subject",
    "crane_up": "camera rises upward in a crane shot",
    "crane_down": "camera descends in a crane shot",
    "tracking": "camera follows the subject movement, tracking shot",
    "handheld": "slight handheld camera movement, cinematic shake",
    "whip_pan": "fast whip pan across the scene, motion blur",
    "push_in": "dramatic camera push in toward the subject",
    "pull_out": "camera pulls out revealing the full scene",
    "dutch_angle": "tilted dutch angle shot, dramatic perspective",
    "rack_focus": "rack focus shift between foreground and background",
}

# ── Animation Styles ──────────────────────────────────────────────────
# Global style prompts prepended to every shot's scene description.
# The "comic" style is tuned for classic cartoon animation (Tex Avery,
# Hanna-Barbera, Looney Tunes aesthetic).

ANIMATION_STYLES = {
    "comic": (
        "classic hand-drawn cartoon animation, bold black outlines, "
        "flat cel shading, vibrant saturated colors, exaggerated "
        "character proportions, squash and stretch motion, clean "
        "vector-like linework, painted backgrounds, retro cartoon "
        "style like classic Hanna-Barbera and Tex Avery animation"
    ),
    "anime": (
        "2D anime style animation, cel shaded, vibrant colors, "
        "detailed character designs, dynamic camera angles, "
        "Studio Ghibli quality backgrounds"
    ),
    "pixar": (
        "3D Pixar-style animation, soft subsurface scattering lighting, "
        "detailed textures, physically based rendering, cinematic "
        "depth of field, warm color palette"
    ),
    "watercolor": (
        "watercolor animation style, flowing paint effects, soft edges, "
        "translucent color washes, paper texture visible, artistic "
        "and dreamlike quality"
    ),
    "stop_motion": (
        "stop motion animation style, handcrafted feel, clay texture, "
        "visible fingerprints, miniature set design, warm practical "
        "lighting, Laika Studios quality"
    ),
    "sketch": (
        "pencil sketch animation, hand-drawn aesthetic, cross-hatching, "
        "loose expressive linework, charcoal texture, monochrome "
        "with selective color accents"
    ),
    "retro_cartoon": (
        "classic 1950s cartoon animation, limited animation style, "
        "UPA-inspired flat design, geometric shapes, mid-century "
        "modern color palette, stylized backgrounds"
    ),
    "noir": (
        "film noir animation style, high contrast chiaroscuro lighting, "
        "dramatic shadows, venetian blind patterns, monochrome with "
        "selective warm accents, detective story atmosphere"
    ),
    "storybook": (
        "children's storybook illustration animation, soft pastel colors, "
        "gentle brush strokes, whimsical character designs, warm and "
        "inviting, picture book quality"
    ),
    "comic_book": (
        "motion comic style, bold ink outlines, halftone dot shading, "
        "dynamic panel compositions, speech bubble ready, vivid "
        "primary colors, Marvel/DC comic book aesthetic"
    ),
}

# ── Negative Prompts ──────────────────────────────────────────────────
# Style-specific negative prompts to avoid common artifacts.

_NEGATIVE_BASE = (
    "blurry, low quality, distorted, deformed, ugly, watermark, "
    "text overlay, compression artifacts, noise, grain"
)

_NEGATIVE_BY_STYLE = {
    "comic": (
        f"{_NEGATIVE_BASE}, photorealistic, 3D render, CGI, "
        "live action, realistic skin texture, motion capture, "
        "uncanny valley"
    ),
    "anime": (
        f"{_NEGATIVE_BASE}, photorealistic, western cartoon, "
        "3D render, clay, stop motion"
    ),
    "pixar": (
        f"{_NEGATIVE_BASE}, 2D, flat, hand-drawn, anime, "
        "stop motion, uncanny valley"
    ),
    "watercolor": (
        f"{_NEGATIVE_BASE}, photorealistic, sharp edges, "
        "digital art, 3D render, flat colors"
    ),
    "stop_motion": (
        f"{_NEGATIVE_BASE}, 2D animation, smooth motion, "
        "photorealistic, CGI, digital"
    ),
    "sketch": (
        f"{_NEGATIVE_BASE}, photorealistic, color, vibrant, "
        "3D render, smooth shading"
    ),
    "noir": (
        f"{_NEGATIVE_BASE}, colorful, bright, cheerful, "
        "cartoon, anime, childish"
    ),
}

# ── Shot Types ────────────────────────────────────────────────────────
# Common cinematographic shot descriptions.

SHOT_TYPES = {
    "extreme_wide": "extreme wide shot, vast landscape",
    "wide": "wide shot, full scene visible",
    "medium_wide": "medium wide shot, character from knees up",
    "medium": "medium shot, character from waist up",
    "medium_close": "medium close-up, character from chest up",
    "close_up": "close-up shot, face fills frame",
    "extreme_close": "extreme close-up, detail shot",
    "over_shoulder": "over-the-shoulder shot",
    "bird_eye": "bird's eye view, looking straight down",
    "worm_eye": "worm's eye view, looking straight up",
    "pov": "point of view shot, first person perspective",
    "two_shot": "two-shot, two characters in frame",
    "establishing": "establishing shot, setting the scene",
}

# ── Action Verbs for Animation ────────────────────────────────────────
# Expressive motion descriptors that LTX responds well to.

MOTION_DESCRIPTORS = {
    "walk": "walking with natural stride",
    "run": "running with dynamic motion",
    "jump": "jumping with squash and stretch",
    "turn": "turning to face a new direction",
    "gesture": "gesturing expressively with hands",
    "look": "looking around with curiosity",
    "react": "reacting with exaggerated surprise",
    "fall": "falling with comedic timing",
    "fly": "flying through the air gracefully",
    "dance": "dancing with rhythmic movement",
    "fight": "action fighting sequence",
    "transform": "transforming shape dramatically",
    "explode": "explosive action with debris",
    "melt": "melting and morphing fluidly",
    "grow": "growing larger in size",
    "shrink": "shrinking smaller in size",
    "spin": "spinning rapidly in place",
    "bounce": "bouncing with elastic energy",
    "slide": "sliding across the surface",
    "sneak": "sneaking with exaggerated tip-toe",
}


# ── Prompt Composition ────────────────────────────────────────────────

def compose_prompt(
    scene_description: str,
    camera: str = "static",
    style: str = "comic",
    style_prompt: str = "",
    shot_type: str = "",
    motion: str = "",
) -> str:
    """
    Combine scene + camera + style into an optimized LTX prompt.

    Args:
        scene_description: The core scene/action description
        camera: Camera movement key from CAMERA_MOVES
        style: Animation style key from ANIMATION_STYLES
        style_prompt: Override/additional style text (prepended)
        shot_type: Shot type key from SHOT_TYPES
        motion: Motion descriptor key from MOTION_DESCRIPTORS

    Returns:
        Composed prompt string optimized for LTX 2.3
    """
    parts = []

    # Style first (sets the visual tone)
    if style_prompt:
        parts.append(style_prompt)
    elif style in ANIMATION_STYLES:
        parts.append(ANIMATION_STYLES[style])

    # Shot type
    if shot_type and shot_type in SHOT_TYPES:
        parts.append(SHOT_TYPES[shot_type])

    # Core scene description
    parts.append(scene_description.strip())

    # Motion descriptor
    if motion and motion in MOTION_DESCRIPTORS:
        parts.append(MOTION_DESCRIPTORS[motion])

    # Camera movement (appended last — LTX treats this as direction)
    cam_text = CAMERA_MOVES.get(camera, "")
    if cam_text:
        parts.append(cam_text)

    return ", ".join(filter(None, parts))


def build_negative_prompt(style: str = "comic") -> str:
    """Get the appropriate negative prompt for a given style."""
    return _NEGATIVE_BY_STYLE.get(style, _NEGATIVE_BASE)


def list_cameras() -> list[str]:
    """Return all available camera move names."""
    return list(CAMERA_MOVES.keys())


def list_styles() -> list[str]:
    """Return all available animation style names."""
    return list(ANIMATION_STYLES.keys())


def list_shot_types() -> list[str]:
    """Return all available shot type names."""
    return list(SHOT_TYPES.keys())


def list_motions() -> list[str]:
    """Return all available motion descriptor names."""
    return list(MOTION_DESCRIPTORS.keys())
