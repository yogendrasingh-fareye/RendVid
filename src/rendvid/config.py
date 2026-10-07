"""Central settings: models, presets, aspect ratios, paths."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
WEB = Path(__file__).resolve().parent / "web"

# --- Models -----------------------------------------------------------------
# Text encoder / tokenizer / scheduler come from the diffusers-format repo;
# the 2B distilled transformer + VAE come from a single checkpoint file.
BASE_REPO = "Lightricks/LTX-Video-0.9.5"
CKPT_REPO = "Lightricks/LTX-Video"
CKPT_FILE = "ltxv-2b-0.9.8-distilled.safetensors"
UPSCALER_REPO = "Lightricks/ltxv-spatial-upscaler-0.9.7"

MUSICGEN_REPO = "facebook/musicgen-small"
WHISPER_MODEL = "small"

# --- Video timing -----------------------------------------------------------
FPS = 24
CLIP_FRAMES = 121  # 8n+1, ~5 s per scene
OVERLAP_FRAMES = 9  # 8n+1, tail of previous scene used to condition the next

# Distilled-model schedules (from Lightricks' reference settings).
FIRST_PASS_TIMESTEPS = [1000, 993, 987, 981, 975, 909, 725, 0.03]
REFINE_TIMESTEPS = [1000, 909, 725, 421, 0]

# Generation size (w, h) per aspect and preset; "max" is generated at half
# size and latent-upscaled 2x, so its halves must also be divisible by 32.
ASPECTS = {
    "16:9": {"fast": (640, 352), "balanced": (768, 448), "max": (1024, 576), "final": (1920, 1080)},
    "9:16": {"fast": (352, 640), "balanced": (448, 768), "max": (576, 1024), "final": (1080, 1920)},
    "1:1": {"fast": (480, 480), "balanced": (576, 576), "max": (768, 768), "final": (1080, 1080)},
}

# fast:     one pass, small, 24 fps
# balanced: one pass, medium, upscale to 720p-ish
# max:      low-res pass -> 2x latent upscale -> refine pass, then 1080p + 48 fps
PRESETS = {
    "fast": {"multiscale": False, "upscale_final": False, "interpolate": False},
    "balanced": {"multiscale": False, "upscale_final": True, "interpolate": False},
    "max": {"multiscale": True, "upscale_final": True, "interpolate": True},
}

STYLES = {
    "none": "",
    "cinematic": "cinematic film still, shallow depth of field, dramatic lighting, anamorphic lens, film grain, color graded",
    "realistic": "photorealistic, natural lighting, highly detailed, 4k footage, realistic textures",
    "anime": "anime style, cel shaded, vibrant colors, clean line art, studio anime production",
    "3d": "3D animated film style, Pixar-like rendering, soft global illumination, stylized characters",
    "documentary": "documentary footage, handheld camera, natural colors, observational style",
    "fantasy": "epic fantasy, magical atmosphere, volumetric light, rich saturated colors, painterly detail",
    "noir": "black and white film noir, high contrast, hard shadows, moody atmosphere",
    "drone": "aerial drone footage, sweeping wide shot, smooth camera movement, golden hour",
}

DEFAULT_NEGATIVE = "worst quality, inconsistent motion, blurry, jittery, distorted, deformed, watermark, text"

VOICES = {
    "af_heart": "American female (Heart)",
    "af_bella": "American female (Bella)",
    "af_nicole": "American female (Nicole)",
    "am_michael": "American male (Michael)",
    "am_fenrir": "American male (Fenrir)",
    "bf_emma": "British female (Emma)",
    "bm_george": "British male (George)",
    "hf_alpha": "Hindi female (Alpha)",
    "hf_beta": "Hindi female (Beta)",
    "hm_omega": "Hindi male (Omega)",
    "hm_psi": "Hindi male (Psi)",
}

FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


SONG_MAX_SECONDS = 360
STORY_MAX_SCENES = 16
SONG_MAX_SCENES = 80


def scene_windows(n: int) -> list[tuple[float, float]]:
    """(start, end) seconds each scene covers once overlaps are trimmed."""
    first = CLIP_FRAMES / FPS
    step = (CLIP_FRAMES - OVERLAP_FRAMES) / FPS
    out = [(0.0, first)]
    for i in range(1, n):
        s = first + (i - 1) * step
        out.append((s, s + step))
    return out


def scene_count(duration_s: float) -> int:
    """Scenes needed so that trimmed, stitched clips reach the target length."""
    total = int(-(-duration_s * FPS // 1))
    step = CLIP_FRAMES - OVERLAP_FRAMES
    return max(1, 1 + -(-(total - CLIP_FRAMES) // step))
