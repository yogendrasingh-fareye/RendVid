"""ffmpeg assembly: scene clips -> one video -> upscale/interpolate -> subtitles + audio."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont, features

from . import config

DEVANAGARI_FONTS = [
    "/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttc",
    "/System/Library/Fonts/Kohinoor.ttc",
    "/System/Library/Fonts/Supplemental/DevanagariMT.ttc",
]

# "balanced" output size; "max" uses ASPECTS[...]["final"].
HD = {"16:9": (1280, 720), "9:16": (720, 1280), "1:1": (720, 720)}


def run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {p.stderr[-2000:]}")


def write_clip(frames: list[Image.Image], path: Path, fps: int = config.FPS) -> None:
    w = imageio.get_writer(
        path, fps=fps, codec="libx264", quality=None, macro_block_size=1,
        pixelformat="yuv420p", ffmpeg_params=["-crf", "14", "-preset", "medium"],
    )
    for f in frames:
        w.append_data(np.asarray(f.convert("RGB")))
    w.close()


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    ).stdout.strip()
    return float(out or 0)


def build_video(clips: list[Path], out: Path, *, size: tuple[int, int] | None, interpolate: bool) -> None:
    """Concatenate clips; optionally motion-interpolate to 48 fps and upscale."""
    lst = out.with_suffix(".txt")
    lst.write_text("".join(f"file '{c.resolve()}'\n" for c in clips))
    filters = []
    if interpolate:  # interpolate before upscaling: ~4x fewer pixels to search
        filters.append("minterpolate=fps=48:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1")
    if size:
        filters.append(f"scale={size[0]}:{size[1]}:flags=lanczos,unsharp=5:5:0.5:5:5:0.0")
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst)]
    if filters:
        cmd += ["-vf", ",".join(filters)]
    cmd += ["-c:v", "libx264", "-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p", "-an", str(out)]
    run(cmd)
    lst.unlink(missing_ok=True)


def _font_for(text: str, size: int) -> ImageFont.FreeTypeFont:
    if re.search(r"[ऀ-ॿ]", text):
        for f in DEVANAGARI_FONTS:
            if Path(f).exists():
                return ImageFont.truetype(f, size, layout_engine=ImageFont.Layout.RAQM if features.check("raqm") else ImageFont.Layout.BASIC)
    return ImageFont.truetype(config.FONT, size)


SIZES = {"s": 0.038, "m": 0.05, "l": 0.075, "xl": 0.11}


def render_text(text: str, width: int, height: int, path: Path, *, size: str = "m", style: str = "caption") -> None:
    """Transparent full-width PNG of wrapped text.

    ``caption``: white with black outline. ``title``: white on a translucent
    dark band, for title cards and lower thirds.
    """
    scale = 1.18 if re.search(r"[\u0900-\u097F]", text) else 1.0  # Devanagari reads small
    fs = max(16, int(min(width, height) * SIZES.get(size, SIZES["m"]) * scale))
    font = _font_for(text, fs)
    stroke = max(2, fs // 12) if style == "caption" else 0
    d = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    measure = lambda t: d.textlength(t, font=font) + 2 * stroke  # noqa: E731
    lines = []
    for para in text.splitlines() or [""]:
        cur = ""
        for word in para.split():  # greedy wrap to 88% of the frame width
            trial = f"{cur} {word}".strip()
            if cur and measure(trial) > width * 0.88:
                lines.append(cur)
                cur = word
            else:
                cur = trial
        lines.append(cur)
    lh = int(fs * 1.3)
    img = Image.new("RGBA", (width, lh * len(lines) + fs), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if style == "title":
        widest = max(measure(ln) for ln in lines)
        pad = fs * 0.6
        d.rounded_rectangle(
            ((width - widest) / 2 - pad, fs * 0.15, (width + widest) / 2 + pad, img.height - fs * 0.15),
            radius=fs // 3, fill=(0, 0, 0, 150),
        )
    for k, line in enumerate(lines):
        x = (width - measure(line)) / 2 + stroke
        d.text((x, fs // 2 + k * lh), line, font=font, fill="white", stroke_width=stroke, stroke_fill="black")
    img.save(path)


def cue_overlays(cues) -> list[dict]:
    """Subtitle cues -> overlay dicts for ``finalize``."""
    return [{"start": s, "end": e, "text": t, "position": "bottom", "size": "m", "style": "caption"} for s, e, t in cues]


def finalize(
    video: Path,
    out: Path,
    *,
    narration: Path | None,
    music: Path | None,
    music_volume: float,
    overlays: list[dict],
    workdir: Path,
    duration: float | None = None,
    song: bool = False,
    voice_volume: float = 1.0,
) -> None:
    """Burn text overlays and mix voice + (ducked, looped, faded) music.

    ``overlays``: dicts with start, end, text, position (top/center/bottom),
    size (s/m/l/xl), style (caption/title). ``duration`` forces the output
    length (video is trimmed, or its last frame held). ``song`` plays the
    music as-is: no loop, no fade-in; it fades out only if cut short.
    """
    vdur = probe_duration(video)
    dur = duration or vdur
    w, h = _video_size(video)
    cmd = ["ffmpeg", "-y", "-i", str(video)]
    fc: list[str] = []
    idx = 1

    vlabel = "0:v"
    reencode = bool(overlays)
    if dur > vdur + 0.02:  # video a bit short: hold the last frame
        fc.append(f"[0:v]tpad=stop_mode=clone:stop_duration={dur - vdur + 0.1:.3f}[vpad]")
        vlabel, reencode = "vpad", True

    # One transparent PNG per overlay, shown during its time window.
    ypos = {"top": f"{int(h * 0.06)}", "center": "(H-h)/2", "bottom": f"H-h-{int(h * 0.07)}"}
    for i, o in enumerate(overlays):
        png = workdir / f"cap_{i:03d}.png"
        render_text(o["text"], w, h, png, size=o.get("size", "m"), style=o.get("style", "caption"))
        cmd += ["-i", str(png)]
        y = ypos.get(o.get("position", "bottom"), ypos["bottom"])
        nxt = f"v{i}"
        fc.append(f"[{vlabel}][{idx}:v]overlay=0:{y}:enable='between(t,{o['start']:.3f},{o['end']:.3f})'[{nxt}]")
        vlabel, idx = nxt, idx + 1

    alabel = None
    if narration:
        cmd += ["-i", str(narration)]
        ni = idx
        idx += 1
    if music:
        cmd += (["-i", str(music)] if song else ["-stream_loop", "-1", "-i", str(music)])
        mi = idx
        idx += 1
        if song:
            cut_short = probe_duration(music) > dur + 0.1
            fades = f",afade=t=out:st={max(dur - 2.5, 0):.3f}:d=2.5" if cut_short else ""
        else:
            fades = f",afade=t=in:d=1.5,afade=t=out:st={max(dur - 2.5, 0):.3f}:d=2.5"
        fc.append(
            f"[{mi}:a]atrim=0:{dur:.3f},asetpts=PTS-STARTPTS,aresample=48000,"
            f"volume={music_volume:.2f}{fades}[mus]"
        )
    if narration and music:
        fc.append(f"[{ni}:a]aresample=48000,volume={voice_volume:.2f},apad,atrim=0:{dur:.3f},asplit=2[vo][vosc]")
        fc.append("[mus][vosc]sidechaincompress=threshold=0.02:ratio=10:attack=15:release=500[duck]")
        fc.append("[duck][vo]amix=inputs=2:normalize=0:duration=first,loudnorm=I=-14:TP=-1.5:LRA=11[aout]")
        alabel = "aout"
    elif narration:
        fc.append(f"[{ni}:a]aresample=48000,volume={voice_volume:.2f},apad,atrim=0:{dur:.3f},loudnorm=I=-14:TP=-1.5:LRA=11[aout]")
        alabel = "aout"
    elif music:
        alabel = "mus"

    if fc:
        cmd += ["-filter_complex", ";".join(fc)]
    cmd += ["-map", f"[{vlabel}]" if vlabel != "0:v" else "0:v"]
    if alabel:
        cmd += ["-map", f"[{alabel}]", "-c:a", "aac", "-b:a", "192k"]
    if reencode:
        cmd += ["-c:v", "libx264", "-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p"]
    else:
        cmd += ["-c:v", "copy"]
    cmd += ["-t", f"{dur:.3f}", "-movflags", "+faststart", str(out)]
    run(cmd)


def _video_size(path: Path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    ).stdout.strip()
    w, h = out.split(",")
    return int(w), int(h)
