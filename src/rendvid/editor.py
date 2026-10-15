"""Non-destructive editor: a timeline (edit.json) over the rendered scene clips.

Exporting never touches the source clips:

1. each timeline clip -> trimmed / sped up / smoothed / reframed / color-graded
   segment, cached by a hash of its settings so unchanged clips are reused;
2. segments joined with cuts or xfade transitions;
3. text overlays + subtitles burned in, audio rebuilt (narration re-timed to
   the new edit, music or song mixed) -> final.mp4.

The previous final.mp4 is kept in versions/ on every export.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Callable

from . import audio, config, stitch, tts
from .model import Cancelled

TRANSITIONS = {"cut", "fade", "dissolve", "wipeleft", "slideleft", "circleopen", "fadeblack"}
FILTERS = {
    "none": "",
    "warm": "colorbalance=rs=.08:gs=.02:bs=-.08:rm=.06:bm=-.06",
    "cool": "colorbalance=rs=-.06:bs=.08:rm=-.04:bm=.06",
    "bw": "hue=s=0",
    "vintage": "curves=preset=vintage,vignette=PI/5",
    "vivid": "eq=saturation=1.35:contrast=1.08",
    "film": "curves=preset=medium_contrast,noise=alls=7:allf=t,vignette=PI/6",
    "dream": "gblur=sigma=1.2,eq=brightness=0.04:saturation=1.15",
}
FITS = {"crop", "blur", "pad"}
RESOLUTIONS = {"source", "720", "1080"}


def edit_path(job_dir: Path) -> Path:
    return job_dir / "edit.json"


def clip_path(job_dir: Path, scene: int) -> Path:
    return job_dir / f"scene_{scene:02d}.mp4"


def default_edit(spec: dict, job_dir: Path) -> dict:
    song = spec.get("mode") == "song" or bool(spec.get("song"))
    music = spec.get("music") or {}
    music_file = spec.get("music_file") if (song or music.get("mode") == "upload") else None
    if not song and music.get("mode") == "generate" and (job_dir / "music.wav").exists():
        music_file = "music.wav"
    preset = spec.get("preset", "balanced")
    return {
        "version": 1,
        "clips": [
            {"scene": i + 1, "trim_in": 0.0, "trim_out": None, "speed": 1.0,
             "transition": "cut", "transition_dur": 0.5}
            for i in range(len(spec["scenes"]))
        ],
        "narrations": [s.get("narration", "") for s in spec["scenes"]],
        "texts": [],
        "subtitles_on": bool(spec.get("lyric_subtitles") if song else spec.get("subtitles", True)),
        "lyrics": copy.deepcopy(spec.get("lyrics") or []),
        "audio": {
            "music_file": music_file,
            "music_volume": 1.0 if song else float(music.get("volume", 0.35)),
            "voice": None if song else spec.get("voice"),
            "voice_volume": 1.0,
        },
        "look": {"filter": "none", "brightness": 0.0, "contrast": 1.0, "saturation": 1.0},
        "export": {
            "aspect": spec.get("aspect", "16:9"),
            "fit": "crop",
            "resolution": "1080" if preset == "max" else "720" if preset == "balanced" else "source",
            "smooth": preset == "max",
            "match_song": song,
        },
    }


def load_edit(spec: dict, job_dir: Path) -> dict:
    p = edit_path(job_dir)
    if p.exists():
        return json.loads(p.read_text())
    return default_edit(spec, job_dir)


def validate(edit: dict, spec: dict, job_dir: Path) -> dict:
    """Clamp and sanitize an edit coming from the browser."""
    n = len(spec["scenes"])
    clips = []
    for c in edit.get("clips", []):
        scene = int(c.get("scene", 0))
        if not 1 <= scene <= n:
            continue
        src_len = stitch.probe_duration(clip_path(job_dir, scene)) if clip_path(job_dir, scene).exists() else 5.0
        t_in = min(max(float(c.get("trim_in") or 0), 0.0), max(src_len - 0.3, 0))
        t_out = c.get("trim_out")
        t_out = src_len if t_out in (None, "") else min(max(float(t_out), t_in + 0.3), src_len)
        clips.append({
            "scene": scene,
            "trim_in": round(t_in, 3),
            "trim_out": None if abs(t_out - src_len) < 0.01 else round(t_out, 3),
            "speed": min(max(float(c.get("speed") or 1.0), 0.25), 4.0),
            "transition": c.get("transition") if c.get("transition") in TRANSITIONS else "cut",
            "transition_dur": min(max(float(c.get("transition_dur") or 0.5), 0.1), 2.0),
        })
    if not clips:
        raise ValueError("The timeline needs at least one clip")
    texts = []
    for t in edit.get("texts", []):
        if str(t.get("text", "")).strip():
            s, e = float(t.get("start") or 0), float(t.get("end") or 0)
            texts.append({
                "text": str(t["text"])[:300], "start": max(s, 0), "end": max(e, s + 0.2),
                "position": t.get("position") if t.get("position") in ("top", "center", "bottom") else "center",
                "size": t.get("size") if t.get("size") in stitch.SIZES else "l",
                "style": "caption" if t.get("style") == "caption" else "title",
            })
    a = edit.get("audio") or {}
    music_file = a.get("music_file")
    if music_file and not (job_dir / Path(music_file).name).exists():
        music_file = None
    look = edit.get("look") or {}
    ex = edit.get("export") or {}
    narr = [str(x or "")[:400] for x in (edit.get("narrations") or [])][:n]
    narr += [""] * (n - len(narr))
    return {
        "version": 1,
        "clips": clips,
        "narrations": narr,
        "texts": texts,
        "subtitles_on": bool(edit.get("subtitles_on")),
        "lyrics": [[float(s), float(e), str(t)] for s, e, t in (edit.get("lyrics") or []) if str(t).strip()],
        "audio": {
            "music_file": Path(music_file).name if music_file else None,
            "music_volume": min(max(float(a.get("music_volume") or 0), 0.0), 2.0),
            "voice": a.get("voice") if a.get("voice") in tts.all_voices() else None,
            "voice_volume": min(max(float(a.get("voice_volume") or 1.0), 0.0), 2.0),
        },
        "look": {
            "filter": look.get("filter") if look.get("filter") in FILTERS else "none",
            "brightness": min(max(float(look.get("brightness") or 0), -0.3), 0.3),
            "contrast": min(max(float(look.get("contrast") or 1), 0.5), 1.8),
            "saturation": min(max(float(look.get("saturation") or 1), 0.0), 2.0),
        },
        "export": {
            "aspect": ex.get("aspect") if ex.get("aspect") in config.ASPECTS else spec.get("aspect", "16:9"),
            "fit": ex.get("fit") if ex.get("fit") in FITS else "crop",
            "resolution": str(ex.get("resolution")) if str(ex.get("resolution")) in RESOLUTIONS else "720",
            "smooth": bool(ex.get("smooth")),
            "match_song": bool(ex.get("match_song")),
        },
    }


# --- export --------------------------------------------------------------------

def _target_size(edit: dict, src: tuple[int, int]) -> tuple[int, int]:
    ex = edit["export"]
    aw, ah = (int(x) for x in ex["aspect"].split(":"))
    short = min(src) if ex["resolution"] == "source" else int(ex["resolution"])
    if aw >= ah:
        h = short
        w = round(short * aw / ah)
    else:
        w = short
        h = round(short * ah / aw)
    return w - w % 2, h - h % 2


def _segment_filter(c: dict, edit: dict, size: tuple[int, int]) -> str:
    w, h = size
    fps = 48 if edit["export"]["smooth"] else config.FPS
    f = [f"setpts=PTS/{c['speed']:.4f}"]
    if edit["export"]["smooth"]:  # interpolate at source size: far cheaper than at 1080p
        f.append("minterpolate=fps=48:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1")
    f.append(f"fps={fps}")
    fit = edit["export"]["fit"]
    if fit == "crop":
        f.append(f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,crop={w}:{h}")
        chain = ",".join(f)
    elif fit == "pad":
        f.append(f"scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black")
        chain = ",".join(f)
    else:  # blur: the frame itself, blurred and zoomed, fills the empty bars
        base = ",".join(f)
        chain = (f"{base},split[a][b];[a]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},"
                 f"gblur=sigma=40,eq=brightness=-0.08[bg];[b]scale={w}:{h}:force_original_aspect_ratio=decrease:"
                 f"flags=lanczos[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2")
    look = edit["look"]
    grade = [FILTERS[look["filter"]]] if FILTERS[look["filter"]] else []
    if (look["brightness"], look["contrast"], look["saturation"]) != (0.0, 1.0, 1.0):
        grade.append(f"eq=brightness={look['brightness']:.3f}:contrast={look['contrast']:.3f}:saturation={look['saturation']:.3f}")
    if max(size) > 900:
        grade.append("unsharp=5:5:0.4:5:5:0.0")
    grade.append("setsar=1,format=yuv420p")
    return chain + "," + ",".join(grade)


def _segment(job_dir: Path, c: dict, edit: dict, size: tuple[int, int]) -> tuple[Path, float]:
    src = clip_path(job_dir, c["scene"])
    src_len = stitch.probe_duration(src)
    t_out = c["trim_out"] if c["trim_out"] is not None else src_len
    key = json.dumps([c, edit["look"], edit["export"], size, src.stat().st_mtime], sort_keys=True)
    out = job_dir / "cache" / f"seg_{hashlib.sha1(key.encode()).hexdigest()[:16]}.mp4"
    if not out.exists():
        out.parent.mkdir(exist_ok=True)
        tmp = out.with_suffix(".tmp.mp4")
        stitch.run([
            "ffmpeg", "-y", "-ss", f"{c['trim_in']:.3f}", "-to", f"{t_out:.3f}", "-i", str(src),
            "-filter_complex", _segment_filter(c, edit, size), "-an",
            "-c:v", "libx264", "-crf", "16", "-preset", "medium", str(tmp),
        ])
        tmp.rename(out)
    return out, stitch.probe_duration(out)


def _join(segs: list[tuple[Path, float]], clips: list[dict], out: Path) -> list[float]:
    """Join segments with cuts/xfades. Returns each clip's start time in the output."""
    cmd = ["ffmpeg", "-y"]
    for p, _ in segs:
        cmd += ["-i", str(p)]
    # Common timebase on every input: concat and xfade refuse mismatched ones.
    fc = [f"[{k}:v]settb=AVTB,setpts=PTS-STARTPTS[s{k}]" for k in range(len(segs))]
    acc, acc_len, starts = "s0", segs[0][1], [0.0]
    for k in range(1, len(segs)):
        tr = clips[k - 1]["transition"]
        seg_len = segs[k][1]
        nxt = f"j{k}"
        if tr == "cut":
            fc.append(f"[{acc}][s{k}]concat=n=2:v=1:a=0,settb=AVTB[{nxt}]")
            starts.append(acc_len)
            acc_len += seg_len
        else:
            d = min(clips[k - 1]["transition_dur"], acc_len / 2, seg_len / 2)
            off = acc_len - d
            fc.append(f"[{acc}][s{k}]xfade=transition={tr}:duration={d:.3f}:offset={off:.3f},settb=AVTB[{nxt}]")
            starts.append(off)
            acc_len = off + seg_len
        acc = nxt
    cmd += ["-filter_complex", ";".join(fc), "-map", f"[{acc}]"]
    cmd += ["-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", str(out)]
    stitch.run(cmd)
    return starts


def export(
    spec: dict, job_dir: Path, report: Callable[[str, float, str], None], should_cancel: Callable[[], bool],
    *, base: float = 0.0,
) -> dict:
    """Render edit.json to final.mp4. Progress is mapped onto [base, 1]."""
    t0 = time.time()
    edit = load_edit(spec, job_dir)
    song = spec.get("mode") == "song" or bool(spec.get("song"))

    def rep(frac: float, msg: str) -> None:
        report("assemble", base + (1 - base) * frac, msg)

    def check() -> None:
        if should_cancel():
            raise Cancelled()

    clips = edit["clips"]
    missing = [c["scene"] for c in clips if not clip_path(job_dir, c["scene"]).exists()]
    if missing:
        raise RuntimeError(f"Scene clips missing: {missing}. Redo those scenes first.")
    w0, h0 = stitch._video_size(clip_path(job_dir, clips[0]["scene"]))
    size = _target_size(edit, (w0, h0))

    segs = []
    for k, c in enumerate(clips):
        check()
        rep(0.7 * k / len(clips), f"Preparing clip {k + 1}/{len(clips)}…")
        segs.append(_segment(job_dir, c, edit, size))
    check()
    rep(0.72, "Joining clips…")
    video = job_dir / "video_edit.mp4"
    starts = _join(segs, clips, video)
    vlen = stitch.probe_duration(video)
    check()

    # Audio + subtitles.
    overlays: list[dict] = []
    narration_wav = None
    a = edit["audio"]
    music = job_dir / a["music_file"] if a["music_file"] else None
    if song:
        if edit["subtitles_on"]:
            overlays += stitch.cue_overlays(edit["lyrics"])
    else:
        # A line may span several clips (Studio shots): its room runs until
        # the next clip that has narration, or the end of the video.
        lines, marks = [], [k for k, c in enumerate(clips) if edit["narrations"][c["scene"] - 1].strip()]
        for j, k in enumerate(marks):
            end = starts[marks[j + 1]] if j + 1 < len(marks) else vlen
            lines.append((starts[k], end - starts[k], edit["narrations"][clips[k]["scene"] - 1]))
        if a["voice"] and lines:
            rep(0.8, "Recording voice-over…")
            narration_wav = job_dir / "narration_edit.wav"
            cues = audio.synth_narration(lines, a["voice"], vlen, narration_wav, job_dir / "tts")
            if edit["subtitles_on"]:
                overlays += stitch.cue_overlays(cues)
    overlays += edit["texts"]
    check()

    duration = None
    if song and music and edit["export"]["match_song"]:
        duration = stitch.probe_duration(music)
    rep(0.88, "Adding text and audio…")
    tmp = job_dir / "final.tmp.mp4"
    stitch.finalize(
        video, tmp,
        narration=narration_wav, music=music, music_volume=a["music_volume"],
        voice_volume=a["voice_volume"], overlays=overlays, workdir=job_dir,
        duration=duration, song=song,
    )
    for p in job_dir.glob("cap_*.png"):
        p.unlink()

    final = job_dir / "final.mp4"
    if final.exists():
        vdir = job_dir / "versions"
        vdir.mkdir(exist_ok=True)
        shutil.move(final, vdir / f"v{len(list(vdir.glob('v*.mp4'))) + 1:02d}_{time.strftime('%H%M%S')}.mp4")
    tmp.rename(final)
    _prune_cache(job_dir, {p.name for p, _ in segs})
    rep(1.0, "Done")
    return {"export": time.time() - t0, "length": stitch.probe_duration(final)}


def _prune_cache(job_dir: Path, keep: set[str], max_files: int = 120) -> None:
    files = sorted((job_dir / "cache").glob("seg_*.mp4"), key=lambda p: p.stat().st_mtime)
    for p in files[: max(0, len(files) - max_files)]:
        if p.name not in keep:
            p.unlink()
