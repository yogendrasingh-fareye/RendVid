"""Run one job end to end: prompts -> scene clips -> audio -> final video.

Every expensive artifact is written to the job folder as soon as it exists
(scene clips, conditioning tails, narration), so an interrupted job resumes
from the first missing piece.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from PIL import Image

from . import audio, config, stitch
from .model import Cancelled, VideoEngine, encode_prompts, free_memory

# Share of the progress bar per stage.
WEIGHTS = {"encode": 0.05, "scenes": 0.75, "audio": 0.08, "assemble": 0.12}
ORDER = list(WEIGHTS)

Report = Callable[[str, float, str], None]


def _overall(stage: str, frac: float) -> float:
    before = sum(WEIGHTS[s] for s in ORDER[: ORDER.index(stage)])
    return before + WEIGHTS[stage] * min(max(frac, 0.0), 1.0)


def full_prompt(spec: dict, scene: dict) -> str:
    parts = [scene["prompt"].strip()]
    style = config.STYLES.get(spec.get("style", "none"), "")
    if style:
        parts.append(style)
    return " ".join(parts)


def run_job(
    spec: dict, job_dir: Path, report: Report, should_cancel: Callable[[], bool], *, assemble: bool = True
) -> dict:
    """Render ``spec`` into ``job_dir``. Returns timing info. Raises Cancelled.

    With ``assemble=False`` only missing scene clips are rendered (used by
    "redo scene", which then re-exports through the editor)."""
    t_start = time.time()
    timings: dict[str, float] = {}

    def rep(stage: str, frac: float, msg: str) -> None:
        report(stage, _overall(stage, frac), msg)

    def check() -> None:
        if should_cancel():
            raise Cancelled()

    scenes = spec["scenes"]
    n = len(scenes)
    preset = config.PRESETS[spec["preset"]]
    size = config.ASPECTS[spec["aspect"]][spec["preset"]]
    clip = lambda i: job_dir / f"scene_{i + 1:02d}.mp4"  # noqa: E731
    tail_dir = lambda i: job_dir / f"tail_{i + 1:02d}"  # noqa: E731
    todo = [i for i in range(n) if not clip(i).exists()]

    # 1. Prompt embeddings (text encoder loaded alone, then freed).
    embeds = {}
    if todo:
        rep("encode", 0.1, "Loading text encoder… (first run downloads ~19 GB, this can take a while)")
        t = time.time()
        negative = spec.get("negative") or config.DEFAULT_NEGATIVE
        enc = encode_prompts([full_prompt(spec, scenes[i]) for i in todo], negative)
        embeds = dict(zip(todo, enc))
        timings["encode"] = time.time() - t
    check()

    # 2. Scenes.
    init_image = None
    if spec.get("init_image"):
        init_image = Image.open(job_dir / spec["init_image"]).convert("RGB")
    if todo:
        rep("scenes", 0, "Loading video model…")
        t = time.time()
        engine = VideoEngine(multiscale=preset["multiscale"])
        timings["load_video_model"] = time.time() - t
        scene_times = []
        try:
            for i in todo:
                check()
                t = time.time()
                sc = scenes[i]
                cond_video = None
                cond_image = None
                if sc.get("keyframe"):  # Studio: animate the SDXL still
                    cond_image = Image.open(job_dir / sc["keyframe"]).convert("RGB")
                elif i == 0 or sc.get("new_shot"):
                    if i == 0:
                        cond_image = init_image
                else:
                    tails = sorted(tail_dir(i - 1).glob("*.png"))
                    cond_video = [Image.open(p).convert("RGB") for p in tails] or None

                def on_step(done: int, total: int, i=i) -> None:
                    rep("scenes", (len(scene_times) + done / total) / len(todo),
                        f"Scene {i + 1}/{n} · step {done}/{total}")

                frames = engine.render(
                    embeds[i], size,
                    seed=int(sc["seed"]) if sc.get("seed") is not None else int(spec.get("seed", 42)) + i,
                    guidance=float(spec.get("guidance", 1.0)),
                    cond_image=cond_image, cond_video=cond_video,
                    num_frames=int(sc.get("frames") or config.CLIP_FRAMES),
                    on_step=on_step, should_cancel=should_cancel,
                )
                # Save the conditioning tail first, then the (trimmed) clip:
                # the clip's existence marks the scene as complete.
                td = tail_dir(i)
                td.mkdir(exist_ok=True)
                for k, f in enumerate(frames[-config.OVERLAP_FRAMES:]):
                    f.save(td / f"{k:02d}.png")
                # Drop the overlap on every scene but the first (it duplicates
                # the conditioning tail), so scene timing matches scene_windows.
                if spec.get("mode") == "studio":  # shots are sized individually
                    keep = frames[config.OVERLAP_FRAMES:] if cond_video else frames
                else:
                    keep = frames if i == 0 else frames[config.OVERLAP_FRAMES:]
                tmp = clip(i).with_suffix(".tmp.mp4")
                stitch.write_clip(keep, tmp)
                tmp.rename(clip(i))
                if i == 0:
                    frames[0].convert("RGB").save(job_dir / "thumb.jpg", quality=88)
                scene_times.append(time.time() - t)
                rep("scenes", len(scene_times) / len(todo), f"Scene {i + 1}/{n} done")
        finally:
            engine.close()
        timings["scenes_total"] = sum(scene_times)
        timings["per_scene_avg"] = sum(scene_times) / len(scene_times)
    check()
    if not assemble:
        return timings

    # 3. Audio.
    t = time.time()
    clips = [clip(i) for i in range(n)]
    durs = [stitch.probe_duration(c) for c in clips]
    starts = [sum(durs[:i]) for i in range(n)]
    total = sum(durs)

    song = spec.get("mode") == "song"
    narration_wav, cues = None, []
    lines = [(starts[i], durs[i], scenes[i].get("narration", "")) for i in range(n)]
    if song:
        cues = [tuple(c) for c in spec.get("lyrics") or []] if spec.get("lyric_subtitles") else []
        if cues:
            _write_srt(cues, job_dir / "subtitles.srt")
    elif spec.get("voice") and any(text.strip() for *_, text in lines):
        rep("audio", 0.1, "Recording voice-over…")
        narration_wav = job_dir / "narration.wav"
        cues = audio.synth_narration(lines, spec["voice"], total, narration_wav, job_dir / "tts")
        _write_srt(cues, job_dir / "subtitles.srt")
    check()

    music = spec.get("music") or {}
    music_path = None
    if song:
        music_path = job_dir / spec["music_file"]
    elif music.get("mode") == "upload" and spec.get("music_file"):
        music_path = job_dir / spec["music_file"]
    elif music.get("mode") == "generate":
        music_path = job_dir / "music.wav"
        if not music_path.exists():
            rep("audio", 0.5, "Composing music…")
            prompt = music.get("prompt") or f"instrumental background score, {spec.get('style', 'cinematic')} mood, {spec.get('title', '')}"
            audio.generate_music(prompt, music_path)
    timings["audio"] = time.time() - t
    check()

    # 4. Assemble.
    t = time.time()
    rep("assemble", 0.05, "Stitching scenes" + (" · smoothing to 48 fps" if preset["interpolate"] else "") + "…")
    out_size = None
    if spec["preset"] == "max":
        out_size = config.ASPECTS[spec["aspect"]]["final"]
    elif preset["upscale_final"]:
        out_size = stitch.HD[spec["aspect"]]
    video = job_dir / "video_noaudio.mp4"
    stitch.build_video(clips, video, size=out_size, interpolate=preset["interpolate"])
    check()
    rep("assemble", 0.7, "Adding subtitles and audio…")
    stitch.finalize(
        video, job_dir / "final.mp4",
        narration=narration_wav,
        music=music_path,
        music_volume=1.0 if song else float(music.get("volume", 0.35)),
        overlays=stitch.cue_overlays(cues) if (song or spec.get("subtitles", True)) else [],
        workdir=job_dir,
        duration=stitch.probe_duration(music_path) if song else None,
        song=song,
    )
    for p in job_dir.glob("cap_*.png"):
        p.unlink()
    timings["assemble"] = time.time() - t
    timings["total"] = time.time() - t_start
    free_memory()
    rep("assemble", 1.0, "Done")
    return timings


def _write_srt(cues: list[tuple[float, float, str]], path: Path) -> None:
    def ts(s: float) -> str:
        ms = int(round(s * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"

    path.write_text("".join(f"{i}\n{ts(s)} --> {ts(e)}\n{t}\n\n" for i, (s, e, t) in enumerate(cues, 1)))

