"""Studio pipeline: narration-led, keyframe-driven story film.

  narration (TTS, cached)  ->  shot list sized to each line
  -> character portraits + one SDXL keyframe per new shot
  -> LTX animates every keyframe (continuations chain from the previous tail)
  -> edit.json (trims, chapter fades, title/end cards, subtitles, music)
  -> editor.export -> final.mp4

Every stage reuses what's already on disk, so cancel/resume and "redo scene"
only recompute what changed.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Callable

from PIL import Image, ImageOps

from . import config, editor, keyframes, tts
from .model import Cancelled, free_memory
from .render import run_job

WEIGHTS = {"voice": (0.0, 0.07), "keyframes": (0.07, 0.22), "video": (0.22, 0.90), "export": (0.90, 1.0)}
FIRST_MAX = config.CLIP_FRAMES / config.FPS  # 5.04 s
CONT_MAX = (config.CLIP_FRAMES - config.OVERLAP_FRAMES) / config.FPS  # 4.67 s
END_CARD = {"Hindi": "समाप्त", "English": "The End"}


def _frames_for(seconds: float, overlap: bool) -> int:
    need = math.ceil(seconds * config.FPS) + 2 + (config.OVERLAP_FRAMES if overlap else 0)
    return min(config.CLIP_FRAMES, max(33, 8 * math.ceil((need - 1) / 8) + 1))


def shot_list(lines: list[dict], durations: list[float]) -> list[dict]:
    """One or more clips per narration line, sized to the spoken audio."""
    scenes = []
    for i, (ln, dur) in enumerate(zip(lines, durations)):
        target = dur + (0.5 if i == 0 else 0.0) + (1.6 if i == len(lines) - 1 else 0.45)
        n = 1 if target <= FIRST_MAX else 1 + math.ceil((target - FIRST_MAX) / CONT_MAX)
        lens = [target / n] * n
        for k, length in enumerate(lens):
            first = k == 0
            keyframe = first and (i == 0 or ln.get("new_shot", True))
            scenes.append({
                "prompt": ln["visual"] if first else f"{ln['visual']} The motion continues smoothly.",
                "narration": ln["narration"] if first else "",
                "new_shot": keyframe,
                "keyframe": f"keyframes/line_{i + 1:03d}.png" if keyframe else None,
                "frames": _frames_for(length, overlap=not keyframe),
                "target": round(length, 3),
                "line": i,
                "chapter": ln.get("chapter", 0),
            })
    return scenes


def keyframe_prompt(spec: dict, ln: dict) -> str:
    looks = {c["name"]: c["look"] for c in spec.get("characters", [])}
    present = [n for n in ln.get("characters", []) if n in looks]
    # "exactly one" only for the main subject (avoids clones); others just their look.
    who = ". ".join((f"exactly one {n}: " if k == 0 else f"{n}: ") + looks[n] for k, n in enumerate(present))
    parts = [ln["visual"], who, spec.get("setting", ""), spec.get("look", ""), config.STYLES.get(spec.get("style", "none"), "")]
    return ". ".join(p.strip().rstrip(".") for p in parts if p and p.strip()) + ". masterpiece, highly detailed"


def run_studio(
    job: dict, job_dir: Path, report: Callable[[str, float, str], None],
    should_cancel: Callable[[], bool], save_spec: Callable[[dict], None],
) -> dict:
    spec = job["spec"]
    timings: dict[str, float] = {}
    t_start = time.time()

    def rep(stage: str, frac: float, msg: str) -> None:
        a, b = WEIGHTS[stage]
        report(stage, a + (b - a) * min(max(frac, 0.0), 1.0), msg)

    def check() -> None:
        if should_cancel():
            raise Cancelled()

    lines = spec["lines"]
    language = spec.get("language", "Hindi")

    # 1. Narration first: it decides every shot's length.
    t = time.time()
    rep("voice", 0.02, f"Recording narration ({len(lines)} lines)…")
    if spec.get("voice") and any(ln["narration"].strip() for ln in lines):
        paths = tts.synth_many(
            [ln["narration"] for ln in lines], spec["voice"], job_dir / "tts",
            on_progress=lambda d, n: rep("voice", d / n, f"Recording narration {d}/{n}…"),
        )
    else:
        paths = [None] * len(lines)
    # Silent beats (visual films) hold for their own `seconds`.
    durations = [tts.duration(p) if p and ln["narration"].strip() else float(ln.get("seconds") or 5)
                 for p, ln in zip(paths, lines)]
    timings["voice"] = time.time() - t
    check()

    if not spec.get("scenes"):
        spec["scenes"] = shot_list(lines, durations)
        save_spec(spec)

    # 2. Portraits + keyframes (SDXL), skipped when all exist. A crew film has
    #    the QA Critic look at every new keyframe and redraw the failures.
    kf_dir = job_dir / "keyframes"
    todo = [(i, sc) for i, sc in enumerate(spec["scenes"]) if sc.get("keyframe") and not (job_dir / sc["keyframe"]).exists()]
    if todo:
        t = time.time()
        rep("keyframes", 0.0, "Loading image model…")
        engine = keyframes.KeyframeEngine()
        try:
            rep("keyframes", 0.05, "Drawing character references…")
            refs = keyframes.make_portraits(spec.get("characters", []), config.STYLES.get(spec.get("style", "none"), ""),
                                            job_dir / "characters", int(spec.get("seed", 42)), engine, should_cancel,
                                            setting=spec.get("setting", ""))
            vw, vh = config.ASPECTS[spec["aspect"]][spec["preset"]]
            kf_dir.mkdir(exist_ok=True)

            def draw(items, label="Keyframe"):
                for k, (i, sc) in enumerate(items):
                    check()
                    ln = lines[sc["line"]]
                    # Reference image only for single-subject shots: with two
                    # subjects SDXL blends the reference into both (a crow-shaped pot).
                    present = [n for n in ln.get("characters", []) if n in refs]
                    ref = Image.open(refs[present[0]]).convert("RGB") if len(present) == 1 else None
                    seed = int(sc.get("kf_seed") or int(spec.get("seed", 42)) + sc["line"])
                    img = engine.keyframe(keyframe_prompt(spec, ln), spec["aspect"], seed, ref)
                    ImageOps.fit(img, (vw, vh), Image.LANCZOS).save(job_dir / sc["keyframe"])
                    if i == 0:
                        img.convert("RGB").save(job_dir / "thumb.jpg", quality=88)
                    rep("keyframes", 0.1 + 0.8 * (k + 1) / len(items), f"{label} {k + 1}/{len(items)}")

            draw(todo)
            if spec.get("crew"):
                from .crew import production

                def redraw(line_idx: list[int]) -> None:
                    again = []
                    for i, sc in enumerate(spec["scenes"]):
                        if sc["line"] in line_idx and sc.get("keyframe"):
                            sc["kf_seed"] = random.randint(1, 2**31 - 1)
                            sc["prompt"] = lines[sc["line"]]["visual"]
                            again.append((i, sc))
                    save_spec(spec)
                    draw(again, "Redrawing keyframe")

                rep("keyframes", 0.92, "QA Critic is checking keyframes…")
                production.review_keyframes(spec["crew"], spec, job_dir, redraw)
                save_spec(spec)
        finally:
            engine.close()
        timings["keyframes"] = time.time() - t
    check()

    # 3. Animate (render.run_job renders only the missing clips).
    def video_report(_stage: str, overall: float, msg: str) -> None:
        # run_job maps encode+scenes onto [0, 0.8]
        rep("video", overall / 0.8, msg)

    timings.update(run_job(spec, job_dir, video_report, should_cancel, assemble=False))
    check()

    # 4. Music bed (generated once), then the edit (created once: later
    #    exports keep the user's edits).
    music = spec.get("music") or {}
    if music.get("mode") == "generate" and not (job_dir / "music.wav").exists():
        from . import audio

        report("export", WEIGHTS["export"][0], "Composing music…")
        audio.generate_music(music.get("prompt") or f"gentle cinematic background score for a story, {spec.get('look', '')}",
                             job_dir / "music.wav")
    if not editor.edit_path(job_dir).exists():
        editor.edit_path(job_dir).write_text(json.dumps(default_edit(spec, job_dir), indent=2))
    free_memory()
    timings.update(editor.export(spec, job_dir, report, should_cancel, base=WEIGHTS["export"][0]))
    timings["total"] = time.time() - t_start
    return timings


def default_edit(spec: dict, job_dir: Path) -> dict:
    scenes = spec["scenes"]
    edit = editor.default_edit(spec, job_dir)
    cut = spec.get("editor") or {}
    chapter_tr = cut.get("chapter_transition", "fadeblack")
    shot_tr = cut.get("shot_transition", "dissolve")
    for k, (c, sc) in enumerate(zip(edit["clips"], scenes)):
        c["trim_out"] = sc["target"]
        nxt = scenes[k + 1] if k + 1 < len(scenes) else None
        if nxt and nxt["chapter"] != sc["chapter"]:
            c.update(transition=chapter_tr, transition_dur=0.6)
        elif nxt and nxt.get("keyframe"):
            c.update(transition=shot_tr, transition_dur=0.3)
    edit["narrations"] = [sc.get("narration", "") for sc in scenes]
    total = sum(sc["target"] for sc in scenes)
    edit["texts"] = []
    if spec.get("title_card", True) and spec.get("title"):
        edit["texts"].append({"text": spec["title"], "start": 0.2, "end": 3.4,
                              "position": "center", "size": "xl", "style": "title"})
        end_text = cut["end_text"] if "end_text" in cut else END_CARD.get(spec.get("language"), "The End")
        if end_text:
            edit["texts"].append({"text": end_text, "start": max(total - 2.2, 0),
                                  "end": total + 1, "position": "center", "size": "xl", "style": "title"})
    edit["subtitles_on"] = bool(spec.get("subtitles", True))
    edit["audio"]["voice"] = spec.get("voice")
    edit["look"]["filter"] = cut.get("filter", "none")
    music = spec.get("music") or {}
    edit["audio"]["music_volume"] = float(music.get("volume", 0.18))
    return editor.validate(edit, spec, job_dir)
