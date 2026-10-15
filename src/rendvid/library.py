"""Characters & Resources: reusable assets the crew can cast in any film.

An asset is a character, prop, location, style or music track, stored as
outputs/.library/<id>/asset.json with images/ (or audio/) beside it. Characters
and styles can be trained into a LoRA (outputs/.library/<id>/lora.safetensors)
so the image model draws them identically in every shot.

GPU work (drawing reference images, training) runs as a job on the normal
render queue (spec mode "asset"), so it never competes with a film render.
"""

from __future__ import annotations

import json
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Callable

from . import config

LIB_DIR = config.OUTPUTS / ".library"
KINDS = ("character", "prop", "location", "style", "music")
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}
TRAINABLE = ("character", "style")
_lock = threading.Lock()


def _dir(aid: str) -> Path:
    return LIB_DIR / aid


def _url(aid: str, sub: str, name: str) -> str:
    return f"/outputs/.library/{aid}/{sub}/{name}"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:24] or "asset"


def load(aid: str) -> dict | None:
    p = _dir(aid) / "asset.json"
    if not p.exists():
        return None
    a = json.loads(p.read_text())
    d = _dir(aid)
    a["images"] = [{"file": f.name, "url": _url(aid, "images", f.name),
                    "source": "generated" if f.name.startswith("gen_") else "upload"}
                   for f in sorted((d / "images").glob("*")) if f.suffix.lower() in IMAGE_EXT]
    a["audio"] = [{"file": f.name, "url": _url(aid, "audio", f.name)}
                  for f in sorted((d / "audio").glob("*")) if f.suffix.lower() in AUDIO_EXT]
    cover = a.get("cover_file")
    if not cover or not (d / "images" / cover).exists():
        cover = a["images"][0]["file"] if a["images"] else None
    a["cover"] = _url(aid, "images", cover) if cover else None
    a["lora"]["ready_path"] = str(d / "lora.safetensors") if (d / "lora.safetensors").exists() else None
    return a


def _save(a: dict) -> None:
    keep = {k: v for k, v in a.items() if k not in ("images", "audio", "cover")}
    keep["lora"] = {k: v for k, v in keep["lora"].items() if k != "ready_path"}
    keep["updated"] = time.time()
    p = _dir(a["id"]) / "asset.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(keep, ensure_ascii=False, indent=1))
    tmp.replace(p)


def update(aid: str, **fields) -> dict:
    with _lock:
        a = load(aid)
        for k, v in fields.items():
            if isinstance(v, dict) and isinstance(a.get(k), dict):
                a[k] = {**a[k], **v}
            else:
                a[k] = v
        _save(a)
    return load(aid)


def list_assets(kind: str | None = None) -> list[dict]:
    LIB_DIR.mkdir(parents=True, exist_ok=True)
    out = [load(p.name) for p in LIB_DIR.iterdir() if (p / "asset.json").exists()]
    out = [a for a in out if a and (not kind or a["kind"] == kind)]
    return sorted(out, key=lambda a: a["updated"], reverse=True)


def create(kind: str, name: str, description: str = "", tags: list[str] | None = None) -> dict:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}")
    if not name.strip():
        raise ValueError("Give it a name")
    aid = f"{_slug(name)}-{uuid.uuid4().hex[:5]}"
    (_dir(aid) / ("audio" if kind == "music" else "images")).mkdir(parents=True)
    a = {
        "id": aid, "kind": kind, "name": name.strip()[:60], "description": description.strip()[:1200],
        "tags": [t.strip()[:30] for t in (tags or []) if t.strip()][:12], "voice": None, "cover_file": None,
        "lora": {"status": "none", "trigger": f"rvd_{_slug(name)}", "steps": 0, "progress": 0.0, "message": ""},
        "generation": {"status": "idle", "progress": 0.0, "message": ""},
        "used_in": [], "created": time.time(), "updated": time.time(),
    }
    _save(a)
    return load(aid)


def delete(aid: str) -> None:
    shutil.rmtree(_dir(aid), ignore_errors=True)


def add_files(aid: str, files: list[tuple[str, bytes]]) -> dict:
    a = load(aid)
    music = a["kind"] == "music"
    allowed, sub = (AUDIO_EXT, "audio") if music else (IMAGE_EXT, "images")
    (_dir(aid) / sub).mkdir(exist_ok=True)
    for name, data in files:
        ext = Path(name).suffix.lower()
        if ext not in allowed:
            raise ValueError(f"Unsupported file type {ext or '(none)'}")
        stem = _slug(Path(name).stem)
        (_dir(aid) / sub / f"up_{int(time.time() * 1000)}_{stem}{ext}").write_bytes(data)
    return update(aid)


def remove_file(aid: str, name: str) -> dict:
    for sub in ("images", "audio"):
        (_dir(aid) / sub / Path(name).name).unlink(missing_ok=True)
    return update(aid)


def image_paths(aid: str) -> list[Path]:
    return [p for p in sorted((_dir(aid) / "images").glob("*")) if p.suffix.lower() in IMAGE_EXT]


def cover_path(aid: str) -> Path | None:
    a = load(aid)
    if not a or not a["cover"]:
        return None
    return _dir(aid) / "images" / a["cover"].rsplit("/", 1)[-1]


def lora_for(aid: str) -> dict | None:
    """The trained LoRA for an asset as {path, trigger, weight}, or None."""
    a = load(aid)
    if a and a["lora"]["status"] == "ready" and a["lora"].get("ready_path"):
        return {"path": a["lora"]["ready_path"], "trigger": a["lora"]["trigger"],
                "weight": 0.9 if a["kind"] == "character" else 0.7}
    return None


def mark_used(aid: str, production: str) -> None:
    a = load(aid)
    if a and production not in a["used_in"]:
        update(aid, used_in=(a["used_in"] + [production])[-50:])


def handle(a: dict) -> str:
    """How the user mentions an asset in chat: @Name with spaces as underscores."""
    return "@" + re.sub(r"\s+", "_", a["name"].strip())


def catalog() -> str:
    """Compact listing for the crew's prompts."""
    rows = []
    for a in list_assets():
        if a["kind"] == "music":
            rows.append(f"- [{a['kind']}] {a['name']} {handle(a)} (id {a['id']}): {a['description'][:160]}")
            continue
        trained = " · trained LoRA" if a["lora"]["status"] == "ready" else ""
        refs = f" · {len(a['images'])} reference images" if a["images"] else ""
        rows.append(f"- [{a['kind']}] {a['name']} {handle(a)} (id {a['id']}){trained}{refs}: {a['description'][:400]}")
    return "\n".join(rows) or "(empty)"


# --- AI description ---------------------------------------------------------------

DESCRIBE = """You are the Production Designer of a film crew. Write the FIXED visual description of a
reusable {kind} so an image model draws it identically in every shot. Return ONLY JSON
{{"name": str, "description": str, "tags": [str]}}.
- description: ENGLISH, 25-60 words, concrete visual facts only: {focus}
- Be culturally and scientifically exact. No story, no personality adjectives that can't be seen.
- tags: 2-5 short lowercase tags.
"""
FOCUS = {
    "character": "species/age, build, face, hair, skin or feathers, clothing with colours and materials, distinctive marks",
    "prop": "shape, size, material, colours, texture, wear, distinctive details",
    "location": "place, era, architecture, vegetation, ground, sky, light and palette",
    "style": "medium (3D, watercolour, anime, photoreal…), rendering, line, texture, lighting, palette, mood",
    "music": "instruments, tempo, mood, region/genre (write it as a short music brief instead)",
}


def describe(aid: str, prompt: str) -> dict:
    from .planner import _chat

    a = load(aid)
    out = _chat(DESCRIBE.format(kind=a["kind"], focus=FOCUS[a["kind"]]),
                f"Name: {a['name']}\nCurrent description: {a['description'] or '(none)'}\nRequest: {prompt or 'improve it'}")
    return update(aid, description=str(out.get("description", a["description"]))[:1200],
                  tags=[str(t)[:30] for t in out.get("tags", a["tags"])][:12])


# --- GPU jobs (run by the job worker) ------------------------------------------------

VARIATIONS = {
    "character": ["full body, front view, standing", "three-quarter view, full body", "side profile, full body",
                  "close-up portrait of the head and face", "in motion, dynamic pose", "sitting or resting pose",
                  "seen from slightly above", "seen from a low angle", "medium shot, looking to the left",
                  "medium shot, looking to the right", "full body from behind, turning its head",
                  "close-up with a happy expression", "close-up with a worried expression", "medium shot in soft morning light",
                  "full body in warm evening light", "medium shot on a simple outdoor background"],
    "prop": ["front view", "three-quarter view", "side view", "top-down view", "close-up detail", "on the ground outdoors",
             "on a table indoors", "in soft morning light", "in warm evening light", "low angle view"],
    "location": ["wide establishing shot", "eye-level view", "from a low angle", "aerial view", "at sunrise", "at midday",
                 "at sunset", "at night", "in the rain", "close detail of the ground and textures"],
    "style": ["a portrait of an old man", "a village street", "a cat on a windowsill", "a mountain landscape",
              "a bowl of fruit", "a child running in a field", "a city at night", "a forest path", "a boat on a river",
              "a bird on a branch", "a market stall", "a classroom", "a temple at dawn", "a car on a road",
              "a woman reading", "a stormy sea"],
}


def queue_generate(jobs, aid: str, count: int) -> dict:
    a = load(aid)
    if a["kind"] == "music":
        raise ValueError("Music has no images to generate; upload audio instead")
    if not a["description"].strip():
        raise ValueError("Write a description first (or use ✨ Write with AI)")
    _queue(jobs, aid, {"action": "generate", "count": max(2, min(int(count), 16))})
    return update(aid, generation={"status": "queued", "progress": 0.0, "message": "Waiting for the GPU…"})


def queue_train(jobs, aid: str, steps: int | None) -> dict:
    a = load(aid)
    if a["kind"] not in TRAINABLE:
        raise ValueError("Only characters and styles can be trained")
    if len(image_paths(aid)) < 6:
        raise ValueError("Add or generate at least 6 images first (12-20 is better)")
    _queue(jobs, aid, {"action": "train", "steps": steps})
    return update(aid, lora={"status": "queued", "progress": 0.0, "message": "Waiting for the GPU…"})


def _queue(jobs, aid: str, task: dict) -> None:
    a = load(aid)
    job = jobs.create({"mode": "asset", "asset": aid, "title": f"{a['name']}: {task['action']}",
                       "preset": "fast", "aspect": "1:1", "scenes": [], **task})
    jobs.enqueue(job["id"])


def run_asset_job(spec: dict, report: Callable[[str, float, str], None], should_cancel: Callable[[], bool]) -> dict:
    aid = spec["asset"]
    t0 = time.time()
    if spec["action"] == "generate":
        field = "generation"
        try:
            _generate(aid, int(spec.get("count", 8)), report, should_cancel)
            update(aid, generation={"status": "idle", "progress": 1.0, "message": "Done"})
        except Exception as e:
            update(aid, generation={"status": "failed", "message": str(e)[:300]})
            raise
    else:
        field = "lora"
        try:
            _train(aid, spec.get("steps"), report, should_cancel)
        except Exception as e:
            update(aid, lora={"status": "failed", "message": str(e)[:300]})
            raise
    return {"total": time.time() - t0, field: "done"}


def _generate(aid: str, count: int, report, should_cancel) -> None:
    from .keyframes import KeyframeEngine
    from .model import Cancelled

    a = load(aid)
    update(aid, generation={"status": "running", "progress": 0.0, "message": "Loading image model…"})
    report("asset", 0.02, "Loading image model…")
    engine = KeyframeEngine()
    try:
        out = _dir(aid) / "images"
        variations = VARIATIONS[a["kind"]]
        base = int(time.time()) % 100000
        for k in range(count):
            if should_cancel():
                raise Cancelled()
            v = variations[k % len(variations)]
            # The variation goes first: it must survive any prompt-length limit.
            if a["kind"] == "style":
                prompt = f"{v}, in this style: {a['description']}. masterpiece, highly detailed"
            elif a["kind"] == "location":
                prompt = f"{v}, no people, {a['description']}. masterpiece, highly detailed"
            else:
                prompt = f"{v}, one subject only, a single {a['description']}. masterpiece, highly detailed"
            update(aid, generation={"status": "running", "progress": k / count, "message": f"Drawing image {k + 1}/{count}…"})
            report("asset", k / count, f"{a['name']}: drawing image {k + 1}/{count}")
            img = engine.keyframe(prompt, "1:1", base + k, None)
            img.save(out / f"gen_{base}_{k:02d}.png")
            frac = (k + 1) / count
            update(aid, generation={"status": "running", "progress": frac, "message": f"Image {k + 1}/{count}"})
            report("asset", frac, f"{a['name']}: image {k + 1}/{count}")
    finally:
        engine.close()


def _train(aid: str, steps: int | None, report, should_cancel) -> None:
    try:
        from .training import train_lora
    except ImportError:
        raise RuntimeError("Training is not installed yet")
    a = load(aid)
    update(aid, lora={"status": "training", "progress": 0.0, "message": "Preparing images…"})

    def prog(done: int, total: int) -> None:
        update(aid, lora={"status": "training", "progress": done / total, "message": f"Step {done}/{total}"})
        report("asset", done / total, f"Training {a['name']}: step {done}/{total}")

    class_word = {"character": a["description"].split(",")[0][:40], "style": "style"}[a["kind"]]
    res = train_lora(a["kind"], a["name"], image_paths(aid), _dir(aid) / "lora.safetensors",
                     trigger=a["lora"]["trigger"], class_word=class_word, steps=steps,
                     on_progress=prog, should_cancel=should_cancel)
    update(aid, lora={"status": "ready", "progress": 1.0, "steps": res.get("steps", steps or 0),
                      "message": f"Trained in {round(res.get('seconds', 0) / 60)} min"})
