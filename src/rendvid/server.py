"""FastAPI app: web UI + JSON API on localhost."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, editor, planner, stitch, tts
from .jobs import ACTIVE, JobManager

app = FastAPI(title="RendVid")
jobs = JobManager()

from .library_api import router as library_router  # noqa: E402
from .library_api import setup as library_setup  # noqa: E402

library_setup(jobs)
app.include_router(library_router)

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}


def _models_ready() -> dict:
    from huggingface_hub import try_to_load_from_cache

    from . import keyframes

    def cached(repo: str, f: str) -> bool:
        return isinstance(try_to_load_from_cache(repo, f), str)

    return {
        "video": cached(config.CKPT_REPO, config.CKPT_FILE),
        "text_encoder": cached(config.BASE_REPO, "text_encoder/model-00004-of-00004.safetensors"),
        "upscaler": cached(config.UPSCALER_REPO, "model_index.json"),
        "llm": planner.llm_available(),
        "brain": planner.brain_label(),
        "hindi_voice": tts.parler_ready(),
        "keyframes": keyframes.ready(),
    }


@app.get("/api/options")
def options():
    return {
        "styles": list(config.STYLES),
        "voices": config.VOICES,
        "aspects": list(config.ASPECTS),
        "presets": list(config.PRESETS),
        "default_negative": config.DEFAULT_NEGATIVE,
        "scene_seconds": (config.CLIP_FRAMES - config.OVERLAP_FRAMES) / config.FPS,
        "song_max_seconds": config.SONG_MAX_SECONDS,
        "studio_voices": tts.all_voices(),
        "models": _models_ready(),
    }


class PlanReq(BaseModel):
    idea: str
    duration: int = 60
    narration: bool = True
    language: str = "English"
    style: str = "none"


@app.post("/api/plan")
def plan(req: PlanReq):
    if not req.idea.strip():
        raise HTTPException(400, "Write an idea first")
    duration = min(max(req.duration, 5), 60)
    return planner.plan(req.idea, duration, narration=req.narration, language=req.language, style=req.style)


@app.post("/api/plan_song")
async def plan_song(
    audio: UploadFile = File(...),
    context: str = Form(""),
    style: str = Form("none"),
    use_lyrics: bool = Form(True),
):
    import tempfile

    from . import lyrics, stitch

    ext = Path(audio.filename or "").suffix.lower()
    if ext not in AUDIO_EXT:
        raise HTTPException(400, f"Unsupported file type {ext or '(none)'}")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / f"song{ext}"
        path.write_bytes(await audio.read())
        duration = stitch.probe_duration(path)
        if duration <= 0:
            raise HTTPException(400, "Could not read the audio file")
        if duration > config.SONG_MAX_SECONDS:
            raise HTTPException(400, f"Song is {duration / 60:.1f} min; the limit is {config.SONG_MAX_SECONDS // 60} min")
        words = await run_in_threadpool(lyrics.transcribe, path) if use_lyrics else {"language": None, "segments": []}
    if not context.strip() and not words["segments"]:
        raise HTTPException(400, "Describe the video you want (no lyrics were found)")
    result = await run_in_threadpool(
        planner.plan_song, context.strip() or "A music video", duration, words["segments"], style=style
    )
    result.update(duration=duration, lyrics=words["segments"], language=words["language"])
    return result


class StudioPlanReq(BaseModel):
    source: str = "script"  # "script" | "idea"
    text: str
    minutes: float = 3
    language: str = "Hindi"
    style: str = "none"


@app.post("/api/plan_studio")
def plan_studio(req: StudioPlanReq):
    from . import studio

    if not req.text.strip():
        raise HTTPException(400, "Paste a script or write an idea first")
    try:
        if req.source == "idea":
            return studio.plan_from_idea(req.text, minutes=min(max(req.minutes, 0.5), 10),
                                         language=req.language, style=req.style)
        return studio.plan_from_script(req.text, language=req.language, style=req.style)
    except ValueError as e:
        raise HTTPException(400, str(e))


def _validate_studio(spec: dict) -> dict:
    # Silent beats (no narration, a visual and `seconds`) are allowed for visual films.
    lines = [ln for ln in spec.get("lines", [])
             if str(ln.get("narration", "")).strip() or str(ln.get("visual", "")).strip()]
    if not lines:
        raise HTTPException(400, "Add at least one line")
    if len(lines) > 200:
        raise HTTPException(400, "Max 200 lines")
    narrated = any(str(ln.get("narration", "")).strip() for ln in lines)
    if narrated and spec.get("voice") not in tts.all_voices():
        raise HTTPException(400, "Choose a narrator voice")
    if not narrated:
        spec["voice"] = None
    for ln in lines:
        ln["narration"] = str(ln.get("narration", "")).strip()
        ln["visual"] = str(ln.get("visual", "")).strip() or ln["narration"]
        ln["seconds"] = min(max(float(ln.get("seconds") or 5), 2.0), 12.0)
        ln["new_shot"] = bool(ln.get("new_shot", True))
        ln["chapter"] = int(ln.get("chapter", 0))
        ln["characters"] = [str(c) for c in ln.get("characters", [])][:4]
    lines[0]["new_shot"] = True
    spec["lines"] = lines
    spec["characters"] = [
        {"name": str(c.get("name", "")).strip()[:30], "look": str(c.get("look", "")).strip()[:1200],
         "library_id": str(c.get("library_id") or "")[:60]}
        for c in spec.get("characters", []) if str(c.get("name", "")).strip() and str(c.get("look", "")).strip()
    ][:6]
    spec["scenes"] = []  # built from narration timing when the job starts
    spec["style"] = spec.get("style") if spec.get("style") in config.STYLES else "none"
    spec["seed"] = int(spec.get("seed") or 42)
    spec["guidance"] = 1.0
    return spec


def _validate(spec: dict) -> dict:
    if spec.get("preset") not in config.PRESETS:
        raise HTTPException(400, "Unknown quality preset")
    if spec.get("aspect") not in config.ASPECTS:
        raise HTTPException(400, "Unknown aspect ratio")
    if spec.get("mode") == "studio":
        return _validate_studio(spec)
    scenes = [s for s in spec.get("scenes", []) if str(s.get("prompt", "")).strip()]
    if not scenes:
        raise HTTPException(400, "Add at least one scene")
    song = spec.get("mode") == "song"
    limit = config.SONG_MAX_SCENES if song else config.STORY_MAX_SCENES
    if len(scenes) > limit:
        raise HTTPException(400, f"Max {limit} scenes")
    if song:
        spec["voice"] = None
    if spec.get("voice") and spec["voice"] not in config.VOICES:
        raise HTTPException(400, "Unknown voice")
    spec["scenes"] = scenes
    spec["style"] = spec.get("style") if spec.get("style") in config.STYLES else "none"
    spec["seed"] = int(spec.get("seed") or 42)
    spec["guidance"] = min(max(float(spec.get("guidance") or 1.0), 1.0), 5.0)
    return spec


async def _store(upload: UploadFile, dest_stem: Path, allowed: set[str]) -> str:
    ext = Path(upload.filename or "").suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type {ext or '(none)'}")
    dest = dest_stem.with_suffix(ext)
    dest.write_bytes(await upload.read())
    return dest.name


@app.post("/api/jobs")
async def create_job(
    spec: str = Form(...),
    image: UploadFile | None = File(None),
    music: UploadFile | None = File(None),
):
    data = _validate(json.loads(spec))
    job = jobs.create(data)
    d = jobs.dir(job["id"])
    try:
        if image and image.filename:
            data["init_image"] = await _store(image, d / "input_image", IMAGE_EXT)
        if music and music.filename:
            data["music_file"] = await _store(music, d / "music_upload", AUDIO_EXT)
            data.setdefault("music", {})["mode"] = "upload"
    except HTTPException:
        jobs.delete(job["id"])
        raise
    if data.get("mode") == "song" and not data.get("music_file"):
        jobs.delete(job["id"])
        raise HTTPException(400, "Music video needs the song file")
    if (data.get("music") or {}).get("mode") == "upload" and not data.get("music_file"):
        data["music"]["mode"] = "none"
    jobs.update(job["id"], spec=data)
    return jobs.enqueue(job["id"])


@app.get("/api/jobs")
def list_jobs():
    return jobs.list()


def _get(job_id: str) -> dict:
    job = jobs.load(job_id)
    if not job:
        raise HTTPException(404, "No such job")
    return job


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = _get(job_id)
    d = jobs.dir(job_id)
    job["clips"] = [f"/outputs/{job_id}/{p.name}" for p in sorted(d.glob("scene_*.mp4")) if ".tmp" not in p.name]
    return job


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    if _get(job_id)["status"] not in ACTIVE:
        raise HTTPException(400, "Job is not running")
    return jobs.cancel(job_id)


@app.post("/api/jobs/{job_id}/resume")
def resume_job(job_id: str):
    if _get(job_id)["status"] in ACTIVE:
        raise HTTPException(400, "Job is already running")
    return jobs.enqueue(job_id)


# --- editor ----------------------------------------------------------------------

def _idle(job_id: str) -> dict:
    job = _get(job_id)
    if job["status"] in ACTIVE:
        raise HTTPException(400, "Wait for the current task to finish (or cancel it)")
    return job


def _save_edit(job: dict, edit: dict) -> dict:
    d = jobs.dir(job["id"])
    try:
        clean = editor.validate(edit, job["spec"], d)
    except ValueError as e:
        raise HTTPException(400, str(e))
    editor.edit_path(d).write_text(json.dumps(clean, indent=2))
    return clean


@app.get("/api/jobs/{job_id}/edit")
def get_edit(job_id: str):
    job = _get(job_id)
    d = jobs.dir(job_id)
    scenes = []
    for i, sc in enumerate(job["spec"]["scenes"]):
        clip = editor.clip_path(d, i + 1)
        scenes.append({
            "scene": i + 1, "prompt": sc["prompt"], "new_shot": bool(sc.get("new_shot")),
            "seed": sc.get("seed"),
            "clip": f"/outputs/{job_id}/{clip.name}?{int(clip.stat().st_mtime)}" if clip.exists() else None,
            "duration": stitch.probe_duration(clip) if clip.exists() else 0,
        })
    versions = sorted((d / "versions").glob("v*.mp4"), reverse=True) if (d / "versions").exists() else []
    audio_files = sorted(p.name for p in d.iterdir() if p.suffix.lower() in AUDIO_EXT | {".wav"}
                         and not p.name.startswith("narration"))
    return {
        "edit": editor.load_edit(job["spec"], d),
        "customized": editor.edit_path(d).exists(),
        "scenes": scenes,
        "mode": job["spec"].get("mode", "story"),
        "versions": [f"/outputs/{job_id}/versions/{p.name}" for p in versions],
        "audio_files": audio_files,
        "voices": tts.all_voices(),
        "filters": list(editor.FILTERS),
        "transitions": sorted(editor.TRANSITIONS),
    }


@app.put("/api/jobs/{job_id}/edit")
def put_edit(job_id: str, edit: dict):
    return _save_edit(_idle(job_id), edit)


@app.delete("/api/jobs/{job_id}/edit")
def reset_edit(job_id: str):
    _idle(job_id)
    editor.edit_path(jobs.dir(job_id)).unlink(missing_ok=True)
    return {"ok": True}


@app.post("/api/jobs/{job_id}/export")
def export_edit(job_id: str, edit: dict):
    _save_edit(_idle(job_id), edit)
    return jobs.enqueue(job_id, task="export")


class RedoScene(BaseModel):
    scene: int
    prompt: str | None = None
    seed: int | None = None
    new_variation: bool = False
    new_keyframe: bool = False  # Studio: also redraw the still


class RedoReq(BaseModel):
    scenes: list[RedoScene]
    cascade: bool = False
    edit: dict | None = None


@app.post("/api/jobs/{job_id}/redo")
def redo_scenes(job_id: str, req: RedoReq):
    import random
    import shutil
    import time

    job = _idle(job_id)
    spec = job["spec"]
    n = len(spec["scenes"])
    d = jobs.dir(job_id)
    targets: set[int] = set()
    for r in req.scenes:
        if not 1 <= r.scene <= n:
            raise HTTPException(400, f"No scene {r.scene}")
        sc = spec["scenes"][r.scene - 1]
        if r.prompt and r.prompt.strip():
            sc["prompt"] = r.prompt.strip()
            if sc.get("keyframe"):  # Studio: the still is drawn from the line's visual
                spec["lines"][sc["line"]]["visual"] = r.prompt.strip()
        if sc.get("keyframe") and (r.new_keyframe or (r.prompt and r.prompt.strip())):
            (d / sc["keyframe"]).unlink(missing_ok=True)
            if r.new_keyframe:
                sc["kf_seed"] = random.randint(1, 2**31 - 1)
        if r.new_variation:
            sc["seed"] = random.randint(1, 2**31 - 1)
        elif r.seed is not None:
            sc["seed"] = r.seed
        targets.add(r.scene)
        if req.cascade:  # following scenes that continue from this one
            k = r.scene + 1
            while k <= n and not spec["scenes"][k - 1].get("new_shot"):
                targets.add(k)
                k += 1
    if req.edit:
        _save_edit(job, req.edit)
    hist = d / "history"
    hist.mkdir(exist_ok=True)
    stamp = time.strftime("%H%M%S")
    for k in sorted(targets):
        clip = editor.clip_path(d, k)
        if clip.exists():
            shutil.move(clip, hist / f"scene_{k:02d}_{stamp}.mp4")
        shutil.rmtree(d / f"tail_{k:02d}", ignore_errors=True)
    jobs.update(job_id, spec=spec)
    job = jobs.enqueue(job_id, task="redo")
    job["redo_scenes"] = sorted(targets)
    return job


@app.post("/api/jobs/{job_id}/audio")
async def upload_audio(job_id: str, audio: UploadFile = File(...)):
    import time

    _idle(job_id)
    name = await _store(audio, jobs.dir(job_id) / f"music_{time.strftime('%H%M%S')}", AUDIO_EXT)
    return {"music_file": name}


class RestoreReq(BaseModel):
    version: str


@app.post("/api/jobs/{job_id}/restore")
def restore_version(job_id: str, req: RestoreReq):
    import shutil
    import time

    _idle(job_id)
    vdir = jobs.dir(job_id) / "versions"
    src = vdir / Path(req.version).name
    if not src.exists():
        raise HTTPException(404, "No such version")
    final = jobs.dir(job_id) / "final.mp4"
    if final.exists():
        shutil.move(final, vdir / f"v{len(list(vdir.glob('v*.mp4'))) + 1:02d}_{time.strftime('%H%M%S')}.mp4")
    shutil.copy2(src, final)
    return jobs.update(job_id, message="Restored an earlier version")


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    _get(job_id)
    try:
        jobs.delete(job_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


app.mount("/outputs", StaticFiles(directory=config.OUTPUTS), name="outputs")
app.mount("/static", StaticFiles(directory=config.WEB), name="static")


# --- crew ----------------------------------------------------------------------------

from .crew import members as crew_members  # noqa: E402
from .crew import production as crew  # noqa: E402
from .crew import runner as crew_runner  # noqa: E402


def _submit_spec(spec: dict) -> dict:
    import shutil

    song = spec.pop("song_path", None)
    data = _validate(spec)
    job = jobs.create(data)
    if song:  # music video: the song becomes the job's soundtrack before it starts
        name = "music_upload" + Path(song).suffix.lower()
        shutil.copy2(song, jobs.dir(job["id"]) / name)
        data["music_file"] = name
        jobs.update(job["id"], spec=data)
    return jobs.enqueue(job["id"])


crew.setup(jobs, _submit_spec)


class CrewMessage(BaseModel):
    text: str = ""
    attachments: list[str] = []


@app.post("/api/crew/attachments")
async def crew_attach(files: list[UploadFile] = File(...)):
    out = []
    for f in files:
        try:
            out.append(crew.save_attachment(f.filename or "file", await f.read()))
        except ValueError as e:
            raise HTTPException(400, str(e))
    return out


@app.get("/api/crew/roster")
def crew_roster():
    return {k: {"role": m["role"], "model": m.get("model", "opus")} for k, m in crew_members.CREW.items()}


@app.get("/api/crew/productions")
def crew_list():
    return crew.list_all()


@app.post("/api/crew/productions")
def crew_create(msg: CrewMessage):
    if not msg.text.strip() and not msg.attachments:
        raise HTTPException(400, "Say what film you want")
    if not planner.llm_available():
        raise HTTPException(400, "The crew needs the Claude CLI (claude) installed and logged in")
    return crew.create(msg.text, msg.attachments)


@app.get("/api/crew/productions/{pid}")
def crew_get(pid: str):
    prod = crew.load(pid)
    if not prod:
        raise HTTPException(404, "No such production")
    if prod.get("job_id"):
        job = jobs.load(prod["job_id"]) or {}
        prod["job"] = {k: job.get(k) for k in ("id", "status", "progress", "message", "stage")}
    return prod


@app.post("/api/crew/productions/{pid}/message")
def crew_message(pid: str, msg: CrewMessage):
    if not crew.load(pid):
        raise HTTPException(404, "No such production")
    try:
        crew.user_message(pid, msg.text, msg.attachments)
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {"ok": True}


class CrewApprove(BaseModel):
    what: str  # "brief" | "storyboard"


@app.post("/api/crew/productions/{pid}/approve")
def crew_approve(pid: str, req: CrewApprove):
    try:
        crew.approve(pid, req.what)
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {"ok": True}


@app.post("/api/crew/productions/{pid}/stop")
def crew_stop(pid: str):
    if not crew.load(pid):
        raise HTTPException(404, "No such production")
    try:
        return {"stopped": crew.stop(pid)}
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.post("/api/crew/productions/{pid}/resume")
def crew_resume(pid: str):
    if not crew.load(pid):
        raise HTTPException(404, "No such production")
    try:
        crew.resume(pid)
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {"ok": True}


@app.delete("/api/crew/productions/{pid}")
def crew_delete(pid: str):
    crew._path(pid).unlink(missing_ok=True)
    return {"ok": True}


@app.get("/api/crew/skills")
def crew_skills():
    """How each role works: outputs/.crew/skills/<role>.md (read-only view for the UI)."""
    out = []
    for k, m in crew_members.CREW.items():
        f = crew_runner.CREW_DIR / "skills" / f"{m['skill']}.md"
        out.append({"key": k, "role": m["role"], "name": m["skill"], "path": str(f),
                    "content": f.read_text() if f.exists() else ""})
    return out


@app.get("/api/crew/memory")
def crew_memory():
    """What the crew has learned: team.md, members/<key>.md and history.md (read-only view for the UI)."""
    d = crew_runner.CREW_DIR

    def read(p: Path) -> str:
        return p.read_text() if p.exists() else ""

    return {"dir": str(d), "team": read(d / "team.md"), "history": read(d / "history.md"),
            "members": {k: read(d / "members" / f"{k}.md") for k in crew_members.CREW}}


@app.get("/edit/{job_id}")
def edit_page(job_id: str):
    return FileResponse(config.WEB / "edit.html")


# --- web app (web-app/, built into web/app by `pnpm build`) ----------------------------

APP_DIR = config.WEB / "app"
app.mount("/assets", StaticFiles(directory=APP_DIR / "assets", check_dir=False), name="assets")


@app.get("/")
def index():
    page = APP_DIR / "index.html"
    if not page.exists():
        raise HTTPException(503, "Web app not built: run `pnpm install && pnpm build` in web-app/")
    return FileResponse(page, headers={"Cache-Control": "no-cache"})
