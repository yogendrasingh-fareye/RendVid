"""HTTP API for Characters & Resources (see library.py)."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from . import library, tts

router = APIRouter(prefix="/api/library")
_jobs = None  # JobManager, set by the server


def setup(jobs) -> None:
    global _jobs
    _jobs = jobs


def _get(aid: str) -> dict:
    a = library.load(aid)
    if not a:
        raise HTTPException(404, "No such asset")
    return a


class NewAsset(BaseModel):
    kind: str
    name: str
    description: str = ""
    tags: list[str] = []


class AssetPatch(BaseModel):
    name: str | None = None
    description: str | None = None
    tags: list[str] | None = None
    voice: str | None = None
    cover: str | None = None


class Describe(BaseModel):
    prompt: str = ""


class Generate(BaseModel):
    count: int = 8


class Train(BaseModel):
    steps: int | None = None


@router.get("")
def list_assets(kind: str | None = None):
    return library.list_assets(kind)


@router.get("/voices")
def voices():
    return tts.all_voices()


@router.post("")
def create(req: NewAsset):
    try:
        return library.create(req.kind, req.name, req.description, req.tags)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/{aid}")
def get(aid: str):
    return _get(aid)


@router.patch("/{aid}")
def patch(aid: str, req: AssetPatch):
    _get(aid)
    fields = {}
    if req.name is not None and req.name.strip():
        fields["name"] = req.name.strip()[:60]
    if req.description is not None:
        fields["description"] = req.description.strip()[:1200]
    if req.tags is not None:
        fields["tags"] = [t.strip()[:30] for t in req.tags if t.strip()][:12]
    if req.voice is not None:
        fields["voice"] = req.voice if req.voice in tts.all_voices() else None
    if req.cover is not None:
        fields["cover_file"] = req.cover.rsplit("/", 1)[-1]
    return library.update(aid, **fields)


@router.delete("/{aid}")
def delete(aid: str):
    a = _get(aid)
    if a["generation"]["status"] in ("queued", "running") or a["lora"]["status"] in ("queued", "training"):
        raise HTTPException(409, "Wait for the running job to finish")
    library.delete(aid)
    return {"ok": True}


@router.post("/{aid}/describe")
async def describe(aid: str, req: Describe):
    _get(aid)
    try:
        return await run_in_threadpool(library.describe, aid, req.prompt)
    except Exception as e:
        raise HTTPException(502, f"Could not write the description: {e}")


@router.post("/{aid}/images")
async def upload(aid: str, files: list[UploadFile] = File(...)):
    _get(aid)
    try:
        return library.add_files(aid, [(f.filename or "file", await f.read()) for f in files])
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.delete("/{aid}/images/{name}")
def remove(aid: str, name: str):
    _get(aid)
    return library.remove_file(aid, name)


@router.post("/{aid}/generate")
def generate(aid: str, req: Generate):
    a = _get(aid)
    if a["generation"]["status"] in ("queued", "running"):
        raise HTTPException(409, "Already generating")
    try:
        return library.queue_generate(_jobs, aid, req.count)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/{aid}/train")
def train(aid: str, req: Train):
    a = _get(aid)
    if a["lora"]["status"] in ("queued", "training"):
        raise HTTPException(409, "Already training")
    try:
        return library.queue_train(_jobs, aid, req.steps)
    except ValueError as e:
        raise HTTPException(400, str(e))
