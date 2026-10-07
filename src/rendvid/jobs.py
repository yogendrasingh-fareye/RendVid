"""Single-worker job queue persisted as outputs/<id>/job.json.

One GPU means one render at a time. Jobs that were running when the server
stopped come back as "interrupted" and can be resumed; queued ones re-queue.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import traceback
import uuid
from pathlib import Path

from . import config
from .model import Cancelled
from . import editor
from .render import run_job

ACTIVE = {"queued", "running"}


class JobManager:
    def __init__(self, root: Path = config.OUTPUTS):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.q: queue.Queue[str] = queue.Queue()
        self.lock = threading.Lock()
        self.cancel_flags: set[str] = set()
        self._recover()
        threading.Thread(target=self._worker, daemon=True).start()

    # --- persistence ---------------------------------------------------------
    def dir(self, job_id: str) -> Path:
        return self.root / job_id

    def load(self, job_id: str) -> dict | None:
        p = self.dir(job_id) / "job.json"
        return json.loads(p.read_text()) if p.exists() else None

    def save(self, job: dict) -> None:
        p = self.dir(job["id"]) / "job.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(job, indent=2))
        tmp.replace(p)

    def update(self, job_id: str, **fields) -> dict:
        with self.lock:
            job = self.load(job_id)
            job.update(fields, updated=time.time())
            self.save(job)
            return job

    def list(self) -> list[dict]:
        jobs = [self.load(p.name) for p in self.root.iterdir() if (p / "job.json").exists()]
        return sorted(jobs, key=lambda j: j["created"], reverse=True)

    def _recover(self) -> None:
        for job in sorted(self.list(), key=lambda j: j["created"]):
            if job["status"] == "running":
                self.update(job["id"], status="interrupted", message="Server stopped mid-render. Resume to continue.")
            elif job["status"] == "queued":
                self.q.put(job["id"])

    # --- API -------------------------------------------------------------------
    def create(self, spec: dict) -> dict:
        job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        self.dir(job_id).mkdir(parents=True)
        job = {
            "id": job_id, "spec": spec, "status": "draft", "progress": 0.0,
            "stage": "", "message": "", "created": time.time(), "updated": time.time(),
        }
        self.save(job)
        return job

    def enqueue(self, job_id: str, task: str | None = None) -> dict:
        """Queue a task: "render" (full), "redo" (missing scenes + export) or "export"."""
        fields = {"status": "queued", "message": "Waiting in queue…", "error": None}
        if task:
            fields["task"] = task
        job = self.update(job_id, **fields)
        self.q.put(job_id)
        return job

    def cancel(self, job_id: str) -> dict:
        job = self.load(job_id)
        if job["status"] == "queued":
            return self.update(job_id, status="cancelled", message="Cancelled")
        self.cancel_flags.add(job_id)
        return self.update(job_id, message="Cancelling after current step…")

    def delete(self, job_id: str) -> None:
        import shutil

        job = self.load(job_id)
        if job and job["status"] in ACTIVE:
            raise ValueError("Cancel the job before deleting it")
        shutil.rmtree(self.dir(job_id), ignore_errors=True)

    # --- worker -------------------------------------------------------------
    def _worker(self) -> None:
        while True:
            job_id = self.q.get()
            job = self.load(job_id)
            if not job or job["status"] != "queued":
                continue
            self.cancel_flags.discard(job_id)
            self.update(job_id, status="running", started=time.time(), message="Starting…")

            def report(stage: str, progress: float, msg: str, job_id=job_id) -> None:
                self.update(job_id, stage=stage, progress=round(progress, 4), message=msg)

            cancel = lambda job_id=job_id: job_id in self.cancel_flags  # noqa: E731
            task = job.get("task", "render")
            d = self.dir(job_id)
            try:
                if job["spec"].get("mode") == "studio" and task in ("render", "redo"):
                    from .studio_render import run_studio

                    timings = run_studio(job, d, report, cancel, lambda spec, job_id=job_id: self.update(job_id, spec=spec))
                elif task == "render":
                    timings = run_job(job["spec"], d, report, cancel)
                elif task == "redo":
                    timings = run_job(job["spec"], d, report, cancel, assemble=False)
                    timings.update(editor.export(job["spec"], d, report, cancel, base=0.8))
                else:
                    timings = editor.export(job["spec"], d, report, cancel)
                if task != "render":
                    timings = {**(job.get("timings") or {}), "last_edit": timings}
                self.update(job_id, status="done", progress=1.0, message="Done", timings=timings,
                            final=f"/outputs/{job_id}/final.mp4")
            except Cancelled:
                self.update(job_id, status="cancelled", message="Cancelled. Resume keeps finished scenes.")
            except Exception as e:
                traceback.print_exc()
                self.update(job_id, status="failed", message=str(e)[:500], error=traceback.format_exc()[-4000:])
            finally:
                self.cancel_flags.discard(job_id)
