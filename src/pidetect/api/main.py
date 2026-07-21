"""Phase 5 v1 — FastAPI backend. See docs/phase5_design.md.

4 endpoints, in-memory job store (no Celery/Redis — a demo serving one user at a
time doesn't need a broker; concurrent uploads serialize behind one worker, a known
v1 limitation, not fixed here):

    POST /jobs                 upload a sheet -> {"job_id": ...}, runs the pipeline
                                (pidetect.pipeline.run_pipeline) via BackgroundTasks
    GET  /jobs/{job_id}        {"status", "stage", "progress"}
    GET  /jobs/{job_id}/result the contracted graph as JSON (symbols, tags, edges)
    GET  /jobs/{job_id}/image  the original uploaded sheet (frontend draws the overlay)

Run:
    PYTHONPATH=src uvicorn pidetect.api.main:app --reload
"""
from __future__ import annotations

import threading
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse

from pidetect.graph.export import graph_to_dict
from pidetect.pipeline import run_pipeline

REPO = Path(__file__).resolve().parents[3]

with open(REPO / "configs" / "phase5.yaml") as f:
    _P5_CFG = yaml.safe_load(f)
with open(REPO / "configs" / "phase4.yaml") as f:
    _P4_CFG = yaml.safe_load(f)
with open(REPO / "configs" / "phase3.yaml") as f:
    _P3_CFG = yaml.safe_load(f)
_P3_CFG["nms_centroid_frac"] = _P4_CFG["nms_centroid_frac"]  # shared dedupe-assertion threshold

_WEIGHTS_PATH = REPO / _P5_CFG["weights_path"]
_DEVICE = _P5_CFG.get("device", "cpu")
_UPLOAD_DIR = REPO / _P5_CFG.get("upload_dir", "data/uploads")
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class JobRecord:
    job_id: str
    status: str = "queued"          # queued | running | done | failed
    stage: str = "queued"
    progress: Optional[dict] = None
    image_path: Optional[Path] = None
    result: Optional[dict] = None
    error: Optional[str] = None


_JOBS: dict[str, JobRecord] = {}
_JOBS_LOCK = threading.Lock()

app = FastAPI(title="PIDetect Phase 5 API")


def _run_job(job_id: str) -> None:
    with _JOBS_LOCK:
        job = _JOBS[job_id]
        job.status = "running"

    def _stage_cb(stage: str, progress: Optional[dict]) -> None:
        with _JOBS_LOCK:
            job.stage = stage
            job.progress = progress

    try:
        if not _WEIGHTS_PATH.exists():
            raise FileNotFoundError(
                f"Weights not found: {_WEIGHTS_PATH} (gitignored -- train locally or "
                "point configs/phase5.yaml at your weights)"
            )
        G = run_pipeline(
            job.image_path, _WEIGHTS_PATH, _P4_CFG, _P3_CFG,
            device=_DEVICE, stage_cb=_stage_cb,
        )
        result = graph_to_dict(G, sheet_id=job_id)
        result["metrics"] = _P5_CFG["metrics"]  # frozen eval numbers, not computed per-upload
        with _JOBS_LOCK:
            job.result = result
            job.status = "done"
            job.stage = "done"
    except Exception as exc:  # noqa: BLE001 -- surface any pipeline failure to the client, not just log it
        with _JOBS_LOCK:
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"


@app.post("/jobs")
async def create_job(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> dict:
    job_id = uuid.uuid4().hex
    suffix = Path(file.filename or "sheet.png").suffix or ".png"
    image_path = _UPLOAD_DIR / f"{job_id}{suffix}"
    image_path.write_bytes(await file.read())

    job = JobRecord(job_id=job_id, image_path=image_path)
    with _JOBS_LOCK:
        _JOBS[job_id] = job

    background_tasks.add_task(_run_job, job_id)
    return {"job_id": job_id}


def _get_job(job_id: str) -> JobRecord:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")
    return job


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str) -> dict:
    job = _get_job(job_id)
    resp = {"job_id": job.job_id, "status": job.status, "stage": job.stage, "progress": job.progress}
    if job.status == "failed":
        resp["error"] = job.error
    return resp


@app.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str) -> JSONResponse:
    job = _get_job(job_id)
    if job.status != "done":
        raise HTTPException(status_code=409, detail=f"Job not done yet (status={job.status})")
    return JSONResponse(job.result)


@app.get("/jobs/{job_id}/image")
async def get_job_image(job_id: str) -> FileResponse:
    job = _get_job(job_id)
    return FileResponse(job.image_path)
