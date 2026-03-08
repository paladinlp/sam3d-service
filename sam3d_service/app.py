from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import io
from pathlib import Path
from threading import Lock
import uuid

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image, UnidentifiedImageError

from sam3d_service.config import Settings
from sam3d_service.runner import InferenceRunner
from sam3d_service.segmenter import ClickSegmenter
from sam3d_service.schemas import (
    ArtifactLinks,
    ClickSegmentationResponse,
    HealthResponse,
    JobCreateResponse,
    JobResult,
    JobStatusResponse,
    JobTimings,
)
from sam3d_service.storage import ALLOWED_ARTIFACTS, JobNotFoundError, JobStore
from sam3d_service.worker import InferenceWorker


def create_app() -> FastAPI:
    settings = Settings.from_env()
    web_root = Path(__file__).resolve().parent / "web"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store = JobStore(settings.data_dir)
        store.recover_incomplete_jobs()
        gpu_lock = Lock()
        runner = InferenceRunner(settings, gpu_lock=gpu_lock)
        segmenter = ClickSegmenter(settings, gpu_lock=gpu_lock)
        worker = InferenceWorker(store, runner)
        await worker.start()
        app.state.settings = settings
        app.state.store = store
        app.state.runner = runner
        app.state.segmenter = segmenter
        app.state.worker = worker
        try:
            yield
        finally:
            await worker.stop()

    app = FastAPI(title="SAM 3D Objects Service", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse((web_root / "index.html").read_text(encoding="utf-8"))

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz(request: Request) -> HealthResponse:
        runner: InferenceRunner = request.app.state.runner
        segmenter: ClickSegmenter = request.app.state.segmenter
        active_settings: Settings = request.app.state.settings
        return HealthResponse(
            status="ok",
            model_loaded=runner.model_loaded,
            checkpoint_ready=runner.checkpoint_ready,
            device=active_settings.device,
            click_segmentation_model_loaded=segmenter.model_loaded,
            click_segmentation_checkpoint_ready=segmenter.checkpoint_ready,
            click_segmentation_device=active_settings.segment_device,
        )

    @app.post("/segment/click", response_model=ClickSegmentationResponse)
    async def segment_click(
        request: Request,
        image: UploadFile = File(...),
        x: float = Form(...),
        y: float = Form(...),
        label: int = Form(1),
    ) -> ClickSegmentationResponse:
        image_bytes, image_size = await _load_image_upload(image)
        if not (0 <= x < image_size[0] and 0 <= y < image_size[1]):
            raise HTTPException(
                status_code=400,
                detail=f"Point ({x}, {y}) is outside the image bounds.",
            )
        segmenter: ClickSegmenter = request.app.state.segmenter
        try:
            payload = await asyncio.to_thread(
                segmenter.segment_click_from_bytes,
                image_bytes,
                x,
                y,
                label,
            )
        except (FileNotFoundError, RuntimeError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return ClickSegmentationResponse(**payload)

    @app.post("/jobs", response_model=JobCreateResponse)
    async def create_job(
        request: Request,
        image: UploadFile = File(...),
        mask: UploadFile = File(...),
        seed: int = Form(42),
    ) -> JobCreateResponse:
        image_bytes, image_size = await _load_image_upload(image)
        mask_bytes, mask_size, has_foreground = await _load_mask_upload(mask)
        if image_size != mask_size:
            raise HTTPException(
                status_code=400,
                detail="Image and mask must have the same width and height.",
            )
        if not has_foreground:
            raise HTTPException(
                status_code=400,
                detail="Mask must contain at least one non-zero pixel.",
            )

        job_id = uuid.uuid4().hex
        store: JobStore = request.app.state.store
        worker: InferenceWorker = request.app.state.worker
        store.create_job(job_id, seed=seed, image_bytes=image_bytes, mask_bytes=mask_bytes)
        await worker.submit(job_id)
        return JobCreateResponse(job_id=job_id, status="queued")

    @app.get("/jobs/{job_id}", response_model=JobStatusResponse)
    async def get_job(request: Request, job_id: str) -> JobStatusResponse:
        store: JobStore = request.app.state.store
        try:
            payload = store.read_job(job_id)
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        result_payload = payload.get("result")
        result = None
        if result_payload is not None:
            artifacts = _artifact_links(request, job_id, result_payload.get("artifacts", {}))
            result = JobResult(
                translation=result_payload.get("translation", []),
                rotation=result_payload.get("rotation", []),
                scale=result_payload.get("scale", []),
                artifacts=artifacts,
                timings=JobTimings(**result_payload.get("timings", {})),
            )

        return JobStatusResponse(
            job_id=payload["job_id"],
            status=payload["status"],
            error=payload.get("error"),
            created_at=payload["created_at"],
            started_at=payload.get("started_at"),
            finished_at=payload.get("finished_at"),
            result=result,
        )

    @app.get("/jobs/{job_id}/artifacts/{name}", name="download_artifact")
    async def download_artifact(job_id: str, name: str, request: Request) -> FileResponse:
        if name not in ALLOWED_ARTIFACTS:
            raise HTTPException(status_code=404, detail=f"Unsupported artifact: {name}")
        store: JobStore = request.app.state.store
        try:
            artifact_path = store.artifact_path(job_id, name)
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(path=artifact_path, filename=Path(name).name)

    return app


async def _load_image_upload(upload: UploadFile) -> tuple[bytes, tuple[int, int]]:
    _ensure_image_upload(upload)
    raw_bytes = await upload.read()
    try:
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Invalid image upload.") from exc
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue(), image.size


async def _load_mask_upload(upload: UploadFile) -> tuple[bytes, tuple[int, int], bool]:
    _ensure_image_upload(upload)
    raw_bytes = await upload.read()
    try:
        mask = Image.open(io.BytesIO(raw_bytes)).convert("L")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Invalid mask upload.") from exc
    mask_array = np.array(mask, dtype=np.uint8)
    has_foreground = bool(np.any(mask_array > 0))
    mask_binary = Image.fromarray(((mask_array > 0) * 255).astype(np.uint8), mode="L")
    buffer = io.BytesIO()
    mask_binary.save(buffer, format="PNG")
    return buffer.getvalue(), mask_binary.size, has_foreground


def _ensure_image_upload(upload: UploadFile) -> None:
    if upload.content_type is None or not upload.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Only image uploads are supported.")


def _artifact_links(request: Request, job_id: str, artifact_payload: dict) -> ArtifactLinks:
    return ArtifactLinks(
        input_image=str(request.url_for("download_artifact", job_id=job_id, name=artifact_payload["input_image"])),
        input_mask=str(request.url_for("download_artifact", job_id=job_id, name=artifact_payload["input_mask"])),
        result_ply=str(request.url_for("download_artifact", job_id=job_id, name=artifact_payload["result_ply"])),
        result_json=str(request.url_for("download_artifact", job_id=job_id, name=artifact_payload["result_json"])),
    )


app = create_app()
