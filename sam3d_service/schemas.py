from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class ArtifactLinks(BaseModel):
    input_image: str
    input_mask: str
    result_ply: Optional[str] = None
    result_json: Optional[str] = None


class JobTimings(BaseModel):
    inference_seconds: Optional[float] = None
    total_seconds: Optional[float] = None


class JobResult(BaseModel):
    translation: List[float]
    rotation: List[float]
    scale: List[float]
    artifacts: ArtifactLinks
    timings: JobTimings


class JobCreateResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    error: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    result: Optional[JobResult] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    checkpoint_ready: bool
    device: str
    click_segmentation_model_loaded: bool
    click_segmentation_checkpoint_ready: bool
    click_segmentation_device: str


class ClickSegmentationResponse(BaseModel):
    x: float
    y: float
    score: float
    width: int
    height: int
    mask_png_base64: str
