from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SceneObjectResult(BaseModel):
    index: int
    translation: List[float] = Field(default_factory=list)
    rotation: List[float] = Field(default_factory=list)
    scale: List[float] = Field(default_factory=list)
    result_ply: Optional[str] = None


class JobTimings(BaseModel):
    inference_seconds: Optional[float] = None
    render_seconds: Optional[float] = None
    total_seconds: Optional[float] = None


class JobResult(BaseModel):
    kind: str
    translation: List[float] = Field(default_factory=list)
    rotation: List[float] = Field(default_factory=list)
    scale: List[float] = Field(default_factory=list)
    objects: List[SceneObjectResult] = Field(default_factory=list)
    alignment: Optional[Dict[str, Any]] = None
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    timings: JobTimings
    preview_url: Optional[str] = None


class JobCreateResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    kind: str
    status: str
    progress: float = 0.0
    stage: Optional[str] = None
    message: Optional[str] = None
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
