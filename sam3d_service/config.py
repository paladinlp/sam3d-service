from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    data_dir: Path
    checkpoint_tag: str
    device: str
    preview_max_points: int
    preview_opacity_threshold: float
    segment_device: str
    segment_model_type: str
    segment_checkpoint_env: str | None
    host: str
    port: int

    @property
    def checkpoint_dir(self) -> Path:
        return self.repo_root / "checkpoints" / self.checkpoint_tag

    @property
    def pipeline_config(self) -> Path:
        return self.checkpoint_dir / "pipeline.yaml"

    @property
    def segment_checkpoint(self) -> Path:
        if self.segment_checkpoint_env:
            return Path(self.segment_checkpoint_env).expanduser().resolve()
        return (
            self.repo_root
            / "checkpoints"
            / "sam"
            / "sam_vit_h_4b8939.pth"
        )

    @classmethod
    def from_env(cls) -> "Settings":
        repo_root = Path(
            os.environ.get(
                "SAM3D_REPO_ROOT",
                str(Path(__file__).resolve().parents[1]),
            )
        ).expanduser().resolve()
        data_dir = Path(
            os.environ.get(
                "SAM3D_DATA_DIR",
                str(repo_root / "data" / "jobs"),
            )
        ).expanduser().resolve()
        checkpoint_tag = os.environ.get("SAM3D_CHECKPOINT_TAG", "hf")
        device = os.environ.get("SAM3D_DEVICE", "cuda")
        preview_max_points = int(os.environ.get("SAM3D_PREVIEW_MAX_POINTS", "25000"))
        preview_opacity_threshold = float(os.environ.get("SAM3D_PREVIEW_OPACITY_THRESHOLD", "0.08"))
        segment_device = os.environ.get("SAM3D_SEGMENT_DEVICE", device)
        segment_model_type = os.environ.get("SAM3D_SEGMENT_MODEL_TYPE", "vit_h")
        segment_checkpoint_env = os.environ.get("SAM3D_SEGMENT_CHECKPOINT")
        host = os.environ.get("SAM3D_HOST", "0.0.0.0")
        port = int(os.environ.get("SAM3D_PORT", "8000"))
        return cls(
            repo_root=repo_root,
            data_dir=data_dir,
            checkpoint_tag=checkpoint_tag,
            device=device,
            preview_max_points=preview_max_points,
            preview_opacity_threshold=preview_opacity_threshold,
            segment_device=segment_device,
            segment_model_type=segment_model_type,
            segment_checkpoint_env=segment_checkpoint_env,
            host=host,
            port=port,
        )
