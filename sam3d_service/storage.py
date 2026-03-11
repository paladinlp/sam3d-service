from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime, timezone
from typing import Any


JOB_META_NAME = "job.json"
RESULT_JSON_NAME = "result.json"
JOB_KIND_SINGLE = "single"
JOB_KIND_SCENE = "scene"
JOB_KIND_ALIGNMENT = "alignment"
INPUT_IMAGE_NAME = "input.png"
INPUT_MASK_NAME = "mask.png"
INPUT_MESH_NAME = "input_mesh.ply"
FOCAL_LENGTH_JSON_NAME = "focal_length.json"
RESULT_PLY_NAME = "result.ply"
POSED_RESULT_PLY_NAME = "posed_result.ply"
PREVIEW_PLY_NAME = "preview.ply"
PREVIEW_GIF_NAME = "preview.gif"
PREVIEW_VIDEO_NAME = "preview.mp4"
SUPERSPLAT_HTML_NAME = "supersplat.html"
SUPERSPLAT_SOG_NAME = "supersplat.sog"
SUPERSPLAT_JS_NAME = "index.js"
SUPERSPLAT_CSS_NAME = "index.css"
SCENE_RESULT_PLY_NAME = "scene_result.ply"
ALIGNMENT_RESULT_PLY_NAME = "aligned_mesh.ply"
ERROR_NAME = "error.txt"


class JobNotFoundError(FileNotFoundError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class JobStore:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def recover_incomplete_jobs(self) -> None:
        for job_dir in self.data_dir.iterdir():
            meta_path = job_dir / JOB_META_NAME
            if not meta_path.exists():
                continue
            meta = self._read_json(meta_path)
            if meta.get("status") in {"queued", "running"}:
                self.mark_failed(
                    meta["job_id"],
                    "Service restarted before the job completed.",
                )

    def create_job(
        self,
        job_id: str,
        kind: str,
        files: dict[str, bytes],
        seed: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        job_dir = self.job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=False)
        for name, content in files.items():
            self._validate_artifact_name(name)
            (job_dir / name).write_bytes(content)
        payload = {
            "job_id": job_id,
            "kind": kind,
            "status": "queued",
            "progress": 0.0,
            "stage": "queued",
            "message": "Waiting for worker.",
            "error": None,
            "seed": int(seed) if seed is not None else None,
            "created_at": utc_now(),
            "started_at": None,
            "finished_at": None,
        }
        if extra:
            payload.update(extra)
        self._write_json(job_dir / JOB_META_NAME, payload)
        return payload

    def read_job(self, job_id: str) -> dict[str, Any]:
        meta = self._read_json(self._require_path(self.job_dir(job_id) / JOB_META_NAME))
        result_path = self.job_dir(job_id) / RESULT_JSON_NAME
        meta["result"] = self._read_json(result_path) if result_path.exists() else None
        return meta

    def mark_running(self, job_id: str) -> None:
        meta = self.read_job(job_id)
        meta["status"] = "running"
        meta["error"] = None
        meta["progress"] = max(float(meta.get("progress", 0.0)), 2.0)
        meta["stage"] = "starting"
        meta["message"] = "Preparing inputs."
        meta["started_at"] = utc_now()
        self._write_json(self.job_dir(job_id) / JOB_META_NAME, meta)

    def update_progress(
        self,
        job_id: str,
        *,
        progress: float | None = None,
        stage: str | None = None,
        message: str | None = None,
    ) -> None:
        meta = self.read_job(job_id)
        if progress is not None:
            meta["progress"] = round(min(max(float(progress), 0.0), 100.0), 1)
        if stage is not None:
            meta["stage"] = stage
        if message is not None:
            meta["message"] = message
        self._write_json(self.job_dir(job_id) / JOB_META_NAME, meta)

    def mark_succeeded(self, job_id: str, result: dict[str, Any]) -> None:
        self._write_json(self.job_dir(job_id) / RESULT_JSON_NAME, result)
        meta = self.read_job(job_id)
        meta["status"] = "succeeded"
        meta["error"] = None
        meta["progress"] = 100.0
        meta["stage"] = "completed"
        meta["message"] = "Artifacts are ready."
        meta["finished_at"] = utc_now()
        self._write_json(self.job_dir(job_id) / JOB_META_NAME, meta)

    def mark_failed(self, job_id: str, error_message: str) -> None:
        job_dir = self.job_dir(job_id)
        (job_dir / ERROR_NAME).write_text(error_message, encoding="utf-8")
        meta = self.read_job(job_id)
        meta["status"] = "failed"
        meta["stage"] = "failed"
        meta["message"] = "Job failed."
        meta["error"] = error_message.splitlines()[0][:500]
        meta["finished_at"] = utc_now()
        self._write_json(job_dir / JOB_META_NAME, meta)

    def artifact_path(self, job_id: str, name: str) -> Path:
        self._validate_artifact_name(name)
        return self._require_path(self.job_dir(job_id) / name)

    def job_dir(self, job_id: str) -> Path:
        return self.data_dir / job_id

    def job_paths(self, job_id: str) -> dict[str, Path]:
        job_dir = self.job_dir(job_id)
        return {
            "job_dir": job_dir,
            "image": job_dir / INPUT_IMAGE_NAME,
            "mask": job_dir / INPUT_MASK_NAME,
            "mesh": job_dir / INPUT_MESH_NAME,
            "focal_length_json": job_dir / FOCAL_LENGTH_JSON_NAME,
            "result_ply": job_dir / RESULT_PLY_NAME,
            "posed_result_ply": job_dir / POSED_RESULT_PLY_NAME,
            "preview_ply": job_dir / PREVIEW_PLY_NAME,
            "preview_gif": job_dir / PREVIEW_GIF_NAME,
            "preview_video": job_dir / PREVIEW_VIDEO_NAME,
            "supersplat_html": job_dir / SUPERSPLAT_HTML_NAME,
            "supersplat_sog": job_dir / SUPERSPLAT_SOG_NAME,
            "supersplat_js": job_dir / SUPERSPLAT_JS_NAME,
            "supersplat_css": job_dir / SUPERSPLAT_CSS_NAME,
            "result_json": job_dir / RESULT_JSON_NAME,
            "error": job_dir / ERROR_NAME,
        }

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        serializable = dict(payload)
        serializable.pop("result", None)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(serializable, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(path)

    def _read_json(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _require_path(self, path: Path) -> Path:
        if not path.exists():
            raise JobNotFoundError(f"Path does not exist: {path}")
        return path

    @staticmethod
    def _validate_artifact_name(name: str) -> None:
        if not name or Path(name).name != name or name in {".", ".."}:
            raise JobNotFoundError(f"Invalid artifact name: {name}")
