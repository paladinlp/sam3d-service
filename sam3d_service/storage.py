from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime, timezone
from typing import Any


JOB_META_NAME = "job.json"
RESULT_JSON_NAME = "result.json"
INPUT_IMAGE_NAME = "input.png"
INPUT_MASK_NAME = "mask.png"
RESULT_PLY_NAME = "result.ply"
ERROR_NAME = "error.txt"
ALLOWED_ARTIFACTS = {
    INPUT_IMAGE_NAME,
    INPUT_MASK_NAME,
    RESULT_JSON_NAME,
    RESULT_PLY_NAME,
    ERROR_NAME,
}


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
        seed: int,
        image_bytes: bytes,
        mask_bytes: bytes,
    ) -> dict[str, Any]:
        job_dir = self.job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=False)
        (job_dir / INPUT_IMAGE_NAME).write_bytes(image_bytes)
        (job_dir / INPUT_MASK_NAME).write_bytes(mask_bytes)
        payload = {
            "job_id": job_id,
            "status": "queued",
            "error": None,
            "seed": int(seed),
            "created_at": utc_now(),
            "started_at": None,
            "finished_at": None,
        }
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
        meta["started_at"] = utc_now()
        self._write_json(self.job_dir(job_id) / JOB_META_NAME, meta)

    def mark_succeeded(self, job_id: str, result: dict[str, Any]) -> None:
        self._write_json(self.job_dir(job_id) / RESULT_JSON_NAME, result)
        meta = self.read_job(job_id)
        meta["status"] = "succeeded"
        meta["error"] = None
        meta["finished_at"] = utc_now()
        self._write_json(self.job_dir(job_id) / JOB_META_NAME, meta)

    def mark_failed(self, job_id: str, error_message: str) -> None:
        job_dir = self.job_dir(job_id)
        (job_dir / ERROR_NAME).write_text(error_message, encoding="utf-8")
        meta = self.read_job(job_id)
        meta["status"] = "failed"
        meta["error"] = error_message.splitlines()[0][:500]
        meta["finished_at"] = utc_now()
        self._write_json(job_dir / JOB_META_NAME, meta)

    def artifact_path(self, job_id: str, name: str) -> Path:
        if name not in ALLOWED_ARTIFACTS:
            raise JobNotFoundError(f"Artifact '{name}' is not supported.")
        return self._require_path(self.job_dir(job_id) / name)

    def job_dir(self, job_id: str) -> Path:
        return self.data_dir / job_id

    def job_paths(self, job_id: str) -> dict[str, Path]:
        job_dir = self.job_dir(job_id)
        return {
            "job_dir": job_dir,
            "image": job_dir / INPUT_IMAGE_NAME,
            "mask": job_dir / INPUT_MASK_NAME,
            "result_ply": job_dir / RESULT_PLY_NAME,
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
