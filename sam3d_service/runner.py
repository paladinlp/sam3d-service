from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import os
import sys
from threading import Lock
import time
from typing import Any

import numpy as np
from PIL import Image

from sam3d_service.config import Settings
from sam3d_service.storage import (
    ALIGNMENT_RESULT_PLY_NAME,
    FOCAL_LENGTH_JSON_NAME,
    INPUT_IMAGE_NAME,
    INPUT_MESH_NAME,
    INPUT_MASK_NAME,
    JOB_KIND_ALIGNMENT,
    JOB_KIND_SCENE,
    JOB_KIND_SINGLE,
    RESULT_JSON_NAME,
    RESULT_PLY_NAME,
    SCENE_RESULT_PLY_NAME,
)


class InferenceRunner:
    def __init__(self, settings: Settings, gpu_lock: Lock | None = None):
        self.settings = settings
        self.gpu_lock = gpu_lock
        self._inference = None
        self._make_scene = None
        self._process_and_save_alignment = None

    @property
    def checkpoint_ready(self) -> bool:
        return self.settings.pipeline_config.is_file()

    @property
    def model_loaded(self) -> bool:
        return self._inference is not None

    def load_model(self) -> None:
        if self._inference is not None:
            return
        if not self.checkpoint_ready:
            raise FileNotFoundError(
                f"Missing pipeline config: {self.settings.pipeline_config}"
            )
        os.environ.setdefault("CONDA_PREFIX", sys.prefix)
        os.environ.setdefault("CUDA_HOME", os.environ["CONDA_PREFIX"])
        notebook_dir = str(self.settings.repo_root / "notebook")
        if notebook_dir not in sys.path:
            sys.path.insert(0, notebook_dir)
        from inference import Inference, make_scene  # pylint: disable=import-error

        self._inference = Inference(str(self.settings.pipeline_config), compile=False)
        self._make_scene = make_scene

    def run_job(self, job_dir: Path, job: dict[str, Any]) -> dict[str, Any]:
        kind = job.get("kind", JOB_KIND_SINGLE)
        seed = int(job.get("seed") or 42)
        if kind == JOB_KIND_SINGLE:
            return self._run_single_job(job_dir, seed=seed)
        if kind == JOB_KIND_SCENE:
            return self._run_scene_job(job_dir, seed=seed)
        if kind == JOB_KIND_ALIGNMENT:
            return self._run_alignment_job(job_dir)
        raise ValueError(f"Unsupported job kind: {kind}")

    def _run_single_job(self, job_dir: Path, seed: int) -> dict[str, Any]:
        if self._inference is None:
            raise RuntimeError("Model is not loaded.")

        image = np.array(
            Image.open(job_dir / INPUT_IMAGE_NAME).convert("RGB"),
            dtype=np.uint8,
        )
        mask = np.array(
            Image.open(job_dir / INPUT_MASK_NAME).convert("L"),
            dtype=np.uint8,
        ) > 0

        start_time = time.perf_counter()
        lock = self.gpu_lock if self.gpu_lock is not None else nullcontext()
        with lock:
            output = self._inference(image, mask, seed=seed)
        inference_seconds = time.perf_counter() - start_time

        output["gs"].save_ply(str(job_dir / RESULT_PLY_NAME))
        return {
            "kind": JOB_KIND_SINGLE,
            "translation": self._tensor_to_flat_list(output.get("translation")),
            "rotation": self._tensor_to_flat_list(output.get("rotation")),
            "scale": self._tensor_to_flat_list(output.get("scale")),
            "artifacts": {
                "input_image": INPUT_IMAGE_NAME,
                "input_mask": INPUT_MASK_NAME,
                "result_ply": RESULT_PLY_NAME,
                "result_json": RESULT_JSON_NAME,
            },
            "preview_artifact": RESULT_PLY_NAME,
            "timings": {
                "inference_seconds": round(inference_seconds, 3),
            },
        }

    def _run_scene_job(self, job_dir: Path, seed: int) -> dict[str, Any]:
        if self._inference is None or self._make_scene is None:
            raise RuntimeError("Model is not loaded.")

        image = np.array(
            Image.open(job_dir / INPUT_IMAGE_NAME).convert("RGB"),
            dtype=np.uint8,
        )
        mask_paths = sorted(job_dir.glob("mask_*.png"))
        if not mask_paths:
            raise ValueError("Scene job does not contain any masks.")

        outputs: list[dict[str, Any]] = []
        object_results = []
        object_ply_files = []
        start_time = time.perf_counter()
        lock = self.gpu_lock if self.gpu_lock is not None else nullcontext()
        with lock:
            for index, mask_path in enumerate(mask_paths):
                mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0
                output = self._inference(image, mask, seed=seed)
                object_name = f"object_{index:03d}.ply"
                output["gs"].save_ply(str(job_dir / object_name))
                object_ply_files.append(object_name)
                object_results.append(
                    {
                        "index": index,
                        "translation": self._tensor_to_flat_list(output.get("translation")),
                        "rotation": self._tensor_to_flat_list(output.get("rotation")),
                        "scale": self._tensor_to_flat_list(output.get("scale")),
                        "result_ply": object_name,
                    }
                )
                outputs.append(output)
            scene_gs = self._make_scene(*outputs)
            scene_gs.save_ply(str(job_dir / SCENE_RESULT_PLY_NAME))
        inference_seconds = time.perf_counter() - start_time

        return {
            "kind": JOB_KIND_SCENE,
            "objects": object_results,
            "artifacts": {
                "input_image": INPUT_IMAGE_NAME,
                "mask_files": [mask_path.name for mask_path in mask_paths],
                "object_ply_files": object_ply_files,
                "result_ply": SCENE_RESULT_PLY_NAME,
                "result_json": RESULT_JSON_NAME,
            },
            "preview_artifact": SCENE_RESULT_PLY_NAME,
            "timings": {
                "inference_seconds": round(inference_seconds, 3),
            },
        }

    def _run_alignment_job(self, job_dir: Path) -> dict[str, Any]:
        process_and_save_alignment = self._load_alignment_helper()

        mesh_path = job_dir / INPUT_MESH_NAME
        if not mesh_path.exists():
            raise FileNotFoundError(f"Missing input mesh: {mesh_path}")
        focal_length_path = job_dir / FOCAL_LENGTH_JSON_NAME
        focal_length_json_path = str(focal_length_path) if focal_length_path.exists() else None

        start_time = time.perf_counter()
        lock = self.gpu_lock if self.gpu_lock is not None else nullcontext()
        with lock:
            success, output_mesh_path, result = process_and_save_alignment(
                mesh_path=str(mesh_path),
                mask_path=str(job_dir / INPUT_MASK_NAME),
                image_path=str(job_dir / INPUT_IMAGE_NAME),
                output_dir=str(job_dir),
                device=self.settings.device,
                focal_length_json_path=focal_length_json_path,
            )
        inference_seconds = time.perf_counter() - start_time
        if not success or not output_mesh_path or result is None:
            raise RuntimeError("3DB mesh alignment did not produce an output mesh.")

        output_mesh_path = Path(output_mesh_path)
        final_mesh_path = job_dir / ALIGNMENT_RESULT_PLY_NAME
        if output_mesh_path.resolve() != final_mesh_path.resolve():
            output_mesh_path.replace(final_mesh_path)

        translation = self._tensor_to_flat_list(result.get("translation"))
        scale_factor = float(result.get("scale_factor", 1.0))
        alignment_payload = {
            "scale_factor": scale_factor,
            "translation": translation,
            "focal_length": float(result.get("focal_length", 0.0)),
            "target_points_count": int(result.get("target_points_count", 0)),
            "cropped_vertices_count": int(result.get("cropped_vertices_count", 0)),
        }

        artifacts = {
            "input_image": INPUT_IMAGE_NAME,
            "input_mask": INPUT_MASK_NAME,
            "input_mesh": INPUT_MESH_NAME,
            "result_ply": ALIGNMENT_RESULT_PLY_NAME,
            "result_json": RESULT_JSON_NAME,
        }
        if focal_length_json_path is not None:
            artifacts["focal_length_json"] = FOCAL_LENGTH_JSON_NAME

        return {
            "kind": JOB_KIND_ALIGNMENT,
            "translation": translation,
            "scale": [scale_factor],
            "alignment": alignment_payload,
            "artifacts": artifacts,
            "preview_artifact": ALIGNMENT_RESULT_PLY_NAME,
            "timings": {
                "inference_seconds": round(inference_seconds, 3),
            },
        }

    def _load_alignment_helper(self):
        if self._process_and_save_alignment is not None:
            return self._process_and_save_alignment
        notebook_dir = str(self.settings.repo_root / "notebook")
        if notebook_dir not in sys.path:
            sys.path.insert(0, notebook_dir)
        from mesh_alignment import process_and_save_alignment  # pylint: disable=import-error

        self._process_and_save_alignment = process_and_save_alignment
        return self._process_and_save_alignment

    @staticmethod
    def _tensor_to_flat_list(value: Any) -> list[float]:
        if value is None:
            return []
        if hasattr(value, "detach"):
            value = value.detach().cpu().reshape(-1).tolist()
        elif hasattr(value, "tolist"):
            value = np.asarray(value).reshape(-1).tolist()
        elif not isinstance(value, list):
            value = [value]
        return [float(item) for item in value]
