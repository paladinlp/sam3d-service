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
    INPUT_IMAGE_NAME,
    INPUT_MASK_NAME,
    RESULT_JSON_NAME,
    RESULT_PLY_NAME,
)


class InferenceRunner:
    def __init__(self, settings: Settings, gpu_lock: Lock | None = None):
        self.settings = settings
        self.gpu_lock = gpu_lock
        self._inference = None

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
        from inference import Inference  # pylint: disable=import-error

        self._inference = Inference(str(self.settings.pipeline_config), compile=False)

    def run_job(self, job_dir: Path, seed: int) -> dict[str, Any]:
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
            "translation": self._tensor_to_flat_list(output.get("translation")),
            "rotation": self._tensor_to_flat_list(output.get("rotation")),
            "scale": self._tensor_to_flat_list(output.get("scale")),
            "artifacts": {
                "input_image": INPUT_IMAGE_NAME,
                "input_mask": INPUT_MASK_NAME,
                "result_ply": RESULT_PLY_NAME,
                "result_json": RESULT_JSON_NAME,
            },
            "timings": {
                "inference_seconds": round(inference_seconds, 3),
            },
        }

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
