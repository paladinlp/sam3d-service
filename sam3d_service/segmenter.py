from __future__ import annotations

import base64
from contextlib import nullcontext
import io
from threading import Lock

import numpy as np
from PIL import Image

from sam3d_service.config import Settings


class ClickSegmenter:
    def __init__(self, settings: Settings, gpu_lock: Lock | None = None):
        self.settings = settings
        self.gpu_lock = gpu_lock
        self._predictor = None
        self._load_lock = Lock()

    @property
    def checkpoint_ready(self) -> bool:
        return self.settings.segment_checkpoint.is_file()

    @property
    def model_loaded(self) -> bool:
        return self._predictor is not None

    def load_model(self) -> None:
        if self._predictor is not None:
            return
        with self._load_lock:
            if self._predictor is not None:
                return
            if not self.checkpoint_ready:
                raise FileNotFoundError(
                    f"Missing Segment Anything checkpoint: {self.settings.segment_checkpoint}"
                )
            try:
                from segment_anything import SamPredictor, sam_model_registry
            except ImportError as exc:
                raise RuntimeError(
                    "segment-anything is not installed. Install sam3d_service/requirements.txt."
                ) from exc

            if self.settings.segment_model_type not in sam_model_registry:
                raise ValueError(
                    f"Unsupported Segment Anything model type: {self.settings.segment_model_type}"
                )

            model = sam_model_registry[self.settings.segment_model_type](
                checkpoint=str(self.settings.segment_checkpoint)
            )
            model.to(device=self.settings.segment_device)
            self._predictor = SamPredictor(model)

    def segment_click_from_bytes(
        self,
        image_bytes: bytes,
        x: float,
        y: float,
        label: int = 1,
    ) -> dict:
        self.load_model()
        image = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"), dtype=np.uint8)
        height, width = image.shape[:2]
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(
                f"Point ({x}, {y}) is outside image bounds ({width}, {height})."
            )

        point_coords = np.array([[x, y]], dtype=np.float32)
        point_labels = np.array([int(label)], dtype=np.int32)
        lock = self.gpu_lock if self.gpu_lock is not None else nullcontext()
        with lock:
            self._predictor.set_image(image)
            masks, scores, _ = self._predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )

        best_index = int(np.argmax(scores))
        best_mask = masks[best_index]
        mask_png = self._mask_to_png_bytes(best_mask)
        return {
            "x": float(x),
            "y": float(y),
            "score": float(scores[best_index]),
            "width": int(width),
            "height": int(height),
            "mask_png_base64": base64.b64encode(mask_png).decode("ascii"),
        }

    @staticmethod
    def _mask_to_png_bytes(mask: np.ndarray) -> bytes:
        buffer = io.BytesIO()
        Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(buffer, format="PNG")
        return buffer.getvalue()
