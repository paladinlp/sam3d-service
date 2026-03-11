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
        self._model = None
        self._predictor = None
        self._mask_generator = None
        self._load_lock = Lock()

    @property
    def checkpoint_ready(self) -> bool:
        return self.settings.segment_checkpoint.is_file()

    @property
    def model_loaded(self) -> bool:
        return self._predictor is not None

    def load_model(self) -> None:
        if self._predictor is not None and self._mask_generator is not None:
            return
        with self._load_lock:
            if self._predictor is not None and self._mask_generator is not None:
                return
            if not self.checkpoint_ready:
                raise FileNotFoundError(
                    f"Missing Segment Anything checkpoint: {self.settings.segment_checkpoint}"
                )
            try:
                from segment_anything import (
                    SamAutomaticMaskGenerator,
                    SamPredictor,
                    sam_model_registry,
                )
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
            self._model = model
            self._predictor = SamPredictor(model)
            # The scene flow needs instance proposals instead of just one best mask.
            self._mask_generator = SamAutomaticMaskGenerator(
                model=model,
                points_per_side=self.settings.segment_auto_points_per_side,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=0,
            )

    def segment_click_from_bytes(
        self,
        image_bytes: bytes,
        x: float,
        y: float,
        label: int = 1,
    ) -> dict:
        return self.segment_points_from_bytes(
            image_bytes,
            [{"x": x, "y": y, "label": label}],
        )

    def segment_points_from_bytes(
        self,
        image_bytes: bytes,
        points: list[dict[str, float | int]],
    ) -> dict:
        self.load_model()
        image = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"), dtype=np.uint8)
        height, width = image.shape[:2]
        if not points:
            raise ValueError("At least one point prompt is required.")

        coords = []
        labels = []
        for point in points:
            x = float(point["x"])
            y = float(point["y"])
            label = int(point.get("label", 1))
            if label not in {0, 1}:
                raise ValueError(f"Unsupported point label: {label}")
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(
                    f"Point ({x}, {y}) is outside image bounds ({width}, {height})."
                )
            coords.append([x, y])
            labels.append(label)

        point_coords = np.asarray(coords, dtype=np.float32)
        point_labels = np.asarray(labels, dtype=np.int32)
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
            "x": float(point_coords[-1][0]),
            "y": float(point_coords[-1][1]),
            "score": float(scores[best_index]),
            "width": int(width),
            "height": int(height),
            "mask_png_base64": base64.b64encode(mask_png).decode("ascii"),
        }

    def generate_mask_candidates_from_bytes(self, image_bytes: bytes) -> dict:
        self.load_model()
        image = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"), dtype=np.uint8)
        height, width = image.shape[:2]
        image_area = float(height * width)

        lock = self.gpu_lock if self.gpu_lock is not None else nullcontext()
        with lock:
            raw_masks = self._mask_generator.generate(image)

        prepared: list[dict] = []
        min_ratio = max(0.0, self.settings.segment_auto_min_area_ratio)
        max_ratio = min(1.0, self.settings.segment_auto_max_area_ratio)

        for item in raw_masks:
            mask = np.asarray(item.get("segmentation"), dtype=bool)
            area = int(item.get("area") or int(mask.sum()))
            if area <= 0:
                continue
            coverage = area / image_area
            if coverage < min_ratio or coverage > max_ratio:
                continue
            predicted_iou = float(item.get("predicted_iou") or 0.0)
            stability_score = float(item.get("stability_score") or 0.0)
            bbox = [int(round(value)) for value in item.get("bbox", [0, 0, width, height])]
            rank_score = self._candidate_rank_score(coverage, predicted_iou, stability_score)
            prepared.append(
                {
                    "mask": mask,
                    "area": area,
                    "coverage": coverage,
                    "predicted_iou": predicted_iou,
                    "stability_score": stability_score,
                    "bbox": bbox,
                    "rank_score": rank_score,
                }
            )

        if not prepared:
            for item in raw_masks:
                mask = np.asarray(item.get("segmentation"), dtype=bool)
                area = int(item.get("area") or int(mask.sum()))
                if area <= 0:
                    continue
                coverage = area / image_area
                predicted_iou = float(item.get("predicted_iou") or 0.0)
                stability_score = float(item.get("stability_score") or 0.0)
                bbox = [int(round(value)) for value in item.get("bbox", [0, 0, width, height])]
                prepared.append(
                    {
                        "mask": mask,
                        "area": area,
                        "coverage": coverage,
                        "predicted_iou": predicted_iou,
                        "stability_score": stability_score,
                        "bbox": bbox,
                        "rank_score": self._candidate_rank_score(
                            coverage, predicted_iou, stability_score
                        ),
                    }
                )

        prepared.sort(key=lambda item: item["rank_score"], reverse=True)

        deduped: list[dict] = []
        for item in prepared:
            if any(
                self._mask_iou(item["mask"], existing["mask"]) >= self.settings.segment_auto_dedup_iou
                for existing in deduped
            ):
                continue
            deduped.append(item)
            if len(deduped) >= self.settings.segment_auto_max_candidates:
                break

        candidates = []
        for index, item in enumerate(deduped):
            mask_png = self._mask_to_png_bytes(item["mask"])
            bbox_key = "-".join(str(value) for value in item["bbox"])
            candidates.append(
                {
                    "candidate_id": f"candidate-{index:03d}-{bbox_key}-{item['area']}",
                    "area": item["area"],
                    "coverage": item["coverage"],
                    "score": item["rank_score"],
                    "predicted_iou": item["predicted_iou"],
                    "stability_score": item["stability_score"],
                    "bbox": item["bbox"],
                    "mask_png_base64": base64.b64encode(mask_png).decode("ascii"),
                }
            )

        return {
            "width": int(width),
            "height": int(height),
            "candidate_count": len(candidates),
            "candidates": candidates,
        }

    @staticmethod
    def _candidate_rank_score(
        coverage: float, predicted_iou: float, stability_score: float
    ) -> float:
        coverage_bonus = min(coverage / 0.08, 1.0)
        return (
            predicted_iou * 0.52
            + stability_score * 0.33
            + coverage_bonus * 0.15
        )

    @staticmethod
    def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        intersection = np.logical_and(mask_a, mask_b).sum()
        union = np.logical_or(mask_a, mask_b).sum()
        if union == 0:
            return 0.0
        return float(intersection / union)

    @staticmethod
    def _mask_to_png_bytes(mask: np.ndarray) -> bytes:
        buffer = io.BytesIO()
        Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(buffer, format="PNG")
        return buffer.getvalue()
