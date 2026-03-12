from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import os
import logging
import sys
from threading import Lock
import time
from typing import Any, Callable

import numpy as np
from PIL import Image

from sam3d_service.config import Settings
from sam3d_service.preview_ply import build_preview_ply
from sam3d_service.supersplat_viewer import build_supersplat_viewer
from sam3d_service.storage import (
    ALIGNMENT_RESULT_PLY_NAME,
    FOCAL_LENGTH_JSON_NAME,
    INPUT_IMAGE_NAME,
    INPUT_MESH_NAME,
    INPUT_MASK_NAME,
    JOB_KIND_ALIGNMENT,
    JOB_KIND_SCENE,
    JOB_KIND_SINGLE,
    POSED_RESULT_PLY_NAME,
    PREVIEW_GIF_NAME,
    PREVIEW_PLY_NAME,
    PREVIEW_VIDEO_NAME,
    RESULT_JSON_NAME,
    RESULT_PLY_NAME,
    SCENE_RESULT_PLY_NAME,
    SUPERSPLAT_CSS_NAME,
    SUPERSPLAT_HTML_NAME,
    SUPERSPLAT_JS_NAME,
    SUPERSPLAT_SOG_NAME,
)


ProgressCallback = Callable[..., None]
LOGGER = logging.getLogger(__name__)


class InferenceRunner:
    def __init__(self, settings: Settings, gpu_lock: Lock | None = None):
        self.settings = settings
        self.gpu_lock = gpu_lock
        self._inference = None
        self._make_scene = None
        self._ready_gaussian_for_video_rendering = None
        self._render_video = None
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
        from inference import (  # pylint: disable=import-error
            Inference,
            make_scene,
            ready_gaussian_for_video_rendering,
            render_video,
        )

        self._inference = Inference(str(self.settings.pipeline_config), compile=False)
        self._make_scene = make_scene
        self._ready_gaussian_for_video_rendering = ready_gaussian_for_video_rendering
        self._render_video = render_video

    def run_job(
        self,
        job_dir: Path,
        job: dict[str, Any],
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        kind = job.get("kind", JOB_KIND_SINGLE)
        seed = int(job.get("seed") or 42)
        if kind == JOB_KIND_SINGLE:
            return self._run_single_job(job_dir, seed=seed, progress_callback=progress_callback)
        if kind == JOB_KIND_SCENE:
            return self._run_scene_job(job_dir, seed=seed, progress_callback=progress_callback)
        if kind == JOB_KIND_ALIGNMENT:
            return self._run_alignment_job(job_dir, progress_callback=progress_callback)
        raise ValueError(f"Unsupported job kind: {kind}")

    def _run_single_job(
        self,
        job_dir: Path,
        seed: int,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        if self._inference is None or self._make_scene is None:
            raise RuntimeError("Model is not loaded.")

        self._emit_progress(
            progress_callback,
            progress=5,
            stage="loading_inputs",
            message="Loading image and mask.",
        )
        image = np.array(
            Image.open(job_dir / INPUT_IMAGE_NAME).convert("RGB"),
            dtype=np.uint8,
        )
        mask = np.array(
            Image.open(job_dir / INPUT_MASK_NAME).convert("L"),
            dtype=np.uint8,
        ) > 0

        inference_seconds = None
        render_seconds = None
        media_artifacts = {"gif_artifact": None, "video_artifact": None, "seconds": None}
        lock = self.gpu_lock if self.gpu_lock is not None else nullcontext()
        with lock:
            self._emit_progress(
                progress_callback,
                progress=18,
                stage="gaussian_inference",
                message="Running SAM 3D object inference.",
            )
            inference_start = time.perf_counter()
            output = self._inference(image, mask, seed=seed)
            inference_seconds = time.perf_counter() - inference_start
            self._emit_progress(
                progress_callback,
                progress=58,
                stage="rendering_preview",
                message="Rendering notebook-style preview.",
            )
            media_artifacts = self._maybe_render_media(job_dir, [output], mode="single")
            render_seconds = media_artifacts["seconds"]

        self._emit_progress(
            progress_callback,
            progress=82,
            stage="writing_artifacts",
            message="Writing gaussian and preview files.",
        )
        output["gs"].save_ply(str(job_dir / RESULT_PLY_NAME))
        scene_gs = self._make_scene(output)
        self._center_gaussian_in_place(scene_gs)
        scene_gs.save_ply(str(job_dir / POSED_RESULT_PLY_NAME))
        preview_artifact = self._maybe_create_preview(
            job_dir / POSED_RESULT_PLY_NAME,
            job_dir / PREVIEW_PLY_NAME,
        )
        self._emit_progress(
            progress_callback,
            progress=90,
            stage="building_viewer",
            message="Packaging PlayCanvas Gaussian viewer.",
        )
        viewer_artifacts = self._maybe_create_supersplat_viewer(
            source_path=job_dir / POSED_RESULT_PLY_NAME,
            job_dir=job_dir,
        )
        self._emit_progress(
            progress_callback,
            progress=96,
            stage="finalizing",
            message="Finalizing result payload.",
        )
        return {
            "kind": JOB_KIND_SINGLE,
            "translation": self._tensor_to_flat_list(output.get("translation")),
            "rotation": self._tensor_to_flat_list(output.get("rotation")),
            "scale": self._tensor_to_flat_list(output.get("scale")),
            "artifacts": {
                "input_image": INPUT_IMAGE_NAME,
                "input_mask": INPUT_MASK_NAME,
                "result_ply": RESULT_PLY_NAME,
                "preview_ply": preview_artifact,
                "preview_gif": media_artifacts["gif_artifact"],
                "preview_video": media_artifacts["video_artifact"],
                **viewer_artifacts,
                "result_json": RESULT_JSON_NAME,
            },
            "preview_artifact": preview_artifact,
            "timings": {
                "inference_seconds": round(inference_seconds, 3),
                "render_seconds": round(render_seconds, 3) if render_seconds is not None else None,
            },
        }

    def _run_scene_job(
        self,
        job_dir: Path,
        seed: int,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        if self._inference is None or self._make_scene is None:
            raise RuntimeError("Model is not loaded.")

        self._emit_progress(
            progress_callback,
            progress=5,
            stage="loading_inputs",
            message="Loading image and scene masks.",
        )
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
        inference_seconds = None
        render_seconds = None
        media_artifacts = {"gif_artifact": None, "video_artifact": None, "seconds": None}
        shared_pointmap = None
        lock = self.gpu_lock if self.gpu_lock is not None else nullcontext()
        with lock:
            inference_start = time.perf_counter()
            self._emit_progress(
                progress_callback,
                progress=10,
                stage="scene_pointmap",
                message="Computing shared scene pointmap once for all selected objects.",
            )
            shared_pointmap = self._compute_shared_pointmap(image)
            for index, mask_path in enumerate(mask_paths):
                progress = 16 + (index / max(len(mask_paths), 1)) * 40
                self._emit_progress(
                    progress_callback,
                    progress=progress,
                    stage="gaussian_inference",
                    message=f"Reconstructing object {index + 1}/{len(mask_paths)}.",
                )
                mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0
                output = self._inference(
                    image,
                    mask,
                    seed=seed,
                    pointmap=shared_pointmap,
                )
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
            self._emit_progress(
                progress_callback,
                progress=60,
                stage="anchoring_plane",
                message="Anchoring selected objects back onto a shared support plane.",
            )
            self._anchor_scene_outputs_to_support_plane(
                outputs=outputs,
                mask_paths=mask_paths,
                shared_pointmap=shared_pointmap,
            )
            self._emit_progress(
                progress_callback,
                progress=62,
                stage="assembling_scene",
                message="Merging object gaussians into one scene.",
            )
            scene_gs = self._make_scene(*outputs)
            scene_gs.save_ply(str(job_dir / SCENE_RESULT_PLY_NAME))
            inference_seconds = time.perf_counter() - inference_start
            self._emit_progress(
                progress_callback,
                progress=72,
                stage="rendering_preview",
                message="Rendering notebook-style scene preview.",
            )
            media_artifacts = self._maybe_render_media(job_dir, scene_gs, mode="scene")
            render_seconds = media_artifacts["seconds"]
        self._emit_progress(
            progress_callback,
            progress=88,
            stage="writing_artifacts",
            message="Writing scene preview files.",
        )
        preview_artifact = self._maybe_create_preview(
            job_dir / SCENE_RESULT_PLY_NAME,
            job_dir / PREVIEW_PLY_NAME,
        )
        self._emit_progress(
            progress_callback,
            progress=92,
            stage="building_viewer",
            message="Packaging PlayCanvas Gaussian viewer.",
        )
        viewer_artifacts = self._maybe_create_supersplat_viewer(
            source_path=job_dir / SCENE_RESULT_PLY_NAME,
            job_dir=job_dir,
        )
        self._emit_progress(
            progress_callback,
            progress=96,
            stage="finalizing",
            message="Finalizing scene payload.",
        )

        return {
            "kind": JOB_KIND_SCENE,
            "objects": object_results,
            "artifacts": {
                "input_image": INPUT_IMAGE_NAME,
                "mask_files": [mask_path.name for mask_path in mask_paths],
                "object_ply_files": object_ply_files,
                "result_ply": SCENE_RESULT_PLY_NAME,
                "preview_ply": preview_artifact,
                "preview_gif": media_artifacts["gif_artifact"],
                "preview_video": media_artifacts["video_artifact"],
                **viewer_artifacts,
                "result_json": RESULT_JSON_NAME,
            },
            "shared_pointmap_reused": shared_pointmap is not None,
            "preview_artifact": preview_artifact,
            "timings": {
                "inference_seconds": round(inference_seconds, 3),
                "render_seconds": round(render_seconds, 3) if render_seconds is not None else None,
            },
        }

    def _run_alignment_job(
        self,
        job_dir: Path,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        process_and_save_alignment = self._load_alignment_helper()

        mesh_path = job_dir / INPUT_MESH_NAME
        if not mesh_path.exists():
            raise FileNotFoundError(f"Missing input mesh: {mesh_path}")
        focal_length_path = job_dir / FOCAL_LENGTH_JSON_NAME
        focal_length_json_path = str(focal_length_path) if focal_length_path.exists() else None

        self._emit_progress(
            progress_callback,
            progress=8,
            stage="loading_inputs",
            message="Loading alignment inputs.",
        )
        start_time = time.perf_counter()
        lock = self.gpu_lock if self.gpu_lock is not None else nullcontext()
        with lock:
            self._emit_progress(
                progress_callback,
                progress=20,
                stage="alignment_inference",
                message="Aligning 3DB mesh to the reconstructed scale.",
            )
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
        self._emit_progress(
            progress_callback,
            progress=84,
            stage="writing_artifacts",
            message="Writing aligned mesh preview.",
        )
        preview_artifact = self._maybe_create_preview(
            final_mesh_path,
            job_dir / PREVIEW_PLY_NAME,
        )
        artifacts["preview_ply"] = preview_artifact
        self._emit_progress(
            progress_callback,
            progress=96,
            stage="finalizing",
            message="Finalizing alignment payload.",
        )

        return {
            "kind": JOB_KIND_ALIGNMENT,
            "translation": translation,
            "scale": [scale_factor],
            "alignment": alignment_payload,
            "artifacts": artifacts,
            "preview_artifact": preview_artifact,
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

    def _maybe_create_preview(self, source_path: Path, preview_path: Path) -> str:
        try:
            build_preview_ply(
                source_path=source_path,
                output_path=preview_path,
                max_points=self.settings.preview_max_points,
                opacity_threshold=self.settings.preview_opacity_threshold,
            )
            return preview_path.name
        except Exception:
            return source_path.name

    def _maybe_create_supersplat_viewer(
        self,
        *,
        source_path: Path,
        job_dir: Path,
    ) -> dict[str, str | None]:
        if not self.settings.supersplat_enabled:
            return {
                "supersplat_html": None,
                "supersplat_sog": None,
                "supersplat_js": None,
                "supersplat_css": None,
            }
        try:
            artifacts = build_supersplat_viewer(
                source_path=source_path,
                output_dir=job_dir,
                command=self.settings.supersplat_command,
            )
            return {
                "supersplat_html": artifacts.html_name,
                "supersplat_sog": artifacts.sog_name,
                "supersplat_js": artifacts.js_name,
                "supersplat_css": artifacts.css_name,
            }
        except Exception as exc:
            LOGGER.warning("SuperSplat viewer generation failed for %s: %s", source_path, exc)
            for artifact_name in (
                SUPERSPLAT_HTML_NAME,
                SUPERSPLAT_SOG_NAME,
                SUPERSPLAT_JS_NAME,
                SUPERSPLAT_CSS_NAME,
            ):
                (job_dir / artifact_name).unlink(missing_ok=True)
            return {
                "supersplat_html": None,
                "supersplat_sog": None,
                "supersplat_js": None,
                "supersplat_css": None,
            }

    def _maybe_render_media(
        self,
        job_dir: Path,
        scene_source: Any,
        mode: str,
    ) -> dict[str, Any]:
        if not self.settings.render_gif_enabled:
            return {"gif_artifact": None, "video_artifact": None, "seconds": None}
        if self._make_scene is None or self._ready_gaussian_for_video_rendering is None or self._render_video is None:
            return {"gif_artifact": None, "video_artifact": None, "seconds": None}

        try:
            import imageio.v2 as imageio

            render_start = time.perf_counter()
            if isinstance(scene_source, list):
                scene_gs = self._make_scene(*scene_source)
            else:
                scene_gs = scene_source
            scene_gs = self._ready_gaussian_for_video_rendering(scene_gs)

            render_kwargs = {
                "resolution": self.settings.render_gif_resolution,
                "num_frames": self.settings.render_gif_frames,
                "r": 1,
                "fov": 60,
            }
            if mode == "single":
                render_kwargs["pitch_deg"] = 15
                render_kwargs["yaw_start_deg"] = -45

            frames = self._render_video(scene_gs, **render_kwargs)["color"]
            gif_path = job_dir / PREVIEW_GIF_NAME
            imageio.mimsave(
                gif_path,
                [np.asarray(frame, dtype=np.uint8) for frame in frames],
                format="GIF",
                duration=1000 / max(self.settings.render_gif_fps, 1),
                loop=0,
            )
            video_path = self._maybe_write_video(job_dir / PREVIEW_VIDEO_NAME, frames)
            return {
                "gif_artifact": gif_path.name,
                "video_artifact": video_path.name if video_path is not None else None,
                "seconds": time.perf_counter() - render_start,
            }
        except Exception:
            return {"gif_artifact": None, "video_artifact": None, "seconds": None}

    def _maybe_write_video(self, output_path: Path, frames: list[Any]) -> Path | None:
        try:
            import imageio.v2 as imageio

            with imageio.get_writer(
                output_path,
                fps=max(self.settings.render_gif_fps, 1),
                codec="libx264",
                format="FFMPEG",
            ) as writer:
                for frame in frames:
                    writer.append_data(np.asarray(frame, dtype=np.uint8))
            return output_path
        except Exception:
            return None

    def _anchor_scene_outputs_to_support_plane(
        self,
        *,
        outputs: list[dict[str, Any]],
        mask_paths: list[Path],
        shared_pointmap: Any | None,
    ) -> None:
        if not outputs or shared_pointmap is None:
            return

        pointmap_np = self._pointmap_to_numpy(shared_pointmap)
        if pointmap_np is None:
            return

        contact_points = []
        masks = []
        for mask_path in mask_paths:
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0
            masks.append(mask)
            contact = self._extract_mask_contact_points(pointmap_np, mask)
            if contact.size:
                contact_points.append(contact)

        if not contact_points:
            return

        stacked_contact_points = np.concatenate(contact_points, axis=0)
        plane = self._fit_support_plane(stacked_contact_points)
        support_height = float(np.nanmedian(stacked_contact_points[:, 1]))

        for output in outputs:
            if plane is not None:
                applied = self._anchor_output_to_plane(output, plane)
                if applied:
                    continue
            self._anchor_output_to_height(output, support_height)

    @staticmethod
    def _pointmap_to_numpy(pointmap: Any) -> np.ndarray | None:
        if pointmap is None:
            return None
        if hasattr(pointmap, "detach"):
            pointmap = pointmap.detach().cpu().numpy()
        else:
            pointmap = np.asarray(pointmap)
        if pointmap.ndim != 3 or pointmap.shape[-1] != 3:
            return None
        if not np.isfinite(pointmap).any():
            return None
        return pointmap

    @staticmethod
    def _extract_mask_contact_points(pointmap: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask.shape != pointmap.shape[:2]:
            return np.empty((0, 3), dtype=np.float32)

        contact_points = []
        xs = np.where(mask.any(axis=0))[0]
        for x in xs:
            ys = np.where(mask[:, x])[0]
            if ys.size == 0:
                continue
            bottom_y = int(ys.max())
            top_y = max(int(ys.max() - 3), int(ys.min()))
            for y in range(bottom_y, top_y - 1, -1):
                point = pointmap[y, x]
                if np.isfinite(point).all():
                    contact_points.append(point)
                    break
        if not contact_points:
            return np.empty((0, 3), dtype=np.float32)
        return np.asarray(contact_points, dtype=np.float32)

    @staticmethod
    def _fit_support_plane(points: np.ndarray) -> tuple[np.ndarray, float] | None:
        if points.shape[0] < 8:
            return None
        center = np.nanmedian(points, axis=0)
        centered = points - center
        centered = centered[np.all(np.isfinite(centered), axis=1)]
        if centered.shape[0] < 8:
            return None
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        normal = vh[-1]
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-6:
            return None
        normal = normal / normal_norm
        if normal[1] < 0:
            normal = -normal
        # Reject near-vertical planes; fallback to shared height anchoring instead.
        if abs(normal[1]) < 0.35:
            return None
        offset = -float(np.dot(normal, center))
        return normal.astype(np.float32), offset

    def _anchor_output_to_plane(
        self,
        output: dict[str, Any],
        plane: tuple[np.ndarray, float],
    ) -> bool:
        import torch

        normal_np, offset = plane
        scene_points = self._transform_output_points_to_scene(output)
        if scene_points is None or scene_points.shape[0] == 0:
            return False
        normal = torch.tensor(
            normal_np,
            dtype=scene_points.dtype,
            device=scene_points.device,
        )
        distances = scene_points @ normal + float(offset)
        if distances.numel() == 0:
            return False
        bottom_distance = torch.quantile(distances, 0.02)
        output["translation"] = output["translation"] - normal.unsqueeze(0) * bottom_distance
        return True

    @staticmethod
    def _anchor_output_to_height(output: dict[str, Any], support_height: float) -> None:
        scene_points = InferenceRunner._transform_output_points_to_scene(output)
        if scene_points is None or scene_points.shape[0] == 0:
            return
        import torch

        bottom_height = torch.quantile(scene_points[:, 1], 0.02)
        output["translation"] = output["translation"].clone()
        output["translation"][0, 1] += support_height - float(bottom_height.detach().cpu())

    @staticmethod
    def _transform_output_points_to_scene(output: dict[str, Any]) -> Any | None:
        import torch

        gaussian = output.get("gaussian")
        rotation = output.get("rotation")
        translation = output.get("translation")
        scale = output.get("scale")
        if (
            gaussian is None
            or rotation is None
            or translation is None
            or scale is None
            or not gaussian
        ):
            return None
        xyz_local = gaussian[0].get_xyz
        if xyz_local is None or xyz_local.numel() == 0:
            return None
        quat = rotation.reshape(-1, 4)[0]
        quat = quat / torch.clamp(torch.linalg.norm(quat), min=1e-8)
        qw, qx, qy, qz = quat.unbind()
        rotation_matrix = torch.stack(
            [
                torch.stack([1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)]),
                torch.stack([2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)]),
                torch.stack([2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]),
            ],
            dim=0,
        ).to(dtype=xyz_local.dtype, device=xyz_local.device)
        scaled_xyz = xyz_local * scale.reshape(-1, 3)[0]
        rotated_xyz = scaled_xyz @ rotation_matrix.transpose(0, 1)
        return rotated_xyz + translation.reshape(-1, 3)[0]

    def _compute_shared_pointmap(self, image: np.ndarray) -> Any | None:
        pipeline = getattr(self._inference, "_pipeline", None)
        merge_mask_to_rgba = getattr(self._inference, "merge_mask_to_rgba", None)
        if pipeline is None or merge_mask_to_rgba is None or not hasattr(pipeline, "compute_pointmap"):
            return None
        try:
            rgba_image = merge_mask_to_rgba(
                image,
                np.ones(image.shape[:2], dtype=np.uint8),
            )
            pointmap_dict = pipeline.compute_pointmap(rgba_image)
            pointmap = pointmap_dict.get("pointmap")
            if pointmap is None:
                return None
            if hasattr(pointmap, "detach"):
                return pointmap.detach().permute(1, 2, 0).cpu()
            if hasattr(pointmap, "permute"):
                return pointmap.permute(1, 2, 0)
            return pointmap
        except Exception as exc:
            LOGGER.warning("Shared scene pointmap computation failed, falling back to per-object pointmaps: %s", exc)
            return None

    @staticmethod
    def _center_gaussian_in_place(scene_gs: Any) -> None:
        xyz = scene_gs.get_xyz
        opacity = scene_gs.get_opacity.squeeze()
        active_mask = opacity > 0.9
        if hasattr(active_mask, "any") and not bool(active_mask.any()):
            active_mask = opacity > 0
        if hasattr(active_mask, "any") and not bool(active_mask.any()):
            active_mask = None
        active_xyz = xyz[active_mask] if active_mask is not None else xyz
        center = (active_xyz.max(dim=0)[0] + active_xyz.min(dim=0)[0]) / 2
        scene_gs.from_xyz(xyz - center)

    @staticmethod
    def _emit_progress(
        callback: ProgressCallback | None,
        *,
        progress: float,
        stage: str,
        message: str,
    ) -> None:
        if callback is None:
            return
        callback(progress=progress, stage=stage, message=message)

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
