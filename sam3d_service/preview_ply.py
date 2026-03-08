from __future__ import annotations

from pathlib import Path

import numpy as np


SH_C0 = 0.28209479177387814
DEFAULT_RGB = np.array([216, 194, 162], dtype=np.uint8)


def build_preview_ply(
    source_path: str | Path,
    output_path: str | Path,
    max_points: int,
    opacity_threshold: float,
) -> int:
    from plyfile import PlyData, PlyElement

    source_path = Path(source_path)
    output_path = Path(output_path)
    ply_data = PlyData.read(str(source_path))
    if "vertex" not in ply_data:
        raise ValueError(f"No vertex element found in {source_path}")

    vertex = ply_data["vertex"].data
    if len(vertex) == 0:
        raise ValueError(f"No vertices found in {source_path}")

    names = set(vertex.dtype.names or ())
    xyz = np.column_stack(
        [
            np.asarray(vertex["x"], dtype=np.float32),
            np.asarray(vertex["y"], dtype=np.float32),
            np.asarray(vertex["z"], dtype=np.float32),
        ]
    )
    rgb = _extract_rgb(vertex, names)
    indices = _select_indices(vertex, names, max_points=max_points, opacity_threshold=opacity_threshold)

    preview = np.empty(
        len(indices),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    preview["x"] = xyz[indices, 0]
    preview["y"] = xyz[indices, 1]
    preview["z"] = xyz[indices, 2]
    preview["red"] = rgb[indices, 0]
    preview["green"] = rgb[indices, 1]
    preview["blue"] = rgb[indices, 2]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(preview, "vertex")], text=False).write(str(output_path))
    return len(indices)


def _extract_rgb(vertex: np.ndarray, names: set[str]) -> np.ndarray:
    if {"red", "green", "blue"}.issubset(names):
        return np.column_stack(
            [
                _normalize_color_channel(np.asarray(vertex["red"])),
                _normalize_color_channel(np.asarray(vertex["green"])),
                _normalize_color_channel(np.asarray(vertex["blue"])),
            ]
        )

    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        rgb = np.column_stack(
            [
                np.asarray(vertex["f_dc_0"], dtype=np.float32),
                np.asarray(vertex["f_dc_1"], dtype=np.float32),
                np.asarray(vertex["f_dc_2"], dtype=np.float32),
            ]
        )
        rgb = np.clip(rgb * SH_C0 + 0.5, 0.0, 1.0)
        return np.asarray(np.round(rgb * 255.0), dtype=np.uint8)

    return np.tile(DEFAULT_RGB[None, :], (len(vertex), 1))


def _normalize_color_channel(channel: np.ndarray) -> np.ndarray:
    if channel.dtype.kind in {"f"}:
        max_value = float(np.max(channel)) if channel.size else 0.0
        if max_value <= 1.0:
            channel = channel * 255.0
    return np.asarray(np.clip(np.round(channel), 0, 255), dtype=np.uint8)


def _select_indices(
    vertex: np.ndarray,
    names: set[str],
    max_points: int,
    opacity_threshold: float,
) -> np.ndarray:
    total = len(vertex)
    if total <= max_points:
        return np.arange(total, dtype=np.int64)

    if "opacity" not in names:
        return np.linspace(0, total - 1, num=max_points, dtype=np.int64)

    opacity = np.asarray(vertex["opacity"], dtype=np.float32)
    opacity = 1.0 / (1.0 + np.exp(-opacity))
    valid = np.flatnonzero(opacity >= opacity_threshold)
    if len(valid) == 0:
        valid = np.argpartition(opacity, -max_points)[-max_points:]

    if len(valid) <= max_points:
        return np.asarray(valid, dtype=np.int64)

    order = np.argsort(opacity[valid])[::-1]
    ranked = valid[order]
    keep_top = min(max_points // 2, len(ranked))
    primary = ranked[:keep_top]
    remaining = ranked[keep_top:]
    secondary_count = max_points - keep_top
    if secondary_count > 0 and len(remaining) > 0:
        secondary = remaining[
            np.linspace(0, len(remaining) - 1, num=min(secondary_count, len(remaining)), dtype=np.int64)
        ]
        indices = np.concatenate([primary, secondary])
    else:
        indices = primary
    return np.asarray(indices[:max_points], dtype=np.int64)
