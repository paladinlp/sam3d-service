from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
from shutil import which
import subprocess

from sam3d_service.storage import (
    SUPERSPLAT_CSS_NAME,
    SUPERSPLAT_HTML_NAME,
    SUPERSPLAT_JS_NAME,
    SUPERSPLAT_SOG_NAME,
)


@dataclass(frozen=True)
class SuperSplatArtifacts:
    html_name: str
    sog_name: str
    js_name: str
    css_name: str


def build_supersplat_viewer(
    source_path: Path,
    output_dir: Path,
    command: str,
) -> SuperSplatArtifacts:
    output_dir = output_dir.resolve()
    source_path = source_path.resolve()
    command_parts = shlex.split(command)
    if not command_parts:
        raise RuntimeError("SuperSplat command is empty.")
    if which(command_parts[0]) is None:
        raise RuntimeError(f"SuperSplat command is unavailable: {command_parts[0]}")

    html_path = output_dir / SUPERSPLAT_HTML_NAME
    sog_path = output_dir / SUPERSPLAT_SOG_NAME
    js_path = output_dir / SUPERSPLAT_JS_NAME
    css_path = output_dir / SUPERSPLAT_CSS_NAME

    for path in (html_path, sog_path, js_path, css_path):
        path.unlink(missing_ok=True)

    process = subprocess.run(
        [
            *command_parts,
            "-w",
            "-U",
            str(source_path),
            str(html_path),
        ],
        cwd=output_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        stderr = process.stderr.strip()
        stdout = process.stdout.strip()
        detail = stderr or stdout or "Unknown splat-transform failure."
        raise RuntimeError(f"SuperSplat viewer generation failed: {detail}")

    missing = [path.name for path in (html_path, sog_path, js_path, css_path) if not path.is_file()]
    if missing:
        raise RuntimeError(
            "SuperSplat viewer generation produced incomplete artifacts: "
            + ", ".join(missing)
        )

    return SuperSplatArtifacts(
        html_name=html_path.name,
        sog_name=sog_path.name,
        js_name=js_path.name,
        css_name=css_path.name,
    )
