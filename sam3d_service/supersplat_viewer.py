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

    _inject_auto_rotate_controls(html_path)

    return SuperSplatArtifacts(
        html_name=html_path.name,
        sog_name=sog_path.name,
        js_name=js_path.name,
        css_name=css_path.name,
    )


def _inject_auto_rotate_controls(html_path: Path) -> None:
    html = html_path.read_text(encoding="utf-8")
    if "sam3d-auto-rotate-button" in html:
        return

    snippet = """
        <style>
            #sam3d-auto-rotate-wrap {
                position: fixed;
                right: 18px;
                bottom: 18px;
                z-index: 30;
                display: flex;
                gap: 10px;
                align-items: center;
                padding: 10px 12px;
                border-radius: 999px;
                background: rgba(10, 16, 20, 0.72);
                backdrop-filter: blur(10px);
                box-shadow: 0 16px 30px rgba(0, 0, 0, 0.24);
            }

            #sam3d-auto-rotate-button {
                appearance: none;
                border: 0;
                border-radius: 999px;
                padding: 10px 14px;
                color: #eef6f2;
                background: linear-gradient(135deg, #0f7a62, #0a5f4d);
                font: 600 14px/1 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                cursor: pointer;
            }

            #sam3d-auto-rotate-hint {
                color: rgba(255, 255, 255, 0.82);
                font: 500 12px/1.2 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                white-space: nowrap;
            }
        </style>
        <div id="sam3d-auto-rotate-wrap">
            <button id="sam3d-auto-rotate-button" type="button">Auto Rotate</button>
            <span id="sam3d-auto-rotate-hint">Slow orbit around the object</span>
        </div>
        <script>
            (() => {
                const button = document.getElementById('sam3d-auto-rotate-button');
                const hint = document.getElementById('sam3d-auto-rotate-hint');
                const canvas = document.getElementById('application-canvas');
                if (!button || !hint || !canvas) {
                    return;
                }

                const state = {
                    enabled: false,
                    rafId: null,
                    pointerX: 0,
                    pointerY: 0,
                    dragging: false,
                    lastTs: 0,
                };

                const dispatchMouse = (type, x, y, buttons) => {
                    canvas.dispatchEvent(new MouseEvent(type, {
                        bubbles: true,
                        cancelable: true,
                        view: window,
                        clientX: x,
                        clientY: y,
                        button: 0,
                        buttons,
                    }));
                };

                const resetPointer = () => {
                    const rect = canvas.getBoundingClientRect();
                    state.pointerX = rect.left + rect.width * 0.5;
                    state.pointerY = rect.top + rect.height * 0.5;
                };

                const startDrag = () => {
                    if (state.dragging) {
                        return;
                    }
                    resetPointer();
                    dispatchMouse('mousedown', state.pointerX, state.pointerY, 1);
                    state.dragging = true;
                };

                const stopDrag = () => {
                    if (!state.dragging) {
                        return;
                    }
                    dispatchMouse('mouseup', state.pointerX, state.pointerY, 0);
                    state.dragging = false;
                };

                const updateUi = () => {
                    button.textContent = state.enabled ? 'Stop Rotate' : 'Auto Rotate';
                    hint.textContent = state.enabled
                        ? 'Slow orbit is running'
                        : 'Slow orbit around the object';
                };

                const tick = (ts) => {
                    if (!state.enabled) {
                        return;
                    }
                    const controlsWrap = document.getElementById('controlsWrap');
                    if (!controlsWrap || controlsWrap.classList.contains('hidden')) {
                        state.rafId = window.requestAnimationFrame(tick);
                        return;
                    }

                    if (!state.dragging) {
                        startDrag();
                    }

                    if (!state.lastTs || ts - state.lastTs >= 24) {
                        const rect = canvas.getBoundingClientRect();
                        state.pointerX += 1.5;
                        if (state.pointerX > rect.left + rect.width * 0.72) {
                            stopDrag();
                            startDrag();
                        } else {
                            dispatchMouse('mousemove', state.pointerX, state.pointerY, 1);
                        }
                        state.lastTs = ts;
                    }
                    state.rafId = window.requestAnimationFrame(tick);
                };

                const stopAutoRotate = () => {
                    state.enabled = false;
                    if (state.rafId !== null) {
                        window.cancelAnimationFrame(state.rafId);
                        state.rafId = null;
                    }
                    stopDrag();
                    updateUi();
                };

                const startAutoRotate = () => {
                    if (state.enabled) {
                        return;
                    }
                    state.enabled = true;
                    state.lastTs = 0;
                    updateUi();
                    state.rafId = window.requestAnimationFrame(tick);
                };

                button.addEventListener('click', () => {
                    if (state.enabled) {
                        stopAutoRotate();
                    } else {
                        startAutoRotate();
                    }
                });

                canvas.addEventListener('pointerdown', () => {
                    if (state.enabled) {
                        stopAutoRotate();
                    }
                });

                window.addEventListener('blur', stopAutoRotate);
                window.addEventListener('beforeunload', stopAutoRotate);
                updateUi();
            })();
        </script>
    """
    html = html.replace("</body>", f"{snippet}\n    </body>")
    html_path.write_text(html, encoding="utf-8")
