# Remote Deployment Checklist

This checklist assumes:

- the remote machine is Linux
- NVIDIA GPU is available
- you already have an official `sam-3d-objects` checkout on the server
- you will copy the whole `sam3d_service/` folder into that checkout

Example repo root used below:

```bash
cd /path/to/sam-3d-objects
```

## 1. Verify the base environment

On the remote machine, confirm:

```bash
nvidia-smi
python --version
```

If the official environment is not ready yet, follow the upstream setup first:

```bash
cat doc/setup.md
```

## 2. Copy the service folder

Copy this folder into the remote repo root:

```text
sam3d_service/
```

After copying, the remote layout should contain:

```text
sam-3d-objects/
├── sam3d_service/
├── sam3d_objects/
├── notebook/
├── checkpoints/
└── ...
```

## 3. Install service dependencies

Activate the same Python environment used by `sam-3d-objects`, then run:

```bash
pip install -r sam3d_service/requirements.txt
```

This installs:

- FastAPI
- Uvicorn
- multipart upload support
- requests
- trimesh
- Segment Anything

## 4. Prepare SAM 3D Objects checkpoints

The 3D service will not start unless this file exists:

```text
checkpoints/hf/pipeline.yaml
```

Quick check:

```bash
ls checkpoints/hf/pipeline.yaml
```

If missing, download the official gated checkpoints first.

## 5. Prepare click-segmentation checkpoint

Click-to-select in the web UI needs a Segment Anything checkpoint.

Default expected path:

```text
checkpoints/sam/sam_vit_h_4b8939.pth
```

Create the directory if needed:

```bash
mkdir -p checkpoints/sam
```

Place the checkpoint there, or override with:

```bash
export SAM3D_SEGMENT_CHECKPOINT=/absolute/path/to/sam_checkpoint.pth
```

Optional overrides:

```bash
export SAM3D_SEGMENT_MODEL_TYPE=vit_h
export SAM3D_SEGMENT_DEVICE=cuda
```

If this checkpoint is missing:

- the main 3D service can still start
- the `/segment/click` feature will return unavailable
- manual mask upload still works

## 6. Optional environment variables

Defaults are usually enough, but you can set:

```bash
export SAM3D_CHECKPOINT_TAG=hf
export SAM3D_DEVICE=cuda
export SAM3D_DATA_DIR=/path/to/job-output
export SAM3D_HOST=0.0.0.0
export SAM3D_PORT=8000
```

## 7. Start the service

From the repo root:

```bash
bash sam3d_service/start_service.sh
```

Expected behavior:

- the service loads the 3D model on startup
- it serves the web UI at `/`
- it exposes tabs for single object, multi-object scene, and 3DB alignment
- it exposes API routes for health, click segmentation, and jobs

## 8. Verify the service

Health check:

```bash
curl http://127.0.0.1:8000/healthz
```

Expected fields:

- `model_loaded`
- `checkpoint_ready`
- `click_segmentation_checkpoint_ready`

Open the web UI from your browser:

```text
http://<server-ip>:8000/
```

## 9. Network checks

If you cannot access the page remotely, verify:

- the process is listening on `0.0.0.0:8000`
- the cloud security group allows TCP 8000
- the host firewall allows TCP 8000

Useful checks:

```bash
ss -ltnp | grep 8000
curl http://127.0.0.1:8000/healthz
```

## 10. Minimal smoke test

Manual mask route:

1. open the web UI
2. upload an image
3. upload a mask
4. submit a 3D job

Click-selection route:

1. open the web UI
2. upload an image
3. click the target object
4. wait for generated mask
5. submit the 3D job

Expected result:

- a job is created
- status moves `queued -> running -> succeeded`
- `PLY` becomes downloadable

## 11. Job outputs

By default, outputs are written under:

```text
data/jobs/<job_id>/
```

Each job directory may contain:

- `input.png`
- `mask.png`
- `mask_000.png`, `mask_001.png`, ...
- `input_mesh.ply`
- `focal_length.json`
- `result.ply`
- `scene_result.ply`
- `aligned_mesh.ply`
- `result.json`
- `error.txt`

## 12. Common failure cases

### Missing 3D checkpoint

Symptom:

```text
Missing pipeline config: .../checkpoints/hf/pipeline.yaml
```

Fix:

- download the official SAM 3D Objects checkpoints

### Click segmentation unavailable

Symptom:

- web page opens
- clicking image fails
- health response shows `click_segmentation_checkpoint_ready=false`

Fix:

- place the Segment Anything checkpoint in `checkpoints/sam/`
- or set `SAM3D_SEGMENT_CHECKPOINT`

### Import error for Segment Anything

Symptom:

- click segmentation endpoint returns dependency error

Fix:

```bash
pip install -r sam3d_service/requirements.txt
```

### Remote page unreachable

Fix:

- confirm `SAM3D_HOST=0.0.0.0`
- confirm firewall/security group allows port `8000`

## 13. Useful run modes

Foreground run:

```bash
bash sam3d_service/start_service.sh
```

Background run with log file:

```bash
nohup bash sam3d_service/start_service.sh > sam3d_service.log 2>&1 &
tail -f sam3d_service.log
```
