# SAM 3D Objects Service Layer

This repository now includes a standalone service layer that wraps the official
SAM 3D Objects inference code without modifying the core inference pipeline.
Everything added for the service lives inside the `sam3d_service/` folder.

## Copy Rule

Copy the whole `sam3d_service/` directory into the root of an existing
`sam-3d-objects` checkout on the remote machine.

For a step-by-step remote deployment checklist, see `sam3d_service/DEPLOY.md`.

## Remote Setup

1. Prepare the official environment and checkpoints described in `doc/setup.md`.
2. Install the service dependencies:

   ```bash
   pip install -r sam3d_service/requirements.txt
   ```

3. Install Node.js and npm if you want the PlayCanvas Gaussian viewer preview.

4. Start the service:

   ```bash
   bash sam3d_service/start_service.sh
   ```

If the official environment and checkpoints are already ready on the remote
machine, copying this folder is enough to start the service.

By default the service listens on `0.0.0.0:8000`.

## Click Segmentation

The web UI now supports clicking on the uploaded image to generate a mask before
submitting the 3D job.

To enable that feature, install the extra dependency in
`sam3d_service/requirements.txt` and download a Segment Anything checkpoint.

Default expected checkpoint path:

```text
checkpoints/sam/sam_vit_h_4b8939.pth
```

You can override it with:

- `SAM3D_SEGMENT_CHECKPOINT`
- `SAM3D_SEGMENT_MODEL_TYPE` (default: `vit_h`)
- `SAM3D_SEGMENT_DEVICE` (default: same as `SAM3D_DEVICE`)

## Environment Variables

- `SAM3D_CHECKPOINT_TAG`: checkpoint folder under `checkpoints/` (default: `hf`)
- `SAM3D_DEVICE`: device string for health reporting (default: `cuda`)
- `SAM3D_DATA_DIR`: output directory for job files (default: `data/jobs`)
- `SAM3D_RENDER_GIF`: render an official-style GIF preview for single and scene jobs (default: `1`)
- `SAM3D_RENDER_GIF_RESOLUTION`: GIF render resolution (default: `512`)
- `SAM3D_RENDER_GIF_FRAMES`: GIF frame count (default: `120`)
- `SAM3D_RENDER_GIF_FPS`: GIF playback FPS (default: `30`)
- `SAM3D_PREVIEW_MAX_POINTS`: max points kept in browser preview PLY (default: `25000`)
- `SAM3D_PREVIEW_OPACITY_THRESHOLD`: minimum opacity kept when downsampling gaussian PLYs (default: `0.08`)
- `SAM3D_SUPERSPLAT_ENABLE`: build a PlayCanvas SuperSplat viewer for gaussian jobs (default: `1`)
- `SAM3D_SUPERSPLAT_COMMAND`: CLI used to generate the viewer (default: `npx -y @playcanvas/splat-transform@1.8.0`)
- `SAM3D_HOST`: service bind address (default: `0.0.0.0`)
- `SAM3D_PORT`: service port (default: `8000`)
- `SAM3D_SEGMENT_AUTO_POINTS_PER_SIDE`: SAM automatic mask sampling density (default: `24`)
- `SAM3D_SEGMENT_AUTO_MAX_CANDIDATES`: max automatic scene candidates returned to the web UI (default: `18`)
- `SAM3D_SEGMENT_AUTO_MIN_AREA_RATIO`: drop tiny masks below this image coverage ratio (default: `0.003`)
- `SAM3D_SEGMENT_AUTO_MAX_AREA_RATIO`: drop giant background masks above this image coverage ratio (default: `0.6`)
- `SAM3D_SEGMENT_AUTO_DEDUP_IOU`: suppress near-duplicate automatic candidates above this IoU (default: `0.9`)

## API

- `GET /` web UI
- `GET /healthz`
- `POST /segment/click`
- `POST /segment/auto`
- `POST /jobs`
- `POST /scene-jobs`
- `POST /alignment-jobs`
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/preview`
- `GET /jobs/{job_id}/artifacts/{name}`

`POST /jobs` expects `multipart/form-data` with:

- `image`: RGB image file
- `mask`: binary or grayscale mask file
- `seed`: optional integer, default `42`

`POST /segment/click` expects `multipart/form-data` with:

- `image`: RGB image file
- `x`: click x coordinate in original image pixels
- `y`: click y coordinate in original image pixels
- `label`: optional point label, default `1`

`POST /segment/auto` expects `multipart/form-data` with:

- `image`: RGB image file

It returns a filtered list of candidate instance masks for the scene workflow.

`POST /scene-jobs` expects `multipart/form-data` with:

- `image`: RGB image file
- `masks`: one or more mask image files
- `seed`: optional integer, default `42`

`POST /alignment-jobs` expects `multipart/form-data` with:

- `image`: RGB image file
- `mask`: binary or grayscale mask file
- `mesh`: input 3DB `.ply` mesh
- `focal_length_json`: optional JSON file with a `focal_length` field

## Local Client Example

The included client can submit a job and download the resulting `PLY` file:

```bash
python -m sam3d_service.client \
  --base-url http://127.0.0.1:8000 \
  --image notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png \
  --mask notebook/images/shutterstock_stylish_kidsroom_1640806567/14.png \
  --output result.ply
```

## Web UI

After the service starts, open:

```text
http://<server-ip>:8000/
```

The page lets you:

- run the single-object flow from `demo_single_object.ipynb`
- run the multi-object scene flow from `demo_multi_object.ipynb`
- run the 3DB mesh alignment flow from `demo_3db_mesh_alignment.ipynb`
- click a foreground object to generate a mask when Segment Anything is ready
- generate multiple automatic scene candidates, select the objects to keep, and use positive/negative points only for missing objects or mask refinement
- reuse one shared scene pointmap during multi-object reconstruction to reduce relative pose drift between selected objects
- anchor selected multi-object outputs back onto one estimated support plane when the photo implies a shared tabletop or floor
- inspect the notebook-style server-rendered GIF and MP4 preview for single and scene jobs
- monitor per-job progress and stage updates while gaussian inference is running
- open a separate PlayCanvas Gaussian viewer for single and scene jobs
- fall back to the old browser `PLY` preview when the PlayCanvas viewer is unavailable
- download all generated artifacts

The `PLY` preview page ships its `three.js` assets inside `sam3d_service/web/static/`,
so it does not depend on an external CDN.

Single-object and multi-object jobs now try to package a PlayCanvas SuperSplat
viewer from the original gaussian `PLY`. This requires `node`, `npm`, and a
working `npx` environment on the server.

If the PlayCanvas viewer build fails, the preview route falls back to the older
lightweight `preview.ply` browser view. The download link still points to the
original full-resolution result.

The alignment tab adds a `trimesh` dependency on top of the service layer.
