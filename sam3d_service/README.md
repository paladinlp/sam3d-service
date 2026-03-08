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

3. Start the service:

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
- `SAM3D_HOST`: service bind address (default: `0.0.0.0`)
- `SAM3D_PORT`: service port (default: `8000`)

## API

- `GET /` web UI
- `GET /healthz`
- `POST /segment/click`
- `POST /jobs`
- `GET /jobs/{job_id}`
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

- upload an image
- click a foreground object to generate a mask
- optionally upload a manual mask instead
- submit a job
- poll its status
- download the generated `PLY`
