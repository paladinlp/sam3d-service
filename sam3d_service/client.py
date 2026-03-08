from __future__ import annotations

import argparse
from pathlib import Path
import time

import requests


class Sam3DServiceClient:
    def __init__(self, base_url: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def submit(self, image_path: str | Path, mask_path: str | Path, seed: int = 42) -> dict:
        with open(image_path, "rb") as image_file, open(mask_path, "rb") as mask_file:
            response = requests.post(
                f"{self.base_url}/jobs",
                files={
                    "image": (Path(image_path).name, image_file, "image/png"),
                    "mask": (Path(mask_path).name, mask_file, "image/png"),
                },
                data={"seed": str(seed)},
                timeout=self.timeout,
            )
        response.raise_for_status()
        return response.json()

    def get_job(self, job_id: str) -> dict:
        response = requests.get(
            f"{self.base_url}/jobs/{job_id}",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def wait(self, job_id: str, poll_interval: float = 5.0) -> dict:
        while True:
            payload = self.get_job(job_id)
            if payload["status"] in {"succeeded", "failed"}:
                return payload
            time.sleep(poll_interval)

    def download(self, url: str, output_path: str | Path) -> Path:
        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(response.content)
        return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit a SAM 3D Objects job.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="result.ply")
    parser.add_argument("--poll-interval", type=float, default=5.0)
    args = parser.parse_args()

    client = Sam3DServiceClient(args.base_url)
    job = client.submit(args.image, args.mask, seed=args.seed)
    result = client.wait(job["job_id"], poll_interval=args.poll_interval)
    if result["status"] != "succeeded":
        raise SystemExit(f"Job failed: {result.get('error')}")
    output_path = client.download(result["result"]["artifacts"]["result_ply"], args.output)
    print(f"Downloaded result to {output_path}")


if __name__ == "__main__":
    main()
