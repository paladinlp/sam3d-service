from __future__ import annotations

import asyncio
from dataclasses import dataclass
import time
import traceback

from sam3d_service.runner import InferenceRunner
from sam3d_service.storage import JobStore


@dataclass(frozen=True)
class JobRequest:
    job_id: str


class InferenceWorker:
    def __init__(self, store: JobStore, runner: InferenceRunner):
        self.store = store
        self.runner = runner
        self.queue: asyncio.Queue[JobRequest] = asyncio.Queue()
        self._consumer_task: asyncio.Task | None = None

    async def start(self) -> None:
        await asyncio.to_thread(self.runner.load_model)
        self._consumer_task = asyncio.create_task(self._consume(), name="sam3d-worker")

    async def stop(self) -> None:
        if self._consumer_task is None:
            return
        self._consumer_task.cancel()
        try:
            await self._consumer_task
        except asyncio.CancelledError:
            pass

    async def submit(self, job_id: str) -> None:
        await self.queue.put(JobRequest(job_id=job_id))

    async def _consume(self) -> None:
        while True:
            request = await self.queue.get()
            try:
                await asyncio.to_thread(self._process_job, request.job_id)
            finally:
                self.queue.task_done()

    def _process_job(self, job_id: str) -> None:
        started = time.perf_counter()
        self.store.mark_running(job_id)
        try:
            job = self.store.read_job(job_id)
            result = self.runner.run_job(
                self.store.job_paths(job_id)["job_dir"],
                seed=int(job.get("seed", 42)),
            )
            result.setdefault("timings", {})["total_seconds"] = round(
                time.perf_counter() - started,
                3,
            )
            self.store.mark_succeeded(job_id, result)
        except Exception:
            self.store.mark_failed(job_id, traceback.format_exc())
