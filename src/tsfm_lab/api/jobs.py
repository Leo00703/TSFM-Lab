"""Background jobs (experiments, benchmarks, weight downloads). One worker: compute is serialised so that
timings are not distorted by concurrent runs, and the GPU is never shared between two jobs."""

from __future__ import annotations

import secrets
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable


class JobCancelled(Exception):
    pass


@dataclass
class Job:
    id: str
    kind: str
    label: str
    status: str = "queued"  # queued | running | done | error | cancelled
    done: int = 0
    total: int = 0
    message: str = ""
    result: dict[str, Any] | None = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    def update(self, done: int | None = None, total: int | None = None, message: str | None = None) -> None:
        if self.cancel.is_set():
            raise JobCancelled()
        if done is not None:
            self.done = done
        if total is not None:
            self.total = total
        if message is not None:
            self.message = message

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "label": self.label, "status": self.status, "done": self.done,
                "total": self.total, "message": self.message, "result": self.result, "error": self.error,
                "created": self.created, "started": self.started, "finished": self.finished}


class JobManager:
    def __init__(self, compute_lock: threading.Lock) -> None:
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tsfm-job")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self.compute_lock = compute_lock

    def submit(self, kind: str, label: str, fn: Callable[[Job], dict[str, Any] | None]) -> Job:
        job = Job(secrets.token_hex(6), kind, label)
        with self._lock:
            self._jobs[job.id] = job
            for old in sorted(self._jobs.values(), key=lambda j: j.created)[:-50]:
                if old.status in ("done", "error", "cancelled"):
                    self._jobs.pop(old.id, None)
        self._pool.submit(self._run, job, fn)
        return job

    def _run(self, job: Job, fn: Callable[[Job], dict[str, Any] | None]) -> None:
        if job.cancel.is_set():
            job.status, job.finished = "cancelled", time.time()
            return
        job.status, job.started = "running", time.time()
        try:
            with self.compute_lock:
                job.result = fn(job)
            job.status = "done"
        except JobCancelled:
            job.status = "cancelled"
        except Exception as error:  # noqa: BLE001 - surfaced to the user, details kept in the message
            job.status = "error"
            job.error = f"{type(error).__name__}: {error}"
            job.message = traceback.format_exc(limit=3)
        finally:
            job.finished = time.time()

    def get(self, job_id: str) -> Job:
        try:
            return self._jobs[job_id]
        except KeyError:
            raise KeyError(f"unknown job '{job_id}'") from None

    def list(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        job.cancel.set()
        return job
