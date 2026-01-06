import logging
import queue
import threading
import time
from typing import Any, Callable, Dict

from .metrics import Metrics
from .state import JobStatus, StateStore


TaskHandler = Callable[[Dict[str, Any]], Any]


class TaskQueue:
    """Lightweight task queue with a single worker thread."""

    def __init__(self, state: StateStore, metrics: Metrics, handler: TaskHandler) -> None:
        self._state = state
        self._metrics = metrics
        self._handler = handler
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def submit(self, task: Dict[str, Any]) -> None:
        self._queue.put(task)
        self._metrics.inc("jobs_submitted")

    def _run(self) -> None:
        log = logging.getLogger("cloudmint.worker")
        while not self._stop.is_set():
            try:
                task = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            job_id = task.get("id")
            if job_id:
                self._state.mark_job(job_id, JobStatus.RUNNING)
                self._metrics.inc("jobs_started")
            try:
                result = self._handler(task)
                if job_id:
                    self._state.mark_job(job_id, JobStatus.COMPLETED, result=result)
                    self._metrics.inc("jobs_completed")
            except Exception as exc:  # pylint: disable=broad-except
                log.exception("task failed", extra={"job_id": job_id})
                if job_id:
                    self._state.mark_job(job_id, JobStatus.FAILED, error=str(exc))
                    self._metrics.inc("jobs_failed")
            finally:
                self._queue.task_done()

    def stop(self) -> None:
        self._stop.set()
        self._worker.join(timeout=2)

    def size(self) -> int:
        return self._queue.qsize()

