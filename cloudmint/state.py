import json
import os
import threading
import time
from enum import Enum
from typing import Any, Dict, Optional

from .config import settings


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkflowStatus(str, Enum):
    REGISTERED = "registered"
    TRIGGERED = "triggered"
    COMPLETED = "completed"
    FAILED = "failed"


class StateStore:
    """Thread-safe state persistence with optional disk durability."""

    def __init__(self, path: str = settings.state_path) -> None:
        self._path = path
        self._lock = threading.RLock()
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.workflows: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if settings.enable_file_persistence and os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                self.jobs = data.get("jobs", {})
                self.workflows = data.get("workflows", {})
            except Exception:
                # Fallback to empty in case of corruption, preserving startup
                self.jobs = {}
                self.workflows = {}

    def _persist(self) -> None:
        if not settings.enable_file_persistence:
            return
        snapshot = {"jobs": self.jobs, "workflows": self.workflows}
        tmp_path = f"{self._path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(snapshot, handle, indent=2)
        os.replace(tmp_path, self._path)

    def create_job(self, job_id: str, payload: Dict[str, Any]) -> None:
        with self._lock:
            self.jobs[job_id] = {
                "id": job_id,
                "payload": payload,
                "status": JobStatus.QUEUED.value,
                "created_at": time.time(),
            }
            self._persist()

    def mark_job(self, job_id: str, status: JobStatus, result: Optional[Any] = None, error: Optional[str] = None) -> None:
        with self._lock:
            if job_id not in self.jobs:
                return
            job = self.jobs[job_id]
            job["status"] = status.value
            if result is not None:
                job["result"] = result
            if error is not None:
                job["error"] = error
            job["updated_at"] = time.time()
            self._persist()

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self.jobs.get(job_id)

    def register_workflow(self, workflow_id: str, definition: Dict[str, Any]) -> None:
        with self._lock:
            self.workflows[workflow_id] = {
                "id": workflow_id,
                "definition": definition,
                "status": WorkflowStatus.REGISTERED.value,
                "runs": [],
                "created_at": time.time(),
            }
            self._persist()

    def record_workflow_run(self, workflow_id: str, run_id: str, status: WorkflowStatus, result: Optional[Any] = None, error: Optional[str] = None) -> None:
        with self._lock:
            workflow = self.workflows.get(workflow_id)
            if workflow is None:
                return
            run_entry = {
                "run_id": run_id,
                "status": status.value,
                "result": result,
                "error": error,
                "timestamp": time.time(),
            }
            workflow["runs"].append(run_entry)
            workflow["status"] = status.value
            self._persist()

    def get_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self.workflows.get(workflow_id)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {"jobs": dict(self.jobs), "workflows": dict(self.workflows)}
