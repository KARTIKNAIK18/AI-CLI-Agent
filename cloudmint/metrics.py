import threading
from typing import Dict


class Metrics:
    """In-memory metrics store with thread-safety."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Dict[str, int] = {
            "requests": 0,
            "jobs_submitted": 0,
            "jobs_started": 0,
            "jobs_completed": 0,
            "jobs_failed": 0,
            "workflows_triggered": 0,
            "agents_invoked": 0,
        }

    def inc(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._counters)
