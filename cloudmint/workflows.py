import asyncio
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from .agents import AIAgent
from .metrics import Metrics
from .state import JobStatus, StateStore, WorkflowStatus


class WorkflowEngine:
    """Simplified workflow engine executing steps within a job."""

    def __init__(self, state: StateStore, metrics: Metrics, agent: AIAgent) -> None:
        self.state = state
        self.metrics = metrics
        self.agent = agent

    def register(self, name: str, steps: List[Dict[str, Any]]) -> str:
        workflow_id = str(uuid.uuid4())
        self.state.register_workflow(workflow_id, {"name": name, "steps": steps})
        return workflow_id

    def trigger(self, workflow_id: str) -> Dict[str, Any]:
        workflow = self.state.get_workflow(workflow_id)
        run_id = str(uuid.uuid4())
        if workflow is None:
            raise ValueError("workflow not found")
        self.metrics.inc("workflows_triggered")
        # workflow execution happens inside a job; state updated in executor
        self.state.record_workflow_run(workflow_id, run_id, WorkflowStatus.TRIGGERED)
        return {"workflow_id": workflow_id, "run_id": run_id, "steps": workflow["definition"]["steps"]}

    async def execute_steps(self, workflow_id: str, run_id: str, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        for index, step in enumerate(steps):
            start = time.time()
            kind = step.get("type", "noop")
            payload = step.get("payload", {})
            if kind == "ai":
                outcome = await self.agent.run(payload.get("prompt", ""), context=payload.get("context"))
            elif kind == "sleep":
                duration = float(payload.get("duration", 0))
                await self._sleep(duration)
                outcome = {"slept_for": duration}
            else:
                outcome = {"echo": payload}
            results.append(
                {
                    "index": index,
                    "type": kind,
                    "result": outcome,
                    "duration": round(time.time() - start, 3),
                }
            )
        self.state.record_workflow_run(workflow_id, run_id, WorkflowStatus.COMPLETED, result=results)
        return {"steps": results}

    async def _sleep(self, duration: float) -> None:
        await asyncio.sleep(duration)


class TaskExecutor:
    """Entry point for worker thread to handle jobs and workflows."""

    def __init__(self, state: StateStore, metrics: Metrics, workflow_engine: WorkflowEngine, agent: AIAgent) -> None:
        self.state = state
        self.metrics = metrics
        self.workflow_engine = workflow_engine
        self.agent = agent
        self._thread_local = threading.local()

    def __call__(self, task: Dict[str, Any]) -> Any:
        job_type = task.get("type", "noop")
        payload = task.get("payload", {})
        if job_type == "workflow":
            return self._run_workflow(task, payload)
        if job_type == "agent":
            return self._run_agent_task(payload)
        return {"echo": payload}

    def _run_workflow(self, task: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        workflow_id = payload["workflow_id"]
        run_id = payload["run_id"]
        steps = payload.get("steps", [])
        result = self._run_coroutine(self.workflow_engine.execute_steps(workflow_id, run_id, steps))
        return result

    def _run_agent_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        prompt = payload.get("prompt", "")
        context = payload.get("context")
        self.metrics.inc("agents_invoked")
        response = self._run_coroutine(self.agent.run(prompt, context=context))
        return {"agent_response": response}

    def _run_coroutine(self, coroutine):
        loop = getattr(self._thread_local, "loop", None)
        if loop is None or loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._thread_local.loop = loop
        return loop.run_until_complete(coroutine)
