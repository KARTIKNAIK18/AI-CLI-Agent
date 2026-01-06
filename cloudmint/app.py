import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .agents import AIAgent
from .config import settings
from .metrics import Metrics
from .observability import RequestMetricsMiddleware, setup_logging
from .queue import TaskQueue
from .state import StateStore
from .workflows import TaskExecutor, WorkflowEngine


class JobRequest(BaseModel):
    type: str = Field("agent", description="Type of job to run")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Job payload")


class JobResponse(BaseModel):
    id: str
    status: str


class WorkflowRequest(BaseModel):
    name: str
    steps: List[Dict[str, Any]]


class WorkflowTriggerResponse(BaseModel):
    workflow_id: str
    run_id: str


class AgentRequest(BaseModel):
    prompt: str
    context: Optional[Dict[str, Any]] = None


def get_app() -> FastAPI:
    setup_logging(settings.log_level)
    metrics = Metrics()
    state = StateStore()
    agent = AIAgent()
    workflow_engine = WorkflowEngine(state, metrics, agent)
    executor = TaskExecutor(state, metrics, workflow_engine, agent)
    task_queue = TaskQueue(state, metrics, executor)

    app = FastAPI(
        title="Cloudmint",
        version="1.0.0",
        description="Unified REST, jobs, workflows, and AI agent backend.",
    )
    app.add_middleware(RequestMetricsMiddleware, metrics=metrics)

    @app.on_event("shutdown")
    def _shutdown() -> None:
        task_queue.stop()

    def get_state() -> StateStore:
        return state

    def get_queue() -> TaskQueue:
        return task_queue

    def get_workflow_engine() -> WorkflowEngine:
        return workflow_engine

    @app.get("/health")
    def health(state_store: StateStore = Depends(get_state)) -> Dict[str, Any]:
        snapshot = state_store.snapshot()
        return {"status": "ok", "jobs": len(snapshot["jobs"]), "workflows": len(snapshot["workflows"])}

    @app.get("/metrics")
    def metrics_endpoint() -> Dict[str, Any]:
        return metrics.snapshot()

    @app.post("/jobs", response_model=JobResponse)
    def enqueue_job(request: JobRequest, queue: TaskQueue = Depends(get_queue), state_store: StateStore = Depends(get_state)) -> JobResponse:
        job_id = str(uuid.uuid4())
        payload = {"type": request.type, "payload": request.payload, "id": job_id}
        state_store.create_job(job_id, payload)
        queue.submit(payload)
        return JobResponse(id=job_id, status="queued")

    @app.get("/jobs/{job_id}", response_model=Dict[str, Any])
    def get_job(job_id: str, state_store: StateStore = Depends(get_state)) -> Dict[str, Any]:
        job = state_store.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        return job

    @app.post("/workflows", response_model=WorkflowTriggerResponse)
    def create_workflow(request: WorkflowRequest, queue: TaskQueue = Depends(get_queue), state_store: StateStore = Depends(get_state), wf_engine: WorkflowEngine = Depends(get_workflow_engine)) -> WorkflowTriggerResponse:
        workflow_id = wf_engine.register(request.name, request.steps)
        trigger_payload = wf_engine.trigger(workflow_id)
        job_id = str(uuid.uuid4())
        task_payload = {"type": "workflow", "payload": trigger_payload, "id": job_id}
        state_store.create_job(job_id, task_payload)
        queue.submit(task_payload)
        return WorkflowTriggerResponse(workflow_id=workflow_id, run_id=trigger_payload["run_id"])

    @app.get("/workflows/{workflow_id}", response_model=Dict[str, Any])
    def get_workflow(workflow_id: str, state_store: StateStore = Depends(get_state)) -> Dict[str, Any]:
        workflow = state_store.get_workflow(workflow_id)
        if not workflow:
            raise HTTPException(status_code=404, detail="workflow not found")
        return workflow

    @app.post("/agents/run")
    def run_agent(request: AgentRequest, queue: TaskQueue = Depends(get_queue), state_store: StateStore = Depends(get_state)) -> JSONResponse:
        job_id = str(uuid.uuid4())
        payload = {"type": "agent", "payload": request.model_dump(), "id": job_id}
        state_store.create_job(job_id, payload)
        queue.submit(payload)
        return JSONResponse({"job_id": job_id, "status": "queued"})

    return app


app = get_app()

