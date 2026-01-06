# Cloudmint Backend

Cloudmint is a production-focused backend that unifies REST APIs, background jobs, workflows, task queues, and AI agents in a single FastAPI service.

## Features
- REST endpoints for job submission, workflow registration/triggering, AI agent execution, health, and metrics.
- Background worker thread for asynchronous task processing with reliable, persisted state.
- Workflow engine to execute ordered steps (AI, sleep, or echo).
- Built-in observability via structured logging, request IDs, and metrics endpoint.

## Getting Started
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the service:
   ```bash
   python main.py
   ```

## Key Endpoints
- `POST /jobs` — enqueue a job with `type` and `payload`.
- `GET /jobs/{job_id}` — inspect job status/result.
- `POST /workflows` — register and trigger a workflow with ordered steps.
- `GET /workflows/{workflow_id}` — inspect workflow definition and runs.
- `POST /agents/run` — run an AI agent task asynchronously.
- `GET /health` — liveness/readiness summary.
- `GET /metrics` — counters for requests, jobs, workflows, and agents.

State is persisted to `cloudmint_state.json` by default. Configure behavior with environment variables:
- `CLOUDMINT_STATE_PATH` — override state file location.
- `CLOUDMINT_LOG_LEVEL` — logging level (default `INFO`).
- `CLOUDMINT_AGENT_MODE` — `local` (deterministic stub) or future remote modes.
- `CLOUDMINT_PERSIST_STATE` — disable disk persistence when set to `false`.
