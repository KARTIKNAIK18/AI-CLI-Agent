import asyncio
import hashlib
from typing import Any, Dict, Optional

from .config import settings


class AIAgent:
    """Minimal AI agent stub with optional deterministic output."""

    async def run(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if settings.agent_mode == "local":
            digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
            await asyncio.sleep(0.01)
            return {
                "agent": "cloudmint-local",
                "summary": f"processed:{digest}",
                "context": context or {},
            }
        # Future extension: plug in hosted models via env configuration
        await asyncio.sleep(0.01)
        return {"agent": settings.agent_model or "cloudmint-generic", "echo": prompt, "context": context or {}}

