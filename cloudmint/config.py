import os
from typing import Optional


def get_env_bool(key: str, default: bool = False) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


class Settings:
    """Runtime configuration for Cloudmint."""

    def __init__(self) -> None:
        self.state_path: str = os.getenv("CLOUDMINT_STATE_PATH", "cloudmint_state.json")
        self.log_level: str = os.getenv("CLOUDMINT_LOG_LEVEL", "INFO").upper()
        self.agent_mode: str = os.getenv("CLOUDMINT_AGENT_MODE", "local")
        self.agent_model: Optional[str] = os.getenv("CLOUDMINT_AGENT_MODEL")
        self.enable_file_persistence: bool = get_env_bool("CLOUDMINT_PERSIST_STATE", True)


settings = Settings()
