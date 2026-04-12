from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel


class Settings(BaseModel):
    llm_provider: str = "ollama"
    llm_model: str = "gemma3:4b"
    llm_base_url: str | None = "http://localhost:11434/v1"
    embedding_dim: int = 1536
    rules_dir: Path = Path("rules/core")
    default_token_budget: int = 2000
    router_unstructured_threshold: float = 0.7
    extraction_version: str = "0.1.0"

    @classmethod
    def from_env(cls) -> Settings:
        kwargs: dict = {}
        env_map = {
            "MNEMOSYNE_LLM_PROVIDER": "llm_provider",
            "MNEMOSYNE_LLM_MODEL": "llm_model",
            "MNEMOSYNE_LLM_BASE_URL": "llm_base_url",
            "MNEMOSYNE_EMBEDDING_DIM": "embedding_dim",
            "MNEMOSYNE_RULES_DIR": "rules_dir",
            "MNEMOSYNE_TOKEN_BUDGET": "default_token_budget",
            "MNEMOSYNE_ROUTER_UNSTRUCTURED_THRESHOLD": "router_unstructured_threshold",
            "MNEMOSYNE_EXTRACTION_VERSION": "extraction_version",
        }
        for env_key, field_name in env_map.items():
            val = os.environ.get(env_key)
            if val is not None:
                kwargs[field_name] = val
        return cls(**kwargs)
