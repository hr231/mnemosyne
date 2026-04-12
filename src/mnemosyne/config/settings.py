from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel


class Settings(BaseModel):
    # LLM
    llm_provider: str = "ollama"
    llm_model: str = "gemma3:4b"
    llm_base_url: str | None = "http://localhost:11434/v1"
    llm_api_key: str | None = None

    # Embedding
    embedding_provider: str = "ollama"
    embedding_model: str = "nomic-embed-text"
    embedding_base_url: str = "http://localhost:11434"
    embedding_api_key: str | None = None
    embedding_dim: int = 768

    # Rules
    rules_dir: Path = Path("rules/core")

    # Context
    default_token_budget: int = 2000

    # Router
    router_unstructured_threshold: float = 0.7

    # Extraction
    extraction_version: str = "0.1.0"

    @property
    def llm_config(self) -> dict:
        return {
            "provider": self.llm_provider,
            "base_url": self.llm_base_url,
            "model": self.llm_model,
            "api_key": self.llm_api_key,
        }

    @property
    def embedding_config(self) -> dict:
        return {
            "provider": self.embedding_provider,
            "base_url": self.embedding_base_url,
            "model": self.embedding_model,
            "api_key": self.embedding_api_key,
            "dimensions": self.embedding_dim,
        }

    @classmethod
    def from_env(cls) -> Settings:
        kwargs: dict = {}
        env_map = {
            "MNEMOSYNE_LLM_PROVIDER": "llm_provider",
            "MNEMOSYNE_LLM_MODEL": "llm_model",
            "MNEMOSYNE_LLM_BASE_URL": "llm_base_url",
            "MNEMOSYNE_LLM_API_KEY": "llm_api_key",
            "MNEMOSYNE_EMBEDDING_PROVIDER": "embedding_provider",
            "MNEMOSYNE_EMBEDDING_MODEL": "embedding_model",
            "MNEMOSYNE_EMBEDDING_BASE_URL": "embedding_base_url",
            "MNEMOSYNE_EMBEDDING_API_KEY": "embedding_api_key",
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
