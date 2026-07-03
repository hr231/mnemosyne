from __future__ import annotations

import logging
import os
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MonitoringConfig(BaseModel):
    """Runtime settings for the Prometheus metrics exporter."""

    enabled: bool = True
    port: int = 9090


class StartupConfig(BaseModel):
    """Settings governing startup sanity checks."""

    fail_fast: bool = True
    required_env_vars: list[str] = Field(
        default_factory=lambda: ["MNEMOSYNE_PG_DSN"]
    )


class MnemosyneConfig(BaseModel):
    """Top-level YAML-backed configuration for the memory subsystem."""

    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    startup: StartupConfig = Field(default_factory=StartupConfig)


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
    embedding_dim: int = Field(768, gt=0)

    # Rules
    rules_dir: Path = Path("rules/core")

    # Context
    default_token_budget: int = Field(2000, gt=0)
    token_encoding: str = "cl100k_base"

    # Router
    router_unstructured_threshold: float = Field(0.7, ge=0.0, le=1.0)
    router_yield_threshold: float = Field(0.3, ge=0.0, le=1.0)

    # Retrieval / full-text search
    fts_language: str = "english"

    # Extraction
    extraction_version: str = "0.1.0"

    # LLM / embedding resilience
    llm_timeout_s: float = Field(60.0, gt=0.0)
    llm_max_retries: int = Field(3, ge=0)
    llm_max_concurrency: int = Field(4, ge=1)
    embedding_timeout_s: float = Field(30.0, gt=0.0)
    embedding_max_retries: int = Field(3, ge=0)

    # Background worker
    worker_poll_interval_s: float = Field(10.0, gt=0.0)
    maintenance_interval_s: float = Field(86400.0, gt=0.0)
    processing_visibility_timeout_s: float = Field(600.0, gt=0.0)

    # Postgres connection pool tuning
    pg_pool_min_size: int = Field(2, ge=0)
    pg_pool_max_size: int = Field(10, ge=1)
    pg_command_timeout_s: float = Field(60.0, gt=0.0)

    # Failure isolation
    memory_fail_open: bool = True

    # save_memory tool input hardening
    save_memory_content_cap: int = Field(10_000, gt=0)

    @property
    def llm_config(self) -> dict:
        return {
            "provider": self.llm_provider,
            "base_url": self.llm_base_url,
            "model": self.llm_model,
            "api_key": self.llm_api_key,
            "timeout": self.llm_timeout_s,
            "max_retries": self.llm_max_retries,
        }

    @property
    def embedding_config(self) -> dict:
        return {
            "provider": self.embedding_provider,
            "base_url": self.embedding_base_url,
            "model": self.embedding_model,
            "api_key": self.embedding_api_key,
            "dimensions": self.embedding_dim,
            "timeout": self.embedding_timeout_s,
            "max_retries": self.embedding_max_retries,
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
            "MNEMOSYNE_DEFAULT_TOKEN_BUDGET": "default_token_budget",
            "MNEMOSYNE_TOKEN_BUDGET": "default_token_budget",
            "MNEMOSYNE_TOKEN_ENCODING": "token_encoding",
            "MNEMOSYNE_ROUTER_UNSTRUCTURED_THRESHOLD": "router_unstructured_threshold",
            "MNEMOSYNE_ROUTER_YIELD_THRESHOLD": "router_yield_threshold",
            "MNEMOSYNE_FTS_LANGUAGE": "fts_language",
            "MNEMOSYNE_EXTRACTION_VERSION": "extraction_version",
            "MNEMOSYNE_LLM_TIMEOUT_S": "llm_timeout_s",
            "MNEMOSYNE_LLM_MAX_RETRIES": "llm_max_retries",
            "MNEMOSYNE_LLM_MAX_CONCURRENCY": "llm_max_concurrency",
            "MNEMOSYNE_EMBEDDING_TIMEOUT_S": "embedding_timeout_s",
            "MNEMOSYNE_EMBEDDING_MAX_RETRIES": "embedding_max_retries",
            "MNEMOSYNE_WORKER_POLL_INTERVAL_S": "worker_poll_interval_s",
            "MNEMOSYNE_MAINTENANCE_INTERVAL_S": "maintenance_interval_s",
            "MNEMOSYNE_PROCESSING_VISIBILITY_TIMEOUT_S": "processing_visibility_timeout_s",
            "MNEMOSYNE_PG_POOL_MIN_SIZE": "pg_pool_min_size",
            "MNEMOSYNE_PG_POOL_MAX_SIZE": "pg_pool_max_size",
            "MNEMOSYNE_PG_COMMAND_TIMEOUT_S": "pg_command_timeout_s",
            "MNEMOSYNE_MEMORY_FAIL_OPEN": "memory_fail_open",
            "MNEMOSYNE_SAVE_MEMORY_CONTENT_CAP": "save_memory_content_cap",
        }
        for env_key, field_name in env_map.items():
            val = os.environ.get(env_key)
            if val is not None:
                kwargs[field_name] = val

        cls._warn_unrecognized_env(env_map)
        return cls(**kwargs)

    @staticmethod
    def _warn_unrecognized_env(env_map: dict[str, str]) -> None:
        """Log a warning for any ``MNEMOSYNE_*`` var that Settings ignores.

        Operational and deployment vars consumed elsewhere (DSN, metrics,
        startup toggles, dev/test flags) are allow-listed so only genuine
        typos in Settings keys surface.
        """
        recognized = set(env_map) | _OPERATIONAL_ENV_VARS
        for key in os.environ:
            if key.startswith("MNEMOSYNE_") and key not in recognized:
                logger.warning(
                    "Ignoring unrecognized environment variable %s "
                    "(not a Settings field)",
                    key,
                )


_OPERATIONAL_ENV_VARS = {
    "MNEMOSYNE_PG_DSN",
    "MNEMOSYNE_SESSION_PERSIST",
    "MNEMOSYNE_STARTUP_WARN_ONLY",
    "MNEMOSYNE_METRICS_ENABLED",
    "MNEMOSYNE_METRICS_ADDR",
    "MNEMOSYNE_METRICS_PORT",
    "MNEMOSYNE_GH_USER",
    "MNEMOSYNE_GH_REPO",
    "MNEMOSYNE_LLM_INTEGRATION",
    "MNEMOSYNE_EMBEDDING_INTEGRATION",
    "MNEMOSYNE_FULL_INTEGRATION",
    "MNEMOSYNE_NLI",
    "MNEMOSYNE_NER",
}
