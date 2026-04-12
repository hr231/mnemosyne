"""Unit tests for the Settings configuration model."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from mnemosyne.config.settings import Settings


class TestSettingsDefaults:
    def test_defaults(self):
        """Settings() should have all expected default values."""
        s = Settings()

        assert s.llm_provider == "ollama"
        assert s.llm_model == "gemma3:4b"
        assert s.llm_base_url == "http://localhost:11434/v1"
        assert s.embedding_dim == 768
        assert s.rules_dir == Path("rules/core")
        assert s.default_token_budget == 2000
        assert s.router_unstructured_threshold == 0.7
        assert s.extraction_version == "0.1.0"

    def test_default_rules_dir_is_path(self):
        """rules_dir must be a Path, not a bare string."""
        s = Settings()
        assert isinstance(s.rules_dir, Path)

    def test_default_llm_base_url_not_none(self):
        """Default llm_base_url points to local Ollama endpoint."""
        s = Settings()
        assert s.llm_base_url is not None
        assert "11434" in s.llm_base_url


class TestSettingsFromEnv:
    def test_from_env(self, monkeypatch):
        """Settings.from_env() should read MNEMOSYNE_EXTRACTION_VERSION from env."""
        monkeypatch.setenv("MNEMOSYNE_EXTRACTION_VERSION", "0.2.0")

        s = Settings.from_env()

        assert s.extraction_version == "0.2.0"

    def test_from_env_llm_provider(self, monkeypatch):
        """Settings.from_env() should read MNEMOSYNE_LLM_PROVIDER."""
        monkeypatch.setenv("MNEMOSYNE_LLM_PROVIDER", "openai")

        s = Settings.from_env()

        assert s.llm_provider == "openai"

    def test_from_env_llm_model(self, monkeypatch):
        """Settings.from_env() should read MNEMOSYNE_LLM_MODEL."""
        monkeypatch.setenv("MNEMOSYNE_LLM_MODEL", "gpt-4o-mini")

        s = Settings.from_env()

        assert s.llm_model == "gpt-4o-mini"

    def test_from_env_embedding_dim(self, monkeypatch):
        """MNEMOSYNE_EMBEDDING_DIM should be read and coerced to int."""
        monkeypatch.setenv("MNEMOSYNE_EMBEDDING_DIM", "768")

        s = Settings.from_env()

        assert s.embedding_dim == 768

    def test_from_env_token_budget(self, monkeypatch):
        """MNEMOSYNE_TOKEN_BUDGET should be read and coerced to int."""
        monkeypatch.setenv("MNEMOSYNE_TOKEN_BUDGET", "4000")

        s = Settings.from_env()

        assert s.default_token_budget == 4000

    def test_from_env_rules_dir(self, monkeypatch):
        """MNEMOSYNE_RULES_DIR should be read and coerced to Path."""
        monkeypatch.setenv("MNEMOSYNE_RULES_DIR", "custom/rules")

        s = Settings.from_env()

        assert s.rules_dir == Path("custom/rules")

    def test_from_env_defaults(self):
        """Settings.from_env() with no env vars set should return defaults."""
        # Ensure no MNEMOSYNE_* vars are set in the environment
        env_keys = [
            "MNEMOSYNE_LLM_PROVIDER",
            "MNEMOSYNE_LLM_MODEL",
            "MNEMOSYNE_LLM_BASE_URL",
            "MNEMOSYNE_EMBEDDING_DIM",
            "MNEMOSYNE_RULES_DIR",
            "MNEMOSYNE_TOKEN_BUDGET",
            "MNEMOSYNE_ROUTER_UNSTRUCTURED_THRESHOLD",
            "MNEMOSYNE_EXTRACTION_VERSION",
        ]
        # Save and clear
        saved = {k: os.environ.pop(k) for k in env_keys if k in os.environ}

        try:
            s = Settings.from_env()
            assert s.llm_provider == "ollama"
            assert s.llm_model == "gemma3:4b"
            assert s.extraction_version == "0.1.0"
            assert s.embedding_dim == 768
            assert s.default_token_budget == 2000
        finally:
            # Restore
            os.environ.update(saved)

    def test_from_env_multiple_vars(self, monkeypatch):
        """Multiple env vars can be set simultaneously."""
        monkeypatch.setenv("MNEMOSYNE_LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("MNEMOSYNE_LLM_MODEL", "claude-3-haiku")
        monkeypatch.setenv("MNEMOSYNE_EXTRACTION_VERSION", "1.0.0")

        s = Settings.from_env()

        assert s.llm_provider == "anthropic"
        assert s.llm_model == "claude-3-haiku"
        assert s.extraction_version == "1.0.0"

    def test_from_env_router_threshold(self, monkeypatch):
        """MNEMOSYNE_ROUTER_UNSTRUCTURED_THRESHOLD should be read and coerced to float."""
        monkeypatch.setenv("MNEMOSYNE_ROUTER_UNSTRUCTURED_THRESHOLD", "0.85")

        s = Settings.from_env()

        assert s.router_unstructured_threshold == pytest.approx(0.85)

    def test_from_env_embedding_provider(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_EMBEDDING_PROVIDER", "openai_compatible")
        s = Settings.from_env()
        assert s.embedding_provider == "openai_compatible"

    def test_from_env_embedding_model(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_EMBEDDING_MODEL", "text-embedding-3-small")
        s = Settings.from_env()
        assert s.embedding_model == "text-embedding-3-small"

    def test_from_env_embedding_base_url(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_EMBEDDING_BASE_URL", "https://api.openai.com")
        s = Settings.from_env()
        assert s.embedding_base_url == "https://api.openai.com"

    def test_from_env_embedding_api_key(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_EMBEDDING_API_KEY", "sk-test-123")
        s = Settings.from_env()
        assert s.embedding_api_key == "sk-test-123"

    def test_from_env_llm_api_key(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_LLM_API_KEY", "sk-llm-key")
        s = Settings.from_env()
        assert s.llm_api_key == "sk-llm-key"


class TestConfigProperties:
    def test_llm_config_defaults(self):
        s = Settings()
        cfg = s.llm_config
        assert cfg["provider"] == "ollama"
        assert cfg["model"] == "gemma3:4b"
        assert cfg["base_url"] == "http://localhost:11434/v1"
        assert cfg["api_key"] is None

    def test_llm_config_reflects_fields(self):
        s = Settings(llm_provider="openai_compatible", llm_model="gpt-4o", llm_api_key="sk-x")
        cfg = s.llm_config
        assert cfg["provider"] == "openai_compatible"
        assert cfg["model"] == "gpt-4o"
        assert cfg["api_key"] == "sk-x"

    def test_embedding_config_defaults(self):
        s = Settings()
        cfg = s.embedding_config
        assert cfg["provider"] == "ollama"
        assert cfg["model"] == "nomic-embed-text"
        assert cfg["base_url"] == "http://localhost:11434"
        assert cfg["api_key"] is None
        assert cfg["dimensions"] == 768

    def test_embedding_config_reflects_fields(self):
        s = Settings(
            embedding_provider="openai_compatible",
            embedding_model="text-embedding-3-small",
            embedding_base_url="https://api.openai.com",
            embedding_api_key="sk-emb",
            embedding_dim=1536,
        )
        cfg = s.embedding_config
        assert cfg["provider"] == "openai_compatible"
        assert cfg["model"] == "text-embedding-3-small"
        assert cfg["base_url"] == "https://api.openai.com"
        assert cfg["api_key"] == "sk-emb"
        assert cfg["dimensions"] == 1536

    def test_embedding_config_from_env(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_EMBEDDING_PROVIDER", "openai_compatible")
        monkeypatch.setenv("MNEMOSYNE_EMBEDDING_BASE_URL", "https://api.together.xyz")
        monkeypatch.setenv("MNEMOSYNE_EMBEDDING_MODEL", "togethercomputer/m2-bert")
        monkeypatch.setenv("MNEMOSYNE_EMBEDDING_API_KEY", "tog-key")
        s = Settings.from_env()
        cfg = s.embedding_config
        assert cfg["provider"] == "openai_compatible"
        assert cfg["base_url"] == "https://api.together.xyz"
        assert cfg["model"] == "togethercomputer/m2-bert"
        assert cfg["api_key"] == "tog-key"
