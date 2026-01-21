"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class LLMConfig(BaseModel):
    """LLM configuration."""

    provider: str = "vllm"
    model: str = "mistralai/Mistral-Small-Instruct-2409"
    base_url: str = "http://localhost:8000/v1"
    max_tokens: int = 4096
    temperature: float = 0.7


class AgentConfig(BaseModel):
    """Agent configuration."""

    max_loops: int = 10
    temperature: float = 0.7
    timeout_seconds: float = 30.0
    enable_checkpointing: bool = True
    checkpoint_interval: int = 1


class AttackConfig(BaseModel):
    """Attack configuration."""

    enabled: bool = False
    type: str = "none"
    scheduler: str = "random"
    probability: float = 0.3


class DefenseConfig(BaseModel):
    """Defense configuration."""

    tool_verification: dict[str, Any] = Field(default_factory=dict)
    redundancy: dict[str, Any] = Field(default_factory=dict)
    rollback: dict[str, Any] = Field(default_factory=dict)
    self_consistency: dict[str, Any] = Field(default_factory=dict)


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""

    metrics: list[str] = Field(default_factory=lambda: ["task_success_rate", "safety_score"])
    failure_propagation: dict[str, Any] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    """Complete experiment configuration."""

    experiment: dict[str, Any] = Field(default_factory=dict)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    attacks: AttackConfig = Field(default_factory=AttackConfig)
    defenses: DefenseConfig = Field(default_factory=DefenseConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


class Settings(BaseSettings):
    """Environment-based settings."""

    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model: str = "mistralai/Mistral-Small-Instruct-2409"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    database_url: str = "sqlite+aiosqlite:///experiments/experiments.db"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def load_config(config_path: str | Path) -> ExperimentConfig:
    """Load experiment configuration from YAML file."""
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    return ExperimentConfig.model_validate(data)


def get_settings() -> Settings:
    """Get environment-based settings."""
    return Settings()
