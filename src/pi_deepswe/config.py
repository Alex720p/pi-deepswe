"""User configuration and pi's custom-provider configuration."""

import ipaddress
import re
import tomllib
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Prices(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input: float = Field(ge=0)
    output: float = Field(ge=0)
    cacheRead: float = Field(default=0, ge=0)
    cacheWrite: float = Field(default=0, ge=0)


def endpoint_url(value: str) -> str:
    """Normalize host loopback for the inference proxy, never the task container."""
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("base_url must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("base_url cannot contain credentials, query parameters, or fragments")
    host = parsed.hostname
    if not re.fullmatch(r"[A-Za-z0-9.-]+", host):
        raise ValueError("Use an IPv4 address or DNS hostname for base_url")
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host.lower() == "localhost"
    # Accessing parsed.port also validates the port range.
    port = parsed.port
    if loopback:
        host = "host.docker.internal"
    netloc = host + (f":{port}" if port is not None else "")
    return urlunsplit((parsed.scheme, netloc, parsed.path.rstrip("/"), "", ""))


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    base_url: str
    model_id: str = Field(min_length=1)
    api_key_env: str | None = None
    context_window: int = Field(default=32768, ge=4096)
    max_tokens: int = Field(default=4096, ge=1)
    reasoning: bool = False
    thinking: Literal["off", "minimal", "low", "medium", "high", "xhigh", "max"] = "off"
    sampling: dict[str, Any] = Field(default_factory=dict)
    compat: dict[str, Any] = Field(default_factory=dict)
    prices: Prices | None = None

    @field_validator("base_url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        endpoint_url(value)
        return value.rstrip("/")

    @field_validator("model_id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if value != value.strip() or any(ord(c) < 32 for c in value):
            raise ValueError("model_id cannot contain surrounding whitespace or control characters")
        return value

    @field_validator("api_key_env")
    @classmethod
    def validate_key(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            raise ValueError("api_key_env must be an environment variable name, not a secret")
        return value

    @model_validator(mode="after")
    def validate_limits(self) -> "ModelConfig":
        if self.max_tokens >= self.context_window:
            raise ValueError("max_tokens must be smaller than context_window")
        if self.thinking != "off" and not self.reasoning:
            raise ValueError("Set reasoning=true before selecting a thinking level")
        return self

    @property
    def effective_url(self) -> str:
        return endpoint_url(self.base_url)

    @property
    def identity(self) -> str:
        return f"benchmark/{self.model_id}"

    def pi_models(self) -> dict[str, Any]:
        model: dict[str, Any] = {
            "id": self.model_id,
            "name": self.model_id,
            "reasoning": self.reasoning,
            "input": ["text"],
            "contextWindow": self.context_window,
            "maxTokens": self.max_tokens,
        }
        if self.sampling:
            model["samplingParams"] = self.sampling
        if self.compat:
            model["compat"] = self.compat
        if self.prices is not None:
            model["cost"] = self.prices.model_dump()
        return {
            "providers": {
                "benchmark": {
                    "baseUrl": self.effective_url,
                    "api": "openai-completions",
                    "apiKey": "${PI_DEEPSWE_API_KEY}" if self.api_key_env else "unused",
                    "models": [model],
                }
            }
        }


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tasks_dir: Path = Path("datasets/deep-swe/tasks")
    jobs_dir: Path = Path("jobs")
    concurrency: int = Field(default=1, ge=1)


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: ModelConfig
    run: RunConfig = Field(default_factory=RunConfig)

    @classmethod
    def load(cls, path: Path) -> "Config":
        with path.open("rb") as stream:
            return cls.model_validate(tomllib.load(stream))
