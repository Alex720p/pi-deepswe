import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from pi_deepswe.cli import build_job_config, check_resources, select_tasks
from pi_deepswe.config import Config, ModelConfig, endpoint_url


@pytest.mark.parametrize(
    "url,expected",
    [
        ("http://localhost:8000/v1/", "http://host.docker.internal:8000/v1"),
        ("http://127.0.0.1:11434/v1", "http://host.docker.internal:11434/v1"),
        ("https://models.example.org:8443/v1", "https://models.example.org:8443/v1"),
    ],
)
def test_endpoint_routing(url, expected):
    assert endpoint_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "ftp://localhost/model",
        "http://user:secret@localhost/v1",
        "http://localhost/v1?api_key=secret",
        "http://localhost:70000/v1",
    ],
)
def test_bad_endpoint(url):
    with pytest.raises(ValueError):
        endpoint_url(url)


def test_custom_model_config_preserves_id_and_references_key():
    model = ModelConfig(
        base_url="http://localhost:8000/v1",
        model_id="org/model-name",
        api_key_env="MODEL_API_KEY",
        sampling={"temperature": 0.6},
    )
    provider = model.pi_models()["providers"]["benchmark"]
    assert provider["models"][0]["id"] == "org/model-name"
    assert provider["models"][0]["samplingParams"] == {"temperature": 0.6}
    assert provider["apiKey"] == "${PI_DEEPSWE_API_KEY}"
    config = Config(model=model)
    job = build_job_config(config, [Path("fixture")], "test")
    assert job.agents[0].env == {"MODEL_API_KEY": "${MODEL_API_KEY}"}
    assert "secret" not in job.model_dump_json()
    assert job.n_attempts == 1 and job.retry.max_retries == 0


def test_limits_and_environment_name_validation():
    with pytest.raises(ValidationError):
        ModelConfig(
            base_url="http://localhost:8000/v1", model_id="model", api_key_env="a secret value"
        )
    with pytest.raises(ValidationError):
        ModelConfig(
            base_url="http://localhost:8000/v1",
            model_id="model",
            context_window=4096,
            max_tokens=4096,
        )
    with pytest.raises(ValidationError):
        ModelConfig(base_url="http://localhost:8000/v1", model_id="model", thinking="high")


def test_subset_is_deterministic_and_traversal_rejected(tmp_path):
    for name in ["z", "a", "b", "c"]:
        (tmp_path / name).mkdir()
        (tmp_path / name / "task.toml").touch()
    first = select_tasks(tmp_path, task=None, n_tasks=2, seed=0)
    second = select_tasks(tmp_path, task=None, n_tasks=2, seed=0)
    assert first == second and len(first) == 2
    with pytest.raises(ValueError):
        select_tasks(tmp_path, task="../outside", n_tasks=None, seed=0)


def test_fixture_resource_check(monkeypatch):
    import pi_deepswe.cli as cli

    from .test_docker import FIXTURE

    monkeypatch.setattr(cli.PiDockerEnvironment, "preflight", lambda: None)
    monkeypatch.setattr(cli.subprocess, "run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli.subprocess, "check_output", lambda *_args, **_kwargs: "/tmp")
    config = Config(model=ModelConfig(base_url="http://localhost:8000/v1", model_id="test"))
    result = check_resources(config, [FIXTURE])
    assert result["required_memory_mb"] == 2048
    assert json.dumps(result)
