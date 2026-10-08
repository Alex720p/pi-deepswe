import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError

from pi_deepswe.cli import build_job_config, check_prerequisites, select_tasks
from pi_deepswe.config import Config, ModelConfig, RunConfig, endpoint_url


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


@pytest.fixture
def prerequisite_mocks(monkeypatch):
    import pi_deepswe.cli as cli

    monkeypatch.setattr(cli.PiDockerEnvironment, "preflight", lambda: None)
    monkeypatch.setattr(cli.subprocess, "run", lambda *_args, **_kwargs: None)

    def fail_resource_probe(*_args, **_kwargs):
        pytest.fail("Host memory and disk space must not be inspected")

    original_read_text = Path.read_text

    def read_text(path, *args, **kwargs):
        if path == Path("/proc/meminfo"):
            fail_resource_probe()
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(shutil, "disk_usage", fail_resource_probe)
    monkeypatch.setattr(cli.subprocess, "check_output", fail_resource_probe)


def test_fixture_prerequisites_without_host_resource_probes(prerequisite_mocks):
    from .test_docker import FIXTURE

    config = Config(model=ModelConfig(base_url="http://localhost:8000/v1", model_id="test"))
    assert check_prerequisites(config, [FIXTURE]) is None


def test_prerequisites_still_require_api_key(prerequisite_mocks, monkeypatch):
    from .test_docker import FIXTURE

    monkeypatch.delenv("TEST_MODEL_API_KEY", raising=False)
    config = Config(
        model=ModelConfig(
            base_url="http://localhost:8000/v1", model_id="test", api_key_env="TEST_MODEL_API_KEY"
        )
    )
    with pytest.raises(ValueError, match="Set TEST_MODEL_API_KEY"):
        check_prerequisites(config, [FIXTURE])


def test_check_output_omits_host_resources(prerequisite_mocks, monkeypatch, capsys):
    import pi_deepswe.cli as cli

    from .test_docker import FIXTURE

    config = Config(
        model=ModelConfig(base_url="http://localhost:8000/v1", model_id="test"),
        run=RunConfig(tasks_dir=FIXTURE.parent),
    )
    monkeypatch.setattr(cli.Config, "load", lambda _path: config)
    monkeypatch.setattr(
        cli.sys, "argv", ["pi-deepswe", "check", "--config", "unused.toml", "--task", FIXTURE.name]
    )
    cli.main()
    output = capsys.readouterr().out
    values, _ = json.JSONDecoder().raw_decode(output)
    assert "resources" not in values
    assert values["tasks"] == [FIXTURE.name]
    assert "Configuration/prerequisite checks passed." in output


@pytest.mark.asyncio
async def test_provenance_omits_resources_preserves_pruning(tmp_path, monkeypatch):
    from pier.job import Job

    import pi_deepswe.cli as cli

    from .test_docker import FIXTURE

    job = SimpleNamespace(job_dir=tmp_path, run=AsyncMock(), on_trial_ended=Mock(), add_hook=Mock())
    create_job = AsyncMock(return_value=job)
    monkeypatch.setattr(Job, "create", create_job)
    monkeypatch.setattr(cli, "dataset_revision", lambda _path: "fixture-revision")
    monkeypatch.setattr(
        cli, "summarize", lambda _path: {"infrastructure_errors": 0, "verified_trials": 1}
    )
    config = Config(
        model=ModelConfig(base_url="http://localhost:8000/v1", model_id="test"),
        run=RunConfig(prune_docker_cache=True),
    )
    assert await cli.launch(config, [FIXTURE]) == 0
    provenance = json.loads((tmp_path / "pi-deepswe-provenance.json").read_text())
    assert "resources" not in provenance
    assert provenance["prune_docker_cache"] is True
    assert provenance["budgets"] == "task defaults"
    job.on_trial_ended.assert_called_once()
    job.run.assert_awaited_once()
