import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from pi_deepswe import cleanup
from pi_deepswe.config import RunConfig
from pi_deepswe.environment import DockerEnvironment

from .test_network import make_environment


def test_pruning_requires_serial_trials():
    assert RunConfig().prune_docker_cache is False
    with pytest.raises(ValidationError, match="requires concurrency=1"):
        RunConfig(concurrency=2, prune_docker_cache=True)


@pytest.mark.asyncio
async def test_images_recorded_before_failed_build_and_verifier_merged(tmp_path, monkeypatch):
    async def fail_start(*_args, **_kwargs):
        raise RuntimeError("build failed")

    monkeypatch.setattr(DockerEnvironment, "start", fail_start)
    agent = make_environment(tmp_path)
    agent.prune_docker_cache = True
    with pytest.raises(RuntimeError, match="build failed"):
        await agent.start(force_build=False)
    verifier = make_environment(tmp_path, verifier=True)
    verifier.session_id = "fixture__verifier"
    verifier.prune_docker_cache = True
    with pytest.raises(RuntimeError, match="build failed"):
        await verifier.start(force_build=False)
    images = json.loads((tmp_path / "docker-images.json").read_text())
    assert images == [
        "fixture-main:latest",
        "fixture-pier-egress-proxy:latest",
        "fixture__verifier-main:latest",
    ]


@pytest.mark.asyncio
async def test_prebuilt_image_recording(tmp_path, monkeypatch):
    async def fake_start(*_args, **_kwargs):
        pass

    monkeypatch.setattr(DockerEnvironment, "start", fake_start)
    env = make_environment(tmp_path, verifier=True)
    env.prune_docker_cache = True
    env.task_env_config.docker_image = "registry.example/task:pinned"
    await env.start(force_build=False)
    assert json.loads((tmp_path / "docker-images.json").read_text()) == [
        "registry.example/task:pinned"
    ]
    await env.start(force_build=True)
    assert json.loads((tmp_path / "docker-images.json").read_text()) == [
        "fixture-main:latest",
        "registry.example/task:pinned",
    ]


def make_event(tmp_path, images):
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    (trial_dir / "docker-images.json").write_text(json.dumps(images))
    (trial_dir / "result.json").write_text('{"reward": 1}')
    return SimpleNamespace(
        trial_id="trial", config=SimpleNamespace(trials_dir=tmp_path, trial_name="trial")
    )


@pytest.mark.asyncio
async def test_prune_skips_referenced_and_absent_images_preserves_result(tmp_path, monkeypatch):
    event = make_event(tmp_path, ["in-use", "gone", "trial-image"])
    calls = []

    async def fake_docker(*args):
        calls.append(args)
        if args[0] == "ps":
            return 0, "existing-container\n" if "ancestor=in-use" in args else ""
        if args[:2] == ("image", "inspect"):
            return (1, "No such image") if args[2] == "gone" else (0, "sha256:example")
        return 0, "reclaimed"

    monkeypatch.setattr(cleanup, "docker_command", fake_docker)
    await cleanup.prune_trial_docker_cache(event)
    removals = [c for c in calls if c[:2] == ("image", "rm")]
    assert removals == [("image", "rm", "trial-image")]
    assert calls[-1] == ("builder", "prune", "--all", "--force")
    assert (tmp_path / "trial/result.json").read_text() == '{"reward": 1}'
    report = json.loads((tmp_path / "trial/docker-cleanup.json").read_text())
    assert report["images"][0]["status"] == "in-use-or-check-failed"
    assert report["images"][1]["status"] == "already-absent"
    assert report["build_cache"]["exit_code"] == 0


@pytest.mark.asyncio
async def test_cleanup_failure_does_not_replace_benchmark_result(tmp_path, monkeypatch):
    event = make_event(tmp_path, [])

    async def fail_docker(*_args):
        raise OSError("Docker unavailable")

    monkeypatch.setattr(cleanup, "docker_command", fail_docker)
    await cleanup.prune_trial_docker_cache(event)
    assert (tmp_path / "trial/result.json").read_text() == '{"reward": 1}'
    report = json.loads((tmp_path / "trial/docker-cleanup.json").read_text())
    assert report["error"] == "Docker unavailable"
