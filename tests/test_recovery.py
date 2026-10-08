import asyncio
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pi_deepswe.agent import InferenceUnavailableError, PiAgent
from pi_deepswe.config import ModelConfig
from pi_deepswe.recovery import archive_unfinished_trials, job_lock, run_job


@pytest.mark.asyncio
async def test_watch_stops_hung_inference_after_consecutive_failures(tmp_path, monkeypatch):
    import pi_deepswe.agent as agent_module

    monkeypatch.setattr(agent_module, "VPN_PROBE_INTERVAL", 0.001)
    model = ModelConfig(base_url="http://localhost:8000/v1", model_id="mock/model")
    agent = PiAgent(logs_dir=tmp_path, model_name=model.identity, model_config=model.model_dump())
    cancelled = asyncio.Event()
    probes = []

    async def execute(*, command, **_kwargs):
        if command.startswith("set -euo pipefail; if"):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        probes.append(command)
        return SimpleNamespace(return_code=1)

    environment = SimpleNamespace(exec=execute, agent_process_env=lambda env: env)
    with pytest.raises(InferenceUnavailableError, match="three consecutive"):
        await agent.execute_with_connection_watch(environment)
    assert len(probes) == 3
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_watch_tolerates_transient_disconnect_and_resets_failures(tmp_path, monkeypatch):
    import pi_deepswe.agent as agent_module

    monkeypatch.setattr(agent_module, "VPN_PROBE_INTERVAL", 0.001)
    model = ModelConfig(base_url="http://localhost:8000/v1", model_id="mock/model")
    agent = PiAgent(logs_dir=tmp_path, model_name=model.identity, model_config=model.model_dump())
    finished = asyncio.Event()
    responses = iter([1, 1, 0, 1, 1, 0])
    probes = []

    async def execute(*, command, **_kwargs):
        if command.startswith("set -euo pipefail; if"):
            await finished.wait()
            return SimpleNamespace(return_code=0)
        status = next(responses)
        probes.append(status)
        if len(probes) == 6:
            finished.set()
        return SimpleNamespace(return_code=status)

    environment = SimpleNamespace(exec=execute, agent_process_env=lambda env: env)
    result = await agent.execute_with_connection_watch(environment)
    assert result.return_code == 0
    assert probes == [1, 1, 0, 1, 1, 0]


def make_trial(job_dir, name, *, error=None, reward=0, missing_result=False):
    trial = job_dir / name
    trial.mkdir()
    (trial / "config.json").write_text("{}")
    (trial / "partial.txt").write_text("saved partial work")
    if not missing_result:
        (trial / "result.json").write_text(
            json.dumps({"exception_info": {"exception_type": error} if error else None})
        )
        (trial / "verifier").mkdir()
        (trial / "verifier/reward.json").write_text(json.dumps({"reward": reward}))
    return trial


def test_resume_preserves_zero_scores_and_graded_timeouts_archives_interrupted(tmp_path):
    completed = make_trial(tmp_path, "completed", reward=0)
    timed_out = make_trial(tmp_path, "timeout", error="AgentTimeoutError", reward=0)
    interrupted = make_trial(tmp_path, "interrupted", error="InferenceUnavailableError")
    incomplete = make_trial(tmp_path, "incomplete", missing_result=True)
    no_grade = make_trial(tmp_path, "no-grade")
    (no_grade / "verifier/reward.json").unlink()
    archived = archive_unfinished_trials(tmp_path)
    assert len(archived) == 3
    assert completed.exists() and timed_out.exists()
    assert not interrupted.exists() and not incomplete.exists() and not no_grade.exists()
    assert all((path / "partial.txt").read_text() == "saved partial work" for path in archived)
    assert archive_unfinished_trials(tmp_path) == []


def test_job_lock_prevents_concurrent_resume(tmp_path):
    with job_lock(tmp_path):
        with pytest.raises(ValueError, match="already running"):
            with job_lock(tmp_path):
                pytest.fail("The second launcher must not acquire the same job")
    with job_lock(tmp_path):
        pass


@pytest.mark.asyncio
async def test_run_pauses_after_saved_result_and_keeps_cleanup(tmp_path, monkeypatch):
    import pi_deepswe.recovery as recovery

    callbacks = []
    cleanup = []

    async def fake_cleanup(event):
        assert (tmp_path / "trial/result.json").exists()
        cleanup.append(event.trial_id)

    async def run():
        event = SimpleNamespace(
            trial_id="trial",
            result=SimpleNamespace(
                exception_info=SimpleNamespace(exception_type="InferenceUnavailableError")
            ),
        )
        make_trial(tmp_path, "trial", error="InferenceUnavailableError")

        async def end():
            for callback in callbacks:
                await callback(event)

        async with asyncio.TaskGroup() as group:
            group.create_task(end())

    monkeypatch.setattr(recovery, "prune_trial_docker_cache", fake_cleanup)
    job = SimpleNamespace(
        job_dir=tmp_path, add_hook=Mock(), on_trial_ended=callbacks.append, run=run
    )
    assert await run_job(job, prune=True) == 75
    assert cleanup == ["trial"]
    assert json.loads((tmp_path / "run-state.json").read_text())["status"] == "paused"
