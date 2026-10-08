import json
import os
import subprocess
from pathlib import Path

import pytest
from pier.job import Job
from pier.models.trial.result import TrialResult

from pi_deepswe import cleanup
from pi_deepswe.cli import build_job_config
from pi_deepswe.config import Config, ModelConfig, RunConfig

from .mock_server import mock_model

pytestmark = pytest.mark.docker
FIXTURE = Path(__file__).parent / "fixtures/task"


def require_docker():
    if os.environ.get("PI_DEEPSWE_DOCKER_TESTS") != "1":
        pytest.skip("Set PI_DEEPSWE_DOCKER_TESTS=1 to run Docker integration tests")


@pytest.mark.asyncio
async def test_full_pipeline(tmp_path, monkeypatch):
    require_docker()
    # Exercise real trial-image removal while preserving the host builder's
    # unrelated cache during tests. The prune command itself is unit-tested.
    real_docker_command = cleanup.docker_command
    prune_calls = []

    async def test_docker_command(*args):
        if args[:2] == ("builder", "prune"):
            prune_calls.append(args)
            return 0, "builder prune mocked in integration test"
        return await real_docker_command(*args)

    monkeypatch.setattr(cleanup, "docker_command", test_docker_command)
    monkeypatch.setenv("MOCK_API_KEY", "fixture-secret-not-for-provenance")
    with mock_model() as (port, requests):
        config = Config(
            model=ModelConfig(
                base_url=f"http://localhost:{port}/v1",
                model_id="mock/model",
                api_key_env="MOCK_API_KEY",
            ),
            run=RunConfig(jobs_dir=tmp_path / "jobs", prune_docker_cache=True),
        )
        job = await Job.create(build_job_config(config, [FIXTURE], "pi-docker-smoke"))
        hook_trials = []

        async def after_grading(event):
            trial = event.config.trials_dir / event.config.trial_name
            assert (trial / "result.json").exists()
            assert (trial / "verifier/reward.json").exists()
            assert (trial / "artifacts/model.patch").exists()
            await cleanup.prune_trial_docker_cache(event)
            hook_trials.append(event.trial_id)

        job.on_trial_ended(after_grading)
        await job.run()
        results = list(job.job_dir.glob("*/result.json"))
        assert len(results) == 1
        result = TrialResult.model_validate_json(results[0].read_text())
        assert result.exception_info is None
        trial = results[0].parent
        rewards = json.loads((trial / "verifier/reward.json").read_text())
        assert rewards["reward"] == 1
        patch = (trial / "artifacts/model.patch").read_text()
        assert "solved" in patch and "new.txt" in patch
        trajectory = json.loads((trial / "agent/trajectory.json").read_text())
        assert len([s for s in trajectory["steps"] if s["source"] == "agent"]) == 4
        assert any("isolation-ok" in str(s.get("observation")) for s in trajectory["steps"])
        assert result.agent_result.n_input_tokens == 44
        assert result.agent_result.n_output_tokens == 12
        assert result.agent_result.cost_usd is None
        assert len(requests) == 4
        assert all(
            r["authorization"] == "Bearer fixture-secret-not-for-provenance" for r in requests
        )
        tools = requests[0]["body"]["tools"]
        assert {t["function"]["name"] for t in tools} == {"read", "bash", "edit", "write"}
        for name in ["config.json", "agent/provenance.json"]:
            assert "fixture-secret-not-for-provenance" not in (trial / name).read_text()
        label = trial.name.lower().replace("_", "-")
        running = subprocess.check_output(["docker", "ps", "--format", "{{.Names}}"], text=True)
        assert label not in running
        assert hook_trials == [trial.name]
        assert prune_calls == [("builder", "prune", "--all", "--force")]
        report = json.loads((trial / "docker-cleanup.json").read_text())
        assert report["build_cache"]["exit_code"] == 0
        assert "error" not in report
        for image in json.loads((trial / "docker-images.json").read_text()):
            assert (
                subprocess.run(
                    ["docker", "image", "inspect", image], capture_output=True
                ).returncode
                != 0
            )


@pytest.mark.asyncio
async def test_timeout_preserves_partial_patch(tmp_path):
    require_docker()
    with mock_model(hang_after_write=True) as (port, requests):
        config = Config(
            model=ModelConfig(base_url=f"http://localhost:{port}/v1", model_id="mock/model"),
            run=RunConfig(jobs_dir=tmp_path / "jobs"),
        )
        job_config = build_job_config(config, [FIXTURE], "pi-docker-timeout")
        job_config.agents[0].override_timeout_sec = 5
        job = await Job.create(job_config)
        await job.run()
        result_path = next(job.job_dir.glob("*/result.json"))
        result = json.loads(result_path.read_text())
        assert result["exception_info"]["exception_type"] == "AgentTimeoutError"
        trial = result_path.parent
        patch = (trial / "artifacts/model.patch").read_text()
        assert "solved" in patch
        assert (trial / "agent/events.jsonl").stat().st_size > 0
        assert (trial / "agent/trajectory.json").exists()
        assert (trial / "verifier/reward.json").exists()
        assert len(requests) >= 2
