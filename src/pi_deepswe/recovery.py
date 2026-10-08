"""Stop on lost inference access and resume unfinished benchmark tasks."""

import fcntl
import json
import shlex
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pier.trial.hooks import TrialEvent

from pi_deepswe.cleanup import prune_trial_docker_cache


class StopForDisconnect(Exception):
    pass


@contextmanager
def job_lock(job_dir: Path):
    with (job_dir / ".pi-deepswe-run.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"A benchmark process is already running in {job_dir}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


async def run_job(job, *, prune: bool) -> int:
    stopping = False

    async def guard_start(_event):
        if stopping:
            raise StopForDisconnect("Inference access was lost")

    async def ended(event):
        nonlocal stopping
        info = event.result.exception_info if event.result is not None else None
        disconnected = info is not None and info.exception_type == "InferenceUnavailableError"
        stopping = stopping or disconnected
        if prune:
            await prune_trial_docker_cache(event)
        if disconnected:
            raise StopForDisconnect("Inference access was lost")

    job.add_hook(TrialEvent.START, guard_start)
    job.on_trial_ended(ended)
    state_path = job.job_dir / "run-state.json"
    state = {"status": "running", "started_at": datetime.now(UTC).isoformat()}
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    disconnected = False
    try:
        try:
            await job.run()
        except* StopForDisconnect:
            disconnected = True
    finally:
        state["status"] = "paused" if stopping else "stopped"
        state["updated_at"] = datetime.now(UTC).isoformat()
        state_path.write_text(json.dumps(state, indent=2) + "\n")
    if disconnected:
        print(
            "Inference access was lost. Results and partial work have been saved.\n"
            "Reconnect the VPN, export the model API key, then run:\n"
            f"  .venv/bin/pi-deepswe resume {shlex.quote(str(job.job_dir))}",
            file=sys.stderr,
        )
        return 75
    state["status"] = "finished"
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    return 0


def archive_unfinished_trials(job_dir: Path) -> list[Path]:
    """Keep every attempt's logs while removing unfinished trials from Pier's scan."""
    archived = []
    for trial in sorted(job_dir.iterdir()):
        if not trial.is_dir() or not (trial / "config.json").is_file():
            continue
        result_path = trial / "result.json"
        retry = not result_path.exists()
        if result_path.exists():
            result = json.loads(result_path.read_text())
            error = result.get("exception_info")
            # An agent exhausting its official time budget is a scored attempt,
            # provided verification completed. Infrastructure failures are retried.
            retry = (
                error is not None and error.get("exception_type") != "AgentTimeoutError"
            ) or not (trial / "verifier/reward.json").exists()
        if retry:
            archive = job_dir / ".interrupted"
            archive.mkdir(exist_ok=True)
            destination = archive / f"{trial.name}-{uuid4().hex[:8]}"
            trial.rename(destination)
            archived.append(destination)
    return archived
