"""Small launcher that delegates task execution and scoring to Pier."""

import argparse
import asyncio
import importlib.metadata
import json
import os
import random
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pier.models.job.config import JobConfig, RetryConfig
from pier.models.task.task import Task
from pier.models.task.verifier_mode import resolve_effective_verifier_env_config
from pier.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig

from pi_deepswe import __version__
from pi_deepswe.config import Config, ModelConfig, RunConfig
from pi_deepswe.environment import PiDockerEnvironment
from pi_deepswe.pins import DEEPSWE_COMMIT, NODE_VERSION, PI_VERSION, PIER_VERSION
from pi_deepswe.recovery import archive_unfinished_trials, job_lock, run_job


def fetch_dataset(destination: Path) -> None:
    if destination.exists():
        raise ValueError(f"{destination} already exists; refusing to replace it")
    destination.mkdir(parents=True)
    subprocess.run(["git", "init", "--quiet", str(destination)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(destination),
            "fetch",
            "--depth",
            "1",
            "https://github.com/datacurve-ai/deep-swe.git",
            DEEPSWE_COMMIT,
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(destination),
            "checkout",
            "--quiet",
            "--detach",
            "FETCH_HEAD",
        ],
        check=True,
    )
    print(f"DeepSWE {DEEPSWE_COMMIT} checked out at {destination}")


def select_tasks(root: Path, *, task: str | None, n_tasks: int | None, seed: int) -> list[Path]:
    if not root.is_dir():
        raise ValueError(f"Tasks directory {root} is missing; run pi-deepswe fetch first")
    if task:
        path = (root / task).resolve()
        if not path.is_relative_to(root.resolve()) or not (path / "task.toml").is_file():
            raise ValueError(f"Unknown task: {task}")
        return [path]
    paths = sorted(p.resolve() for p in root.iterdir() if (p / "task.toml").is_file())
    if not paths:
        raise ValueError(f"No tasks found in {root}")
    if n_tasks is not None:
        if n_tasks < 1 or n_tasks > len(paths):
            raise ValueError(f"n-tasks must be between 1 and {len(paths)}")
        random.Random(seed).shuffle(paths)
        paths = paths[:n_tasks]
    return paths


def check_prerequisites(config: Config, task_paths: list[Path]) -> None:
    PiDockerEnvironment.preflight()
    subprocess.run(["docker", "compose", "version"], check=True, capture_output=True)
    if importlib.metadata.version("datacurve-pier") != PIER_VERSION:
        raise ValueError(f"This adapter requires datacurve-pier=={PIER_VERSION}")
    if config.model.api_key_env and not os.environ.get(config.model.api_key_env):
        raise ValueError(
            f"Set {config.model.api_key_env}, or omit api_key_env for an unauthenticated server"
        )
    tasks = [Task(path) for path in task_paths]
    for task in tasks:
        if task.config.environment.os.value != "linux" or task.has_steps:
            raise ValueError("This adapter currently supports single-step Linux tasks")
        if task.config.environment.allow_internet:
            raise ValueError(
                f"Task {task.name} permits internet; inference-only tasks are required"
            )
        verifier_env = resolve_effective_verifier_env_config(task.config, None)
        if verifier_env is None or verifier_env.allow_internet:
            raise ValueError(f"Task {task.name} must use a separate no-network verifier")


def build_job_config(config: Config, paths: list[Path], job_name: str) -> JobConfig:
    model = config.model
    agent_env = {model.api_key_env: "${" + model.api_key_env + "}"} if model.api_key_env else {}
    return JobConfig(
        job_name=job_name,
        jobs_dir=config.run.jobs_dir.resolve(),
        n_attempts=1,
        n_concurrent_trials=config.run.concurrency,
        retry=RetryConfig(max_retries=0),
        agents=[
            AgentConfig(
                import_path="pi_deepswe.agent:PiAgent",
                model_name=model.identity,
                kwargs={"model_config": model.model_dump(mode="json")},
                env=agent_env,
            )
        ],
        environment=EnvironmentConfig(
            import_path="pi_deepswe.environment:PiDockerEnvironment",
            kwargs={
                "model_base_url": model.effective_url,
                "prune_docker_cache": config.run.prune_docker_cache,
            },
        ),
        tasks=[TaskConfig(path=p) for p in paths],
    )


def dataset_revision(tasks_dir: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(tasks_dir), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def summarize(job_dir: Path) -> dict:
    trials = []
    # Pier keeps one result.json in each immediate trial directory.
    for path in sorted(job_dir.glob("*/result.json")):
        result = json.loads(path.read_text())
        reward_path = path.parent / "verifier/reward.json"
        rewards = json.loads(reward_path.read_text()) if reward_path.exists() else None
        exception = result.get("exception_info")
        trials.append(
            {
                "task": result.get("task_name"),
                "rewards": rewards,
                "exception": exception,
                "path": str(path.parent),
                "usage": result.get("agent_result"),
            }
        )
    return {
        "job_dir": str(job_dir),
        "trials": trials,
        "verified_trials": sum(t["rewards"] is not None for t in trials),
        "infrastructure_errors": sum(t["exception"] is not None for t in trials),
    }


async def launch(config: Config, paths: list[Path]) -> int:
    # Import lazily: --help/config inspection do not load optional agent providers.
    from pier.job import Job

    job_name = datetime.now(UTC).strftime("pi-%Y%m%d-%H%M%S-") + uuid4().hex[:6]
    job = await Job.create(build_job_config(config, paths, job_name))
    revision = dataset_revision(config.run.tasks_dir)
    (job.job_dir / "pi-deepswe-provenance.json").write_text(
        json.dumps(
            {
                "adapter_version": __version__,
                "pi_version": PI_VERSION,
                "pier_version": PIER_VERSION,
                "node_version": NODE_VERSION,
                "deepswe_commit": revision,
                "pinned_deepswe_commit": DEEPSWE_COMMIT,
                "model": config.model.model_dump(mode="json"),
                "effective_base_url": config.model.effective_url,
                "tasks": [p.name for p in paths],
                "attempts": 1,
                "concurrency": config.run.concurrency,
                "prune_docker_cache": config.run.prune_docker_cache,
                "whole_trial_retries": 0,
                "budgets": "task defaults",
                "label": "pi + model on DeepSWE",
            },
            indent=2,
        )
        + "\n"
    )
    with job_lock(job.job_dir):
        status = await run_job(job, prune=config.run.prune_docker_cache)
    summary = summarize(job.job_dir)
    print(json.dumps(summary, indent=2))
    return status or (
        1 if summary["infrastructure_errors"] or not summary["verified_trials"] else 0
    )


async def resume(job_dir: Path) -> int:
    from pier.job import Job

    job_dir = job_dir.resolve()
    saved = JobConfig.model_validate_json((job_dir / "config.json").read_text())
    if (saved.jobs_dir / saved.job_name).resolve() != job_dir:
        raise ValueError("Resume from the original job directory; saved paths must still exist")
    if len(saved.agents) != 1 or saved.agents[0].import_path != "pi_deepswe.agent:PiAgent":
        raise ValueError("Only jobs created by pi-deepswe can be resumed")
    if saved.environment.import_path != "pi_deepswe.environment:PiDockerEnvironment":
        raise ValueError("The saved job does not use the pi-deepswe environment")
    model = ModelConfig.model_validate(saved.agents[0].kwargs["model_config"])
    pruning = saved.environment.kwargs.get("prune_docker_cache", False)
    config = Config(
        model=model,
        run=RunConfig(concurrency=saved.n_concurrent_trials, prune_docker_cache=pruning),
    )
    paths = [task.get_local_path() for task in saved.tasks]
    check_prerequisites(config, paths)
    with job_lock(job_dir):
        archived = archive_unfinished_trials(job_dir)
        print(f"Resuming {job_dir}; archived {len(archived)} unfinished attempts.")
        job = await Job.create(saved)
        status = await run_job(job, prune=pruning)
    summary = summarize(job_dir)
    print(json.dumps(summary, indent=2))
    return status or (
        1 if summary["infrastructure_errors"] or not summary["verified_trials"] else 0
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Stock pi on DeepSWE via Pier")
    commands = result.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="Download the pinned official DeepSWE checkout")
    fetch.add_argument("--destination", type=Path, default=Path("datasets/deep-swe"))
    for name in ("run", "check"):
        command = commands.add_parser(
            name, help="Run tasks" if name == "run" else "Check configuration and prerequisites"
        )
        command.add_argument("--config", type=Path, required=True)
        selection = command.add_mutually_exclusive_group()
        selection.add_argument("--task", help="Task folder name (default: abs-module-cache-flags)")
        selection.add_argument("--n-tasks", type=int, help="Deterministic random subset")
        selection.add_argument("--all", action="store_true", help="Run the full corpus")
        command.add_argument("--seed", type=int, default=0)
        command.add_argument(
            "--prune-docker-cache",
            action="store_true",
            help="After each trial remove its unused images and prune builder-wide unused cache; "
            "requires concurrency=1",
        )
    summary = commands.add_parser("summary", help="Inspect saved verifier rewards and errors")
    summary.add_argument("job_dir", type=Path)
    restart = commands.add_parser(
        "resume", help="Resume unfinished tasks using the saved job settings"
    )
    restart.add_argument("job_dir", type=Path)
    return result


def main() -> None:
    args = parser().parse_args()
    try:
        if args.command == "fetch":
            fetch_dataset(args.destination)
            return
        if args.command == "summary":
            print(json.dumps(summarize(args.job_dir), indent=2))
            return
        if args.command == "resume":
            sys.exit(asyncio.run(resume(args.job_dir)))
        config = Config.load(args.config)
        if args.prune_docker_cache:
            config.run = RunConfig.model_validate(
                {**config.run.model_dump(), "prune_docker_cache": True}
            )
        task = args.task
        if task is None and args.n_tasks is None and not args.all:
            task = "abs-module-cache-flags"
        paths = select_tasks(config.run.tasks_dir, task=task, n_tasks=args.n_tasks, seed=args.seed)
        check_prerequisites(config, paths)
        print(
            json.dumps(
                {
                    "tasks": [p.name for p in paths],
                    "model": config.model.identity,
                    "endpoint": config.model.effective_url,
                    "prune_docker_cache": config.run.prune_docker_cache,
                },
                indent=2,
            )
        )
        if args.command == "check":
            print(
                "Configuration/prerequisite checks passed. "
                "Endpoint reachability is checked inside each trial."
            )
        else:
            sys.exit(asyncio.run(launch(config, paths)))
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"pi-deepswe: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
