"""Reclaim completed trial images and unused builder cache after verification."""

import asyncio
import json
import logging

from pier.trial.hooks import TrialHookEvent

logger = logging.getLogger(__name__)


async def docker_command(*args: str) -> tuple[int, str]:
    process = await asyncio.create_subprocess_exec(
        "docker",
        *args,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        output, _ = await asyncio.wait_for(process.communicate(), timeout=300)
    except (TimeoutError, asyncio.CancelledError):
        if process.returncode is None:
            process.kill()
        await process.communicate()
        raise
    return process.returncode, output.decode(errors="replace")


async def prune_trial_docker_cache(event: TrialHookEvent) -> None:
    # END is awaited after verifier shutdown and result.json persistence, before
    # the queue releases the trial slot. Only serial runs enable this hook.
    trial_dir = event.config.trials_dir / event.config.trial_name
    report: dict = {"images": [], "build_cache": None}
    try:
        manifest = trial_dir / "docker-images.json"
        images = json.loads(manifest.read_text()) if manifest.exists() else []
        for image in images:
            code, output = await docker_command(
                "ps", "--all", "--filter", f"ancestor={image}", "--format", "{{.ID}}"
            )
            if code or output.strip():
                report["images"].append({"image": image, "status": "in-use-or-check-failed"})
                continue
            code, output = await docker_command("image", "inspect", image, "--format", "{{.Id}}")
            if code:
                report["images"].append({"image": image, "status": "already-absent"})
                continue
            # Never force deletion: Docker protects any remaining references.
            code, output = await docker_command("image", "rm", image)
            report["images"].append({"image": image, "exit_code": code, "output": output})
            if code:
                logger.warning("Docker image cleanup failed for %s: %s", image, output.strip())
        code, output = await docker_command("builder", "prune", "--all", "--force")
        report["build_cache"] = {"exit_code": code, "output": output}
        if code:
            logger.warning("Docker build-cache cleanup failed: %s", output.strip())
    except Exception as exc:
        # Cleanup failures must not replace an already saved benchmark outcome.
        report["error"] = str(exc)
        logger.warning("Docker cleanup failed for %s: %s", event.trial_id, exc)
    finally:
        try:
            (trial_dir / "docker-cleanup.json").write_text(json.dumps(report, indent=2) + "\n")
        except OSError as exc:
            logger.warning("Could not save Docker cleanup report for %s: %s", event.trial_id, exc)
