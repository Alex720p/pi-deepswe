"""Stock pi installed-agent adapter for Pier 0.3.1."""

import asyncio
import base64
import gzip
import json
import shlex
import tempfile
from importlib.resources import files
from typing import Any
from urllib.parse import urlsplit

from pier.agents.installed.base import BaseInstalledAgent, NonZeroAgentExitCodeError
from pier.environments.base import BaseEnvironment
from pier.models.agent.context import AgentContext
from pier.models.agent.install import AgentInstallSpec, InstallStep
from pier.models.agent.network import NetworkAllowlist

from pi_deepswe.config import ModelConfig
from pi_deepswe.pins import NODE_VERSION, PI_VERSION, PIER_VERSION
from pi_deepswe.trajectory import convert_events, read_events

RUNTIME = "/opt/pi-deepswe"
NODE = f"{RUNTIME}/node/bin/node"
CLI = f"{RUNTIME}/runtime/node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js"
LOGS = "/logs/agent"
CONFIG_DIR = "/installed-agent/pi-deepswe"
VPN_PROBE_INTERVAL = 30
VPN_PROBE_FAILURES = 3


class InferenceUnavailableError(NonZeroAgentExitCodeError):
    """Inference access failed; preserve this attempt and stop the job."""


# A separate process group lets timeout cleanup stop pi and its shell-tool children
# before taking the commit snapshot. Never kill PID 1 or an unvalidated PID.
STOP_SCRIPT = r"""
if [ -f /logs/agent/pi.pid ]; then
    pi_task_pid=$(cat /logs/agent/pi.pid)
    if [[ "$pi_task_pid" =~ ^[0-9]+$ ]] && [ "$pi_task_pid" -gt 1 ]; then
        kill -TERM -- "-$pi_task_pid" 2>/dev/null || true
        for pi_task_wait in 1 2 3 4 5; do
            kill -0 -- "-$pi_task_pid" 2>/dev/null || break
            sleep 1
        done
        kill -KILL -- "-$pi_task_pid" 2>/dev/null || true
    fi
fi
"""
COMMIT_SCRIPT = r"""
set -euo pipefail
git -c safe.directory='*' add -A
if ! git -c safe.directory='*' diff --cached --quiet; then
    git -c safe.directory='*' -c user.name='pi-deepswe' \
        -c user.email='pi-deepswe@localhost' -c core.hooksPath=/dev/null \
        -c commit.gpgSign=false commit -m 'DeepSWE agent submission'
fi
git -c safe.directory='*' rev-parse HEAD
"""


def endpoint_probe_command(base_url: str, headers_path: str) -> str:
    # Some authenticated gateways return 403 for /models even when inference
    # works. Squid adds X-Squid-Error to its own denials; distinguish those.
    return (
        "set -euo pipefail; "
        f"pi_http_status=$(curl --silent --show-error --output /dev/null --max-time 20 "
        f"--dump-header {shlex.quote(headers_path)} "
        f"--noproxy '' --proxy \"${{http_proxy:-}}\" --write-out '%{{http_code}}' "
        f"{shlex.quote(base_url + '/models')}); "
        f"if grep -qi '^X-Squid-Error:' {shlex.quote(headers_path)}; then "
        'echo "Inference proxy could not reach the configured model endpoint." >&2; exit 1; fi; '
        'case "$pi_http_status" in 000|407|502|503|504) '
        'echo "Model endpoint unavailable ($pi_http_status). '
        'For host-local servers, bind a Docker-reachable interface." >&2; exit 1 ;; esac'
    )


def install_steps() -> list[InstallStep]:
    assets = files("pi_deepswe").joinpath("runtime")
    checksums = json.loads(assets.joinpath("node-checksums.json").read_text())
    script = [
        "set -euo pipefail",
        "apt-get update",
        "DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "
        "curl ca-certificates git xz-utils util-linux gzip",
        "rm -rf /var/lib/apt/lists/*",
        f"mkdir -p {RUNTIME}/node {RUNTIME}/runtime",
        'case "$(uname -m)" in x86_64) pi_node_arch=x64 ;; '
        "aarch64) pi_node_arch=arm64 ;; "
        '*) echo "Unsupported container architecture" >&2; exit 1 ;; esac',
        f'pi_node_archive="node-v{NODE_VERSION}-linux-$pi_node_arch.tar.xz"',
        f'curl --fail --location --retry 3 "https://nodejs.org/dist/v{NODE_VERSION}/'
        '$pi_node_archive" --output /tmp/pi-node.tar.xz',
        'case "$pi_node_arch" in '
        f"x64) pi_node_sha={checksums[f'node-v{NODE_VERSION}-linux-x64.tar.xz']} ;; "
        f"arm64) pi_node_sha={checksums[f'node-v{NODE_VERSION}-linux-arm64.tar.xz']} ;; esac",
        'printf "%s  /tmp/pi-node.tar.xz\\n" "$pi_node_sha" | sha256sum --check -',
        f"tar -xJf /tmp/pi-node.tar.xz --strip-components=1 -C {RUNTIME}/node",
        "rm /tmp/pi-node.tar.xz",
    ]
    for filename in ("package.json", "package-lock.json"):
        # Pier emits each install step on one Dockerfile line. Compress locks to
        # stay below Docker's 64 KiB line limit, with a deterministic gzip header.
        encoded = base64.b64encode(
            gzip.compress(assets.joinpath(filename).read_bytes(), mtime=0)
        ).decode()
        script.append(
            f"printf %s {shlex.quote(encoded)} | base64 -d | gzip -d > {RUNTIME}/runtime/{filename}"
        )
    script.extend(
        [
            f"export PATH={RUNTIME}/node/bin:$PATH",
            f"npm ci --prefix {RUNTIME}/runtime --ignore-scripts --no-audit --no-fund",
            f"{NODE} {CLI} --version",
        ]
    )
    return [InstallStep(user="root", run="\n".join(script))]


class PiAgent(BaseInstalledAgent):
    SUPPORTS_ATIF = True

    def __init__(self, *args, model_config: dict[str, Any], version: str = PI_VERSION, **kwargs):
        if version != PI_VERSION:
            raise ValueError(f"This runtime lock requires pi {PI_VERSION}")
        self.model_config = ModelConfig.model_validate(model_config)
        super().__init__(*args, version=version, **kwargs)
        if self.model_name != self.model_config.identity:
            raise ValueError(f"model_name must be {self.model_config.identity!r}")
        self.repo_dir: str | None = None

    @staticmethod
    def name() -> str:
        return "pi"

    def get_version_command(self) -> str:
        return f"{NODE} {CLI} --version"

    def install_spec(self) -> AgentInstallSpec:
        return AgentInstallSpec(
            agent_name=self.name(),
            version=PI_VERSION,
            steps=install_steps(),
            verification_command=self.get_version_command(),
            metadata={"node_version": NODE_VERSION, "pi_version": PI_VERSION},
        )

    def network_allowlist(self) -> NetworkAllowlist:
        return NetworkAllowlist(domains=[urlsplit(self.model_config.effective_url).hostname])

    def runtime_env(self) -> dict[str, str]:
        env = {
            "PI_CODING_AGENT_DIR": CONFIG_DIR,
            "PI_OFFLINE": "1",
            "PI_SKIP_VERSION_CHECK": "1",
            "PI_TELEMETRY": "0",
            "PATH": (
                f"{RUNTIME}/node/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
            ),
        }
        if self.model_config.api_key_env:
            key = self._get_env(self.model_config.api_key_env)
            if not key:
                raise ValueError(f"Set {self.model_config.api_key_env} before running")
        return env

    async def setup(self, environment: BaseEnvironment) -> None:
        await super().setup(environment)
        result = await environment.exec(
            command="git -c safe.directory='*' rev-parse --show-toplevel",
            timeout_sec=15,
        )
        if result.return_code != 0 or not (result.stdout or "").strip():
            raise RuntimeError("Task working directory must be inside its Git repository")
        self.repo_dir = result.stdout.strip()
        result = await environment.exec(
            command=f"mkdir -p {CONFIG_DIR} {LOGS}/sessions; chmod -R a+rwX {CONFIG_DIR} {LOGS}",
            user="root",
            timeout_sec=15,
        )
        if result.return_code != 0:
            raise RuntimeError("Could not prepare pi configuration and logs")
        # Docker's default -e KEY=value arguments can expose secrets in process
        # listings. Transfer a short-lived 0600 file instead; the launch shell
        # reads it into pi's environment without embedding its value in argv.
        if self.model_config.api_key_env:
            key = self._get_env(self.model_config.api_key_env)
            if not key:
                raise ValueError(f"Set {self.model_config.api_key_env} before running")
            identity = await environment.exec(command="id -u", timeout_sec=15)
            if identity.return_code != 0 or not (identity.stdout or "").strip().isdigit():
                raise RuntimeError("Could not identify the task user for credential transfer")
            uid = int(identity.stdout.strip())
            credential_path = f"{CONFIG_DIR}/credential"
            with tempfile.NamedTemporaryFile(mode="w", prefix="pi-deepswe-credential-") as stream:
                stream.write(key)
                stream.flush()
                await environment.upload_file(stream.name, credential_path)
            permissions = await environment.exec(
                command=f"chown {uid} {credential_path} && chmod 600 {credential_path}",
                user="root",
                timeout_sec=15,
            )
            if permissions.return_code != 0:
                raise RuntimeError("Could not restrict the transient credential file")
        models = json.dumps(self.model_config.pi_models(), ensure_ascii=False)
        result = await environment.exec(
            command=f"printf %s {shlex.quote(models)} > {CONFIG_DIR}/models.json",
            timeout_sec=15,
        )
        if result.return_code != 0:
            raise RuntimeError("Could not write pi's model configuration")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "provenance.json").write_text(
            json.dumps(
                {
                    "pi_version": PI_VERSION,
                    "node_version": NODE_VERSION,
                    "pier_version": PIER_VERSION,
                    "model": self.model_config.model_dump(mode="json"),
                    "effective_base_url": self.model_config.effective_url,
                    "repo_dir": self.repo_dir,
                    "configuration": "stock pi, no skills/extensions/MCP/prompt templates",
                    "submission": "bounded commit finalization after pi exits or times out",
                },
                indent=2,
            )
            + "\n"
        )
        # Check through precisely the same proxy used by pi. /models is not
        # universally implemented, so any HTTP response other than proxy denial
        # proves connectivity; inference errors remain visible in the trial.
        env = environment.agent_process_env(self.runtime_env()) or {}
        probe = endpoint_probe_command(
            self.model_config.effective_url, f"{CONFIG_DIR}/endpoint-headers"
        )
        # Do not log the environment dictionary, which contains provider secrets.
        result = await environment.exec(command=probe, env=env, timeout_sec=25)
        if result.return_code != 0:
            raise InferenceUnavailableError(
                f"Endpoint preflight failed: {result.stderr or result.stdout}"
            )

    async def execute_with_connection_watch(self, environment: BaseEnvironment):
        env = environment.agent_process_env(self.runtime_env())
        process = asyncio.create_task(
            environment.exec(command=self.command(), cwd=self.repo_dir, env=env)
        )

        async def watch():
            failures = 0
            while True:
                await asyncio.sleep(VPN_PROBE_INTERVAL)
                probe = endpoint_probe_command(
                    self.model_config.effective_url, f"{CONFIG_DIR}/watch-headers"
                )
                try:
                    result = await environment.exec(command=probe, env=env, timeout_sec=25)
                    unreachable = result.return_code != 0
                except Exception:
                    unreachable = True
                failures = failures + 1 if unreachable else 0
                if failures >= VPN_PROBE_FAILURES:
                    raise InferenceUnavailableError(
                        "Inference endpoint became unreachable in three consecutive probes; "
                        "reconnect the VPN and resume the job."
                    )

        watcher = asyncio.create_task(watch())
        try:
            done, _ = await asyncio.wait({process, watcher}, return_when=asyncio.FIRST_COMPLETED)
            if process in done:
                return process.result()
            watcher.result()
        finally:
            for task in (process, watcher):
                if not task.done():
                    task.cancel()
            await asyncio.gather(process, watcher, return_exceptions=True)

    def command(self) -> str:
        args = [
            NODE,
            CLI,
            "--mode",
            "json",
            "--provider",
            "benchmark",
            "--model",
            self.model_config.model_id,
            "--thinking",
            self.model_config.thinking,
            "--no-extensions",
            "--no-skills",
            "--no-mcp",
            "--no-prompt-templates",
            "--no-approve",
            "--offline",
            "--session-dir",
            f"{LOGS}/sessions",
        ]
        return (
            "set -euo pipefail; "
            f"if [ -f {CONFIG_DIR}/credential ]; then "
            f'export PI_DEEPSWE_API_KEY="$(cat {CONFIG_DIR}/credential)"; fi; '
            f"setsid {shlex.join(args)} < {LOGS}/instruction.md "
            f"> {LOGS}/events.jsonl 2> {LOGS}/stderr.txt & "
            "pi_task_pid=$!; "
            f"printf '%s\\n' \"$pi_task_pid\" > {LOGS}/pi.pid; "
            'wait "$pi_task_pid"'
        )

    async def finalize(self, environment: BaseEnvironment) -> None:
        result = await environment.exec(
            command=(
                STOP_SCRIPT
                + f"\nrm -f {CONFIG_DIR}/credential\n{{\n"
                + COMMIT_SCRIPT
                + f"\n}} > {LOGS}/submission.txt 2>&1"
            ),
            cwd=self.repo_dir,
            timeout_sec=30,
        )
        if result.return_code != 0:
            raise RuntimeError("Could not commit the submission; see agent/submission.txt")

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        result = await environment.exec(
            command=f"printf %s {shlex.quote(instruction)} > {LOGS}/instruction.md",
            timeout_sec=15,
        )
        if result.return_code != 0:
            raise RuntimeError("Could not write the task instruction")
        error: BaseException | None = None
        try:
            result = await self.execute_with_connection_watch(environment)
            events, _ = read_events(self.logs_dir / "events.jsonl")
            failures = [
                e
                for e in events
                if e.get("type") == "message_end"
                and e.get("message", {}).get("role") == "assistant"
                and e.get("message", {}).get("stopReason") in {"error", "aborted"}
            ]
            # A recovered API error is retained in the trajectory, not a trial failure.
            assistants = [
                e["message"]
                for e in events
                if e.get("type") == "message_end"
                and e.get("message", {}).get("role") == "assistant"
            ]
            if failures and assistants[-1].get("stopReason") in {"error", "aborted"}:
                raise InferenceUnavailableError(
                    "pi ended with a model error; see agent/events.jsonl"
                )
            if result.return_code != 0:
                raise NonZeroAgentExitCodeError(
                    f"pi exited with status {result.return_code}; see agent/stderr.txt"
                )
            if not any(e.get("type") == "agent_settled" for e in events):
                raise NonZeroAgentExitCodeError("pi did not emit agent_settled; see agent logs")
        except BaseException as exc:
            error = exc
            raise
        finally:
            task = asyncio.create_task(self.finalize(environment))
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                await asyncio.shield(task)
                raise
            except Exception as exc:
                (self.logs_dir / "finalization-error.txt").write_text(str(exc) + "\n")
                if error is None:
                    raise
            self.populate_context_post_run(context)

    def populate_context_post_run(self, context: AgentContext) -> None:
        events, malformed = read_events(self.logs_dir / "events.jsonl")
        trajectory = convert_events(
            events,
            version=PI_VERSION,
            model_name=self.model_config.identity,
            priced=self.model_config.prices is not None,
            malformed_lines=malformed,
        )
        if trajectory is None:
            return
        (self.logs_dir / "trajectory.json").write_text(trajectory.model_dump_json(indent=2) + "\n")
        metrics = trajectory.final_metrics
        assert metrics is not None
        context.n_input_tokens = metrics.total_prompt_tokens
        context.n_output_tokens = metrics.total_completion_tokens
        context.n_cache_tokens = metrics.total_cached_tokens
        context.cost_usd = metrics.total_cost_usd
        context.n_agent_steps = sum(s.source == "agent" for s in trajectory.steps)
        context.peak_context_tokens = metrics.extra["peak_context_tokens"]
        context.summarization_count = metrics.extra["summarization_count"]
        context.metadata = {
            "malformed_event_lines": malformed,
            "usage_complete": metrics.extra["usage_complete"],
            "cost_source": trajectory.extra["cost_source"],
        }
