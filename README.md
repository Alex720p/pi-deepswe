# pi-deepswe

Run stock [pi](https://github.com/earendil-works/pi) on the official
[DeepSWE](https://github.com/datacurve-ai/deep-swe) tasks, using
[Pier](https://github.com/datacurve-ai/pier) for Docker execution and grading.

Each attempt runs pi inside the task container. The adapter commits pi's changes,
Pier collects the resulting patch, stops the agent container, and grades it in a
fresh container without network access. Held-out tests and reference solutions
are not mounted into the agent container.

## Setup

Requires Linux, Python 3.10+ for bootstrap, Git, and a running Docker daemon with
Docker Compose. Bootstrap installs uv 0.12.23 and Python 3.12.15 inside this
workspace; the host's Python and Node installations are left alone.

```sh
python3 scripts/bootstrap.py
.venv/bin/pi-deepswe fetch
cp configs/config.example.toml config.local.toml
```

`fetch` checks out DeepSWE revision
`0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea` (113 tasks) in `datasets/deep-swe`.
It refuses to replace an existing directory. The checkout, jobs, caches, and
local configuration are ignored by Git.

Edit `config.local.toml` to provide your model's `base_url` and `model_id`. The
server must support streaming OpenAI-compatible Chat Completions with tool
calling. This project connects to an existing server; it does not provision one.
If authentication is required, set `api_key_env` to the name of an exported
variable. For example, `api_key_env = "MODEL_API_KEY"`; keep its value out of TOML.
Omit this setting for a server that accepts an unused dummy key.

Match `context_window` and `max_tokens` to your server's actual limits. Defaults
are 32768 and 4096. For reasoning models, set `reasoning = true` and select the
supported `thinking` level. Models with `xhigh` or `max` levels also need an
explicit mapping, for example `[model.thinking_level_map]` with `max = "max"`.
Optional sampling, compatibility, and USD-per-million
prices are illustrated in the example configuration. Compatibility settings must
match your server. With no configured prices, reported dollar cost is unknown.

### Local model connectivity

Host `localhost` and IPv4 loopback addresses become `host.docker.internal` for
container inference. The proxy container maps that name to Docker's host gateway.
Your model server must listen on an interface reachable from Docker, such as the
host's Docker bridge address or `0.0.0.0`, rather than only `127.0.0.1`. Remote
IPv4 addresses and DNS hostnames also work; IPv6 endpoints are not supported in v1.

Only the model hostname and its configured TCP port are permitted by the
inference proxy. Custom ports such as 8000, 11434, and HTTPS 8443 are supported.
Do not use `--network host` or enable unrestricted task networking. The verifier
receives neither inference credentials nor an inference proxy.

## Run

Start with the default single task:

```sh
.venv/bin/pi-deepswe check --config config.local.toml
.venv/bin/pi-deepswe run --config config.local.toml --task abs-module-cache-flags
```

`check` validates configuration, Docker Compose, dependencies, and task
compatibility. Endpoint connectivity is checked through the actual inference proxy during trial setup;
`check` itself does not make an inference request.

All paths in TOML resolve from your current working directory. Each run gets a
new job directory, one attempt per task, no whole-trial retries, and the task's
original timeouts and CPU/memory limits. Concurrency defaults to one. The default
agent budget for `abs-module-cache-flags` is three hours, and the verifier budget
is 30 minutes. pi retains its normal API retry and compaction behavior.

For a deterministic subset or the whole corpus:

```sh
.venv/bin/pi-deepswe run --config config.local.toml --n-tasks 10 --seed 0
.venv/bin/pi-deepswe run --config config.local.toml --all
```

The subset shuffles sorted task-folder names using the provided seed. Record the
selected task IDs with any results; this selection need not match Pier's native
subset ordering. The benchmark's task limits are retained. Images/build caches
can accumulate over a full run; automatic cleanup is available below.

### VPN interruptions and resuming

The agent checks inference connectivity through its Docker proxy every 30 seconds.
Three consecutive failed probes stop the active agent, save its partial patch and
logs, and allow offline grading to finish. A failed endpoint preflight or a
terminal model API error also stops the job before further tasks run. Recoverable
API retries stay within pi. A VPN-related stop exits with code 75 and prints the
job path and resume command. For VPN-limited runs, keep `concurrency = 1`.

After reconnecting the VPN, export the model API key again and resume:

```sh
.venv/bin/pi-deepswe resume jobs/<job-name>
```

Resume uses the saved task selection, model configuration, budgets, and Docker
pruning setting. It retains completed scored attempts, including zero scores and
graded agent timeouts. Interrupted attempts and infrastructure failures restart
from scratch; their earlier logs and patches are moved into the job's `.interrupted/`
directory. Unstarted tasks run normally. Resumption continues the same job and
keeps its original provenance. Use the same workspace/task paths and do not run
two launchers against the same job. `run-state.json` records whether a run paused,
stopped, or finished. Ordinary SSH disconnects can be handled by running in `tmux`.

The stock configuration uses pi's default system prompt and four coding tools:
`read`, `bash`, `edit`, and `write`. Skills, extensions, MCP, prompt templates,
automatic updates, and trust-gated project configuration are disabled. Ordinary
repository context files are retained. No host pi configuration is mounted.
Changes are staged and committed during bounded finalization, including after a
timeout. Existing commits are preserved; finalization does not create an empty
commit. Ignored files are not added.

### Inspect results

```sh
.venv/bin/pi-deepswe summary jobs/<job-name>
.venv/bin/pier view jobs
```

Pier's native job and trial results remain the source of truth. Useful artifacts:

- `pi-deepswe-provenance.json`: versions, benchmark revision, model settings, task
  selection, run budgets, and Docker pruning settings.
- `<trial>/agent/`: `events.jsonl`, native sessions, `stderr.txt`, ATIF
  `trajectory.json`, `submission.txt`, and adapter provenance.
- `<trial>/artifacts/model.patch`: the submitted committed patch.
- `<trial>/verifier/`: `reward.json`, test reports, and verifier output.

A verifier-produced zero reward is a completed unsuccessful attempt; missing
verifier output is reported separately from scores. The launcher exits 0 for a
verified run without infrastructure errors, 1 for failed/incomplete execution,
2 for configuration or preflight errors, and 75 for an inference-related pause.
Timeouts preserve partial logs and patches, retain an infrastructure error, and
allow Pier to grade the partial work.

The trajectory converter counts authoritative completed messages once, connects
tool calls to their results, preserves reasoning, and includes reported
compaction usage. Missing usage remains unknown. Raw JSON and session logs remain
available when a process ends with a truncated event. These logs can contain
repository content and model output; review them before sharing.

Label measured results **pi + your model on DeepSWE**. They measure this agent
configuration and are not directly equivalent to the reference mini-swe-agent
leaderboard runs. The integration milestone is valid patch collection, saved
logs, and verifier output; the selected model need not solve the task.

## Development and verification

Host Python packages are locked in `uv.lock`. Container inference uses pi 1.1.0,
Node 22.19.0 with verified archive checksums, and `npm ci --ignore-scripts` against
the bundled npm lock. The adapter and Docker environment load through Pier's
custom import interfaces; no Pier fork is needed.

```sh
.venv/bin/ruff check src tests scripts
.venv/bin/ruff format --check src tests scripts
.venv/bin/pytest -q
PI_DEEPSWE_DOCKER_TESTS=1 .venv/bin/pytest tests/test_docker.py -q
```

Docker tests use a deterministic local mock SSE server and a small repository.
They make no external model calls. They verify tracked/untracked patch transfer,
model authentication, inference-only networking, held-out test isolation,
separate verifier grading, token accounting, timeout recovery, and cleanup.
Normal test runs skip these Docker tests. Builds download Ubuntu/system packages,
Node, and the locked npm packages.

## Cleanup

For a sequential full-suite run with automatic Docker cleanup:

```sh
.venv/bin/pi-deepswe run --config config.local.toml --all --prune-docker-cache
```

Alternatively set `prune_docker_cache = true` under `[run]`. This option requires
`concurrency = 1`. After each trial finishes grading and saves its results, cleanup
removes that trial's unused agent/proxy/verifier images and its task base-image
tag, then runs `docker builder prune --all --force`. It skips images referenced
by existing containers and never forces image removal. Other projects' image
tags and volumes are preserved, but unused build cache is pruned across the
selected Docker builder, so other builds may need to rebuild cached layers.
Only images recorded by the current trial are considered; older trial images
are not swept. Downloads/builds can therefore repeat for related tasks. Saved
results, patches, and logs remain under `jobs/`, including a per-trial
`docker-cleanup.json` report. Cleanup errors are logged without changing scores.
This limits accumulation between tasks, but a large individual task still needs
enough disk space to build and run. The default keeps caching enabled.

Pier stops trial containers and networks after completion, keeping logs and the
image layers needed for repeat runs. If a process is forcibly killed, use the
trial's generated Compose files and recorded project name to run `docker compose
... down`; inspect those files rather than deleting unrelated containers.

Remove selected job folders or this workspace's `.cache` when no run is active.
Inspect `docker system df` before removing unused images or build cache; avoid
broad Docker pruning when the machine also hosts other workloads.
