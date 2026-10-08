"""A pinned Pier Docker environment with exact inference host/port egress."""

import json
from urllib.parse import urlsplit

from pier.environments.agent_setup import EGRESS_PROXY_SERVICE
from pier.environments.docker.docker import DockerEnvironment

from pi_deepswe.config import endpoint_url


class PiDockerEnvironment(DockerEnvironment):
    def __init__(self, *args, model_base_url: str, **kwargs):
        self.model_base_url = endpoint_url(model_base_url)
        super().__init__(*args, **kwargs)

    def _prepare_egress_proxy_compose(self) -> None:
        # Verifiers are created without an agent install or network allowlist.
        # Preserve Pier's no-network setup for those environments.
        if not self.network_allowlist.domains:
            return
        if self.task_env_config.allow_internet:
            raise ValueError("pi-deepswe requires a task with inference-only/no-network egress")
        endpoint = urlsplit(self.model_base_url)
        if self.network_allowlist.domains != [endpoint.hostname]:
            raise ValueError("The network allowlist must contain only the configured model host")
        super()._prepare_egress_proxy_compose()
        path = self._egress_proxy_compose_path
        if path is None:
            raise RuntimeError("Pier did not generate an inference proxy")
        port = endpoint.port or (443 if endpoint.scheme == "https" else 80)
        script_path = self.trial_paths.trial_dir / "egress-proxy/start-squid.sh"
        script = script_path.read_text()
        # Fail on upstream drift instead of silently retaining a broader policy.
        for old in ("acl SSL_ports port 443", "acl Safe_ports port 80 443"):
            if script.count(old) != 1:
                raise RuntimeError(
                    "Pinned Pier proxy template changed; review networking integration"
                )
        script = script.replace("acl SSL_ports port 443", f"acl SSL_ports port {port}")
        script = script.replace("acl Safe_ports port 80 443", f"acl Safe_ports port {port}")
        script_path.write_text(script)
        compose = json.loads(path.read_text())
        compose["services"][EGRESS_PROXY_SERVICE]["extra_hosts"] = {
            "host.docker.internal": "host-gateway"
        }
        path.write_text(json.dumps(compose, indent=2) + "\n")
