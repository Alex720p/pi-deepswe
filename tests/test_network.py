import json

from pier.models.agent.network import NetworkAllowlist
from pier.models.task.task import Task
from pier.models.trial.paths import TrialPaths

from pi_deepswe.environment import PiDockerEnvironment

from .test_docker import FIXTURE


def make_environment(tmp_path, *, verifier=False, url="https://models.example:8443/v1"):
    task = Task(FIXTURE)
    return PiDockerEnvironment(
        environment_dir=FIXTURE / ("tests" if verifier else "environment"),
        environment_name="fixture",
        session_id="fixture",
        trial_paths=TrialPaths(tmp_path),
        task_env_config=task.config.verifier.environment if verifier else task.config.environment,
        network_allowlist=None if verifier else NetworkAllowlist(domains=["models.example"]),
        model_base_url=url,
    )


def test_custom_https_port_and_exact_host_policy(tmp_path):
    env = make_environment(tmp_path)
    env._prepare_egress_proxy_compose()
    script = (tmp_path / "egress-proxy/start-squid.sh").read_text()
    assert "acl Safe_ports port 8443\n" in script
    assert "acl SSL_ports port 8443\n" in script
    compose = json.loads((tmp_path / "docker-compose-egress-proxy.json").read_text())
    proxy = compose["services"]["pier-egress-proxy"]
    assert proxy["environment"]["ALLOWLIST_DOMAINS"] == "models.example"
    assert proxy["extra_hosts"] == {"host.docker.internal": "host-gateway"}
    assert compose["networks"]["pier-egress-internal"]["internal"] is True


def test_verifier_gets_no_inference_proxy(tmp_path):
    env = make_environment(tmp_path, verifier=True)
    env._prepare_egress_proxy_compose()
    assert env._egress_proxy_compose_path is None
    assert not (tmp_path / "egress-proxy").exists()
    assert env.task_env_config.allow_internet is False
