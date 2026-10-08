import subprocess

from pi_deepswe.agent import COMMIT_SCRIPT, PiAgent
from pi_deepswe.config import ModelConfig


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def test_commit_snapshot_includes_tracked_untracked_and_preserves_existing_commits(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "test")
    git(tmp_path, "config", "user.email", "test@localhost")
    (tmp_path / "tracked.txt").write_text("before\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "initial")
    before = git(tmp_path, "rev-parse", "HEAD")
    (tmp_path / "tracked.txt").write_text("after\n")
    (tmp_path / "untracked.txt").write_text("new\n")
    subprocess.run(["bash", "-c", COMMIT_SCRIPT], cwd=tmp_path, check=True, capture_output=True)
    patch = git(tmp_path, "diff", before, "HEAD")
    assert "+after" in patch and "untracked.txt" in patch
    after = git(tmp_path, "rev-parse", "HEAD")
    subprocess.run(["bash", "-c", COMMIT_SCRIPT], cwd=tmp_path, check=True, capture_output=True)
    assert git(tmp_path, "rev-parse", "HEAD") == after
    assert not git(tmp_path, "status", "--porcelain")


def test_model_id_shell_quoting_and_install_line_size(tmp_path):
    model = ModelConfig(base_url="http://localhost:8000/v1", model_id="org/a'; $(touch injected)")
    agent = PiAgent(logs_dir=tmp_path, model_name=model.identity, model_config=model.model_dump())
    command = agent.command()
    # The provider's exact model ID remains one quoted CLI argument.
    assert "--model 'org/a'" in command
    spec = agent.install_spec()
    from pier.environments.agent_setup import dockerfile_install_commands

    for line in dockerfile_install_commands(spec, user="root"):
        assert len(line.encode()) < 65535
    assert spec.fingerprint() == agent.install_spec().fingerprint()
