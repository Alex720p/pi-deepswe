import os
import shutil
import subprocess

import pytest

from pi_deepswe.agent import endpoint_probe_command

from .mock_server import mock_model


@pytest.mark.skipif(not shutil.which("curl"), reason="host curl is required for the probe test")
@pytest.mark.parametrize(
    "status,squid_error,expected",
    [(200, False, 0), (403, False, 0), (403, True, 1), (503, False, 1)],
)
def test_connectivity_distinguishes_gateway_auth_from_proxy_denial(
    tmp_path, status, squid_error, expected
):
    with mock_model(models_status=status, squid_error=squid_error) as (port, _requests):
        command = endpoint_probe_command(f"http://127.0.0.1:{port}/v1", str(tmp_path / "headers"))
        result = subprocess.run(
            ["bash", "-c", command], capture_output=True, env={**os.environ, "http_proxy": ""}
        )
        assert result.returncode == expected
