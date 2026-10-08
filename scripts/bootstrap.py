#!/usr/bin/env python3
"""Install pinned uv/Python in this workspace and synchronize the locked project."""

import hashlib
import io
import os
import platform
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

UV_VERSION = "0.12.23"
WHEELS = {
    "x86_64": (
        "https://files.pythonhosted.org/packages/e5/83/85a6c63c24905af4924fddb11a499b934913f59a134248367a1ef1a4716f/uv-0.12.23-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "565c6e2874dbeae86c02f3dea97255e878fec672659a73d4930c6b93fcab2fff",
    ),
    "aarch64": (
        "https://files.pythonhosted.org/packages/39/77/ff15f878a215b543fd5aa05d2e3f939f3a2c46502ff58362eb5b51434af3/uv-0.12.23-py3-none-manylinux_2_17_aarch64.manylinux2014_aarch64.musllinux_1_1_aarch64.whl",
        "895137194d242cc8075c3006288b485af54feeae68512f4ffc96f75b27543cc0",
    ),
}


def main():
    root = Path(__file__).resolve().parents[1]
    if platform.system() != "Linux" or platform.machine() not in WHEELS:
        sys.exit("Bootstrap supports Linux x86_64/aarch64; install uv 0.12.23 manually elsewhere.")
    uv = root / ".tools/bin/uv"
    if not uv.exists():
        url, expected = WHEELS[platform.machine()]
        print(f"Downloading uv {UV_VERSION}", flush=True)
        with urllib.request.urlopen(url, timeout=60) as response:
            raw = response.read()
        if hashlib.sha256(raw).hexdigest() != expected:
            sys.exit("uv wheel checksum mismatch")
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            name = next(n for n in archive.namelist() if n.endswith("/scripts/uv"))
            uv.parent.mkdir(parents=True, exist_ok=True)
            uv.write_bytes(archive.read(name))
            uv.chmod(0o755)
    actual = subprocess.check_output([str(uv), "--version"], text=True)
    if actual.split()[1] != UV_VERSION:
        sys.exit(f"Expected uv {UV_VERSION}, found {actual.strip()}")
    env = {
        **os.environ,
        "UV_PYTHON_INSTALL_DIR": str(root / ".tools/python"),
        "UV_CACHE_DIR": str(root / ".cache/uv"),
    }
    subprocess.run(
        [str(uv), "python", "install", "--no-bin", "3.12.15"], cwd=root, env=env, check=True
    )
    subprocess.run([str(uv), "sync", "--locked"], cwd=root, env=env, check=True)
    print("Ready. Run .venv/bin/pi-deepswe --help")


if __name__ == "__main__":
    main()
