"""Docker lifecycle and health check utilities for AIC competition CLI."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]


HEALTH_URL = "http://localhost:8000/api/v1/health"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def check_api_health(timeout: float = 3.0) -> dict[str, Any] | None:
    """Check if the API server is up and healthy."""
    if requests is None:
        return None
    try:
        resp = requests.get(HEALTH_URL, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def ensure_docker_runtime(
    *,
    max_wait_seconds: int = 120,
    force_restart: bool = False,
) -> bool:
    """Ensure Docker runtime is running and healthy; auto-starts if down."""
    health = check_api_health()
    if health and not force_restart:
        print("[OK] API server is already running and healthy.")
        print(f"     Status: {health}")
        return True

    print("\n[!] API server is not running or restart was requested.")
    print("[*] Launching Docker runtime via scripts/docker_up_test_data.bat ...\n")

    bat_script = PROJECT_ROOT / "scripts" / "docker_up_test_data.bat"
    if not bat_script.exists():
        print(f"[ERROR] Script not found: {bat_script}")
        return False

    try:
        # Run docker_up_test_data.bat
        process = subprocess.Popen(
            [str(bat_script)],
            cwd=str(PROJECT_ROOT),
            shell=True,
        )
        returncode = process.wait()
        if returncode != 0:
            print(f"[ERROR] Docker startup script exited with code {returncode}")
            return False
    except Exception as e:
        print(f"[ERROR] Failed to run Docker startup script: {e}")
        return False

    # Poll health until ready
    print("\n[*] Waiting for API to report healthy...")
    start_time = time.time()
    while time.time() - start_time < max_wait_seconds:
        health = check_api_health(timeout=2.0)
        if health and health.get("status") == "ok":
            print(f"[OK] API is healthy! (Took {time.time() - start_time:.1f}s)")
            print(f"     Status: {health}\n")
            return True
        time.sleep(2)
        print(".", end="", flush=True)

    print(f"\n[ERROR] Timed out waiting for API after {max_wait_seconds}s.")
    return False
