"""Configurações e descoberta do Ollama."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

DEFAULT_PORT = 11434


def _is_wsl() -> bool:
    version_file = Path("/proc/version")
    if not version_file.exists():
        return False
    try:
        content = version_file.read_text().lower()
        return "microsoft" in content or "wsl" in content
    except OSError:
        return False


def _wsl_windows_ip() -> str | None:
    try:
        result = subprocess.run(
            ["ip", "route", "show", "default"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    parts = result.stdout.split()
    if "via" in parts:
        return parts[parts.index("via") + 1]
    return None


def get_ollama_host() -> str:
    """Retorna a URL base do Ollama."""
    env_host = os.getenv("OLLAMA_HOST")
    if env_host:
        return env_host.rstrip("/")

    if _is_wsl():
        ip = _wsl_windows_ip()
        if ip:
            return f"http://{ip}:{DEFAULT_PORT}"

    return f"http://localhost:{DEFAULT_PORT}"
