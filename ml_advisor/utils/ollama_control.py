"""Ciclo de vida do Ollama: verificação, start, stop e sessão.

Este módulo oferece:

- ``ping()`` — verifica se o Ollama está respondendo.
- ``OllamaSession`` — context manager que garante que o Ollama está
  no ar durante um bloco de código. Se ele já estiver rodando, deixa
  como está; se não, inicia e (opcionalmente) encerra ao final.
- Um CLI simples via ``python -m ml_advisor.utils.ollama_control``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import httpx

from ml_advisor.utils.config import get_ollama_host

_START_WAIT_SECONDS = 20
_PING_TIMEOUT = 3.0


# ---------------------------------------------------------------------------
# Operações de baixo nível
# ---------------------------------------------------------------------------


def ping(host: str | None = None, timeout: float = _PING_TIMEOUT) -> bool:
    """Retorna ``True`` se o Ollama responder em ``host``."""
    if host is None:
        host = get_ollama_host()

    try:
        response = httpx.get(f"{host}/api/tags", timeout=timeout, trust_env=False)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def _run_powershell(command: str, timeout: int = 10) -> bool:
    """Executa um comando PowerShell (do Windows) a partir do WSL2."""
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def start() -> bool:
    """Inicia o Ollama no Windows. Retorna ``True`` se bem-sucedido."""
    cmd = 'Start-Process "$env:LOCALAPPDATA\\Programs\\Ollama\\ollama app.exe"'
    return _run_powershell(cmd)


def stop() -> bool:
    """Encerra todos os processos do Ollama no Windows."""
    cmd = (
        "Get-Process | " 'Where-Object { $_.ProcessName -like "*ollama*" } | ' "Stop-Process -Force"
    )
    return _run_powershell(cmd)


def wait_until_ready(
    host: str | None = None,
    max_seconds: int = _START_WAIT_SECONDS,
) -> bool:
    """Aguarda o Ollama ficar pronto, checando a cada segundo."""
    if host is None:
        host = get_ollama_host()

    deadline = time.monotonic() + max_seconds
    while time.monotonic() < deadline:
        if ping(host):
            return True
        time.sleep(1)
    return False


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


class OllamaSession:
    """Context manager que garante o Ollama ativo durante um bloco.

    Uso típico::

        with OllamaSession(auto_close=True) as host:
            # o Ollama está garantidamente no ar aqui
            fazer_algo(host)

    Regra importante: só fecha no final se **ele** iniciou o serviço.
    Se o Ollama já estava rodando quando o bloco começou, ele é deixado
    intacto — evitando derrubar um serviço em uso por outra tarefa.
    """

    def __init__(
        self,
        host: str | None = None,
        auto_close: bool = False,
        wait_seconds: int = _START_WAIT_SECONDS,
        verbose: bool = True,
    ) -> None:
        self.host = host or get_ollama_host()
        self.auto_close = auto_close
        self.wait_seconds = wait_seconds
        self.verbose = verbose

        self._was_running_before = False
        self._we_started_it = False

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def __enter__(self) -> str:
        self._was_running_before = ping(self.host)

        if self._was_running_before:
            self._log(f"✓ Ollama já estava rodando em {self.host}")
            return self.host

        self._log("→ Ollama não respondeu. Tentando iniciar no Windows...")
        if not start():
            raise RuntimeError(
                "Não foi possível iniciar o Ollama. " "Abra manualmente no Windows e tente de novo."
            )

        self._we_started_it = True

        if not wait_until_ready(self.host, self.wait_seconds):
            raise RuntimeError(f"Ollama iniciou mas não respondeu em {self.wait_seconds}s.")

        self._log(f"✓ Ollama iniciado em {self.host}")
        return self.host

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if not self.auto_close:
            return
        if not self._we_started_it:
            self._log("→ Ollama já estava rodando antes; deixando como está.")
            return

        self._log("→ Encerrando Ollama (foi iniciado por esta sessão)...")
        if stop():
            self._log("✓ Ollama encerrado.")
        else:
            self._log("⚠ Falha ao encerrar o Ollama.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_status(_args: argparse.Namespace) -> int:
    host = get_ollama_host()
    if ping(host):
        print(f"✓ Ollama está rodando em {host}")
        return 0
    print(f"✗ Ollama não responde em {host}")
    return 1


def _cmd_start(_args: argparse.Namespace) -> int:
    host = get_ollama_host()
    if ping(host):
        print(f"✓ Ollama já está rodando em {host}")
        return 0

    print("→ Iniciando Ollama...")
    if not start():
        print("✗ Falha ao iniciar.")
        return 1

    if wait_until_ready(host):
        print(f"✓ Ollama iniciado em {host}")
        return 0

    print("✗ Iniciou, mas não respondeu a tempo.")
    return 1


def _cmd_stop(_args: argparse.Namespace) -> int:
    print("→ Encerrando Ollama...")
    if stop():
        print("✓ Encerrado.")
        return 0
    print("✗ Falha ao encerrar.")
    return 1


def _cmd_restart(_args: argparse.Namespace) -> int:
    _cmd_stop(_args)
    time.sleep(1)
    return _cmd_start(_args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ollama_control",
        description="Controla o ciclo de vida do Ollama (a partir do WSL2).",
    )
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("status", help="Mostra se o Ollama está respondendo.")
    sub.add_parser("start", help="Inicia o Ollama no Windows.")
    sub.add_parser("stop", help="Encerra o Ollama no Windows.")
    sub.add_parser("restart", help="Reinicia o Ollama.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "status": _cmd_status,
        "start": _cmd_start,
        "stop": _cmd_stop,
        "restart": _cmd_restart,
    }
    return handlers[args.action](args)


if __name__ == "__main__":
    sys.exit(main())
