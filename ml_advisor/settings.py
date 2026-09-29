"""Configurações centralizadas do ml_advisor.

Prioridade (maior para menor):

1. Argumentos explícitos no código.
2. Variáveis de ambiente ``ML_ADVISOR_*``.
3. Arquivo ``ml_advisor.toml`` (procura em ``$ML_ADVISOR_CONFIG`` ou CWD).
4. Valores do preset escolhido.
5. Defaults do projeto.

Presets são perfis pré-definidos otimizados para diferentes capacidades
de hardware. Ver ``PRESETS``.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path


class SettingsError(ValueError):
    """Erro de configuração."""


PRESETS: dict[str, dict] = {
    "minimal": {
        "model": "phi3:mini-4k-instruct-q4_K_M",
        "timeout": 180.0,
        "temperature": 0.3,
        "max_tokens": 512,
    },
    "balanced": {
        "model": "mistral:7b-instruct-q4_K_M",
        "timeout": 300.0,
        "temperature": 0.2,
        "max_tokens": 1024,
    },
    "quality": {
        "model": "llama3.1:8b-instruct-q4_K_M",
        "timeout": 600.0,
        "temperature": 0.2,
        "max_tokens": 2048,
    },
}

DEFAULT_PRESET = "balanced"
DEFAULT_CONFIG_FILENAME = "ml_advisor.toml"
_ENV_PREFIX = "ML_ADVISOR_"


@dataclass(frozen=True)
class Settings:
    """Configuração resolvida e validada."""

    preset: str
    model: str
    timeout: float
    temperature: float
    max_tokens: int

    def to_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        return "\n".join(
            [
                f"preset:      {self.preset}",
                f"model:       {self.model}",
                f"timeout:     {self.timeout}s",
                f"temperature: {self.temperature}",
                f"max_tokens:  {self.max_tokens}",
            ]
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _env(name: str) -> str | None:
    raw = os.getenv(f"{_ENV_PREFIX}{name}")
    if raw is None or raw.strip() == "":
        return None
    return raw.strip()


def _env_float(name: str) -> float | None:
    raw = _env(name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise SettingsError(f"{_ENV_PREFIX}{name} deve ser número, recebido: {raw!r}") from exc


def _env_int(name: str) -> int | None:
    raw = _env(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise SettingsError(f"{_ENV_PREFIX}{name} deve ser inteiro, recebido: {raw!r}") from exc


def _resolve_config_path(explicit: Path | str | None) -> Path | None:
    """Descobre qual arquivo TOML usar, se houver."""
    if explicit is not None:
        path = Path(explicit)
        if not path.exists():
            raise SettingsError(f"Arquivo de config não encontrado: {path}")
        return path

    env_path = _env("CONFIG")
    if env_path is not None:
        path = Path(env_path)
        if not path.exists():
            raise SettingsError(f"Arquivo de config não encontrado: {path}")
        return path

    cwd_file = Path.cwd() / DEFAULT_CONFIG_FILENAME
    if cwd_file.exists():
        return cwd_file

    return None


def _load_toml_section(path: Path) -> dict:
    try:
        with path.open("rb") as fp:
            data = tomllib.load(fp)
    except tomllib.TOMLDecodeError as exc:
        raise SettingsError(f"TOML inválido em {path}: {exc}") from exc

    section = data.get("ml_advisor", {})
    if not isinstance(section, dict):
        raise SettingsError(f"Seção [ml_advisor] em {path} deve ser uma tabela.")
    return section


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------


def load_settings(
    preset: str | None = None,
    model: str | None = None,
    timeout: float | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    config_path: Path | str | None = None,
) -> Settings:
    """Resolve a configuração efetiva seguindo a cadeia de prioridade."""
    toml_data: dict = {}
    path = _resolve_config_path(config_path)
    if path is not None:
        toml_data = _load_toml_section(path)

    resolved_preset = _first_not_none(
        preset,
        _env("PRESET"),
        toml_data.get("preset"),
        DEFAULT_PRESET,
    )
    if resolved_preset not in PRESETS:
        raise SettingsError(f"Preset inválido: {resolved_preset!r}. " f"Opções: {sorted(PRESETS)}")
    preset_values = PRESETS[resolved_preset]

    resolved_model = _first_not_none(
        model,
        _env("MODEL"),
        toml_data.get("model"),
        preset_values["model"],
    )
    resolved_timeout = _first_not_none(
        timeout,
        _env_float("TIMEOUT"),
        toml_data.get("timeout"),
        preset_values["timeout"],
    )
    resolved_temperature = _first_not_none(
        temperature,
        _env_float("TEMPERATURE"),
        toml_data.get("temperature"),
        preset_values["temperature"],
    )
    resolved_max_tokens = _first_not_none(
        max_tokens,
        _env_int("MAX_TOKENS"),
        toml_data.get("max_tokens"),
        preset_values["max_tokens"],
    )

    if resolved_timeout <= 0:
        raise SettingsError(f"timeout deve ser > 0, recebido: {resolved_timeout}")
    if not 0.0 <= resolved_temperature <= 2.0:
        raise SettingsError(f"temperature deve estar em [0, 2], recebido: {resolved_temperature}")
    if resolved_max_tokens <= 0:
        raise SettingsError(f"max_tokens deve ser > 0, recebido: {resolved_max_tokens}")

    return Settings(
        preset=resolved_preset,
        model=resolved_model,
        timeout=float(resolved_timeout),
        temperature=float(resolved_temperature),
        max_tokens=int(resolved_max_tokens),
    )
