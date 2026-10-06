"""Testes para o módulo de configurações."""

from __future__ import annotations

import os

import pytest

from ml_advisor.settings import (
    AVAILABLE_PROVIDERS,
    DEFAULT_PRESET,
    DEFAULT_PROVIDER,
    PRESETS,
    SettingsError,
    list_presets,
    load_settings,
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("ML_ADVISOR_"):
            monkeypatch.delenv(key, raising=False)
    yield


# --- Defaults e presets ----------------------------------------------------

def test_defaults():
    s = load_settings()
    assert s.provider == DEFAULT_PROVIDER
    assert s.preset == DEFAULT_PRESET
    assert s.model == PRESETS[DEFAULT_PROVIDER][DEFAULT_PRESET]["model"]


def test_preset_via_arg():
    s = load_settings(preset="minimal")
    assert s.preset == "minimal"
    assert s.model == PRESETS["ollama"]["minimal"]["model"]


def test_preset_via_env(monkeypatch):
    monkeypatch.setenv("ML_ADVISOR_PRESET", "quality")
    s = load_settings()
    assert s.preset == "quality"


def test_preset_via_toml(tmp_path):
    cfg = tmp_path / "ml_advisor.toml"
    cfg.write_text('[ml_advisor]\npreset = "minimal"\n')
    s = load_settings(config_path=cfg)
    assert s.preset == "minimal"


def test_preset_arg_beats_env(monkeypatch):
    monkeypatch.setenv("ML_ADVISOR_PRESET", "quality")
    s = load_settings(preset="minimal")
    assert s.preset == "minimal"


# --- Provider --------------------------------------------------------------

def test_provider_default():
    s = load_settings()
    assert s.provider == "ollama"
    assert s.is_available is True


def test_provider_via_arg():
    s = load_settings(provider="openai")
    assert s.provider == "openai"
    assert s.is_available is False  # planejado, mas não implementado


def test_provider_via_env(monkeypatch):
    monkeypatch.setenv("ML_ADVISOR_PROVIDER", "openai")
    s = load_settings()
    assert s.provider == "openai"


def test_provider_arg_beats_env(monkeypatch):
    monkeypatch.setenv("ML_ADVISOR_PROVIDER", "openai")
    s = load_settings(provider="ollama")
    assert s.provider == "ollama"


def test_provider_invalid():
    with pytest.raises(SettingsError, match="Provider inválido"):
        load_settings(provider="nao-existe")


def test_preset_valida_por_provider():
    """Preset válido em um provider pode não existir em outro."""
    # 'minimal' existe em todos os providers atuais, então esse teste
    # valida que o mecanismo funciona com um preset fictício.
    with pytest.raises(SettingsError, match="Preset inválido"):
        load_settings(provider="ollama", preset="nao-existe")


# --- Cadeia de prioridade --------------------------------------------------

def test_model_override():
    s = load_settings(model="custom:model")
    assert s.model == "custom:model"
    assert s.preset == DEFAULT_PRESET


def test_env_beats_toml(tmp_path, monkeypatch):
    cfg = tmp_path / "ml_advisor.toml"
    cfg.write_text('[ml_advisor]\nmodel = "toml:model"\n')
    monkeypatch.setenv("ML_ADVISOR_MODEL", "env:model")
    s = load_settings(config_path=cfg)
    assert s.model == "env:model"


def test_arg_beats_env(monkeypatch):
    monkeypatch.setenv("ML_ADVISOR_MODEL", "env:model")
    s = load_settings(model="arg:model")
    assert s.model == "arg:model"


# --- Validação -------------------------------------------------------------

def test_invalid_timeout_env(monkeypatch):
    monkeypatch.setenv("ML_ADVISOR_TIMEOUT", "abc")
    with pytest.raises(SettingsError, match="número"):
        load_settings()


def test_timeout_must_be_positive():
    with pytest.raises(SettingsError, match="timeout"):
        load_settings(timeout=-1.0)


def test_temperature_range():
    with pytest.raises(SettingsError, match="temperature"):
        load_settings(temperature=3.0)


def test_config_path_missing():
    with pytest.raises(SettingsError, match="não encontrado"):
        load_settings(config_path="/caminho/que/nao/existe.toml")


def test_to_dict_roundtrip():
    s = load_settings(preset="minimal")
    d = s.to_dict()
    assert d["preset"] == "minimal"
    assert d["provider"] == "ollama"
    assert set(d) == {
        "provider",
        "preset",
        "model",
        "timeout",
        "temperature",
        "max_tokens",
        "system_prompt",
    }


# --- Utilitários -----------------------------------------------------------

def test_list_presets_all():
    all_presets = list_presets()
    assert set(all_presets) == set(PRESETS)


def test_list_presets_by_provider():
    ollama_presets = list_presets("ollama")
    assert set(ollama_presets) == set(PRESETS["ollama"])


def test_list_presets_invalid_provider():
    with pytest.raises(SettingsError, match="desconhecido"):
        list_presets("nao-existe")


def test_available_providers_is_subset():
    """Todo provider implementado tem presets definidos."""
    for provider in AVAILABLE_PROVIDERS:
        assert provider in PRESETS


def test_system_prompt_via_arg():
    s = load_settings(system_prompt="Você é um auditor.")
    assert s.system_prompt == "Você é um auditor."


def test_system_prompt_via_env(monkeypatch):
    monkeypatch.setenv("ML_ADVISOR_SYSTEM_PROMPT", "Você é um auditor.")
    s = load_settings()
    assert s.system_prompt == "Você é um auditor."


def test_system_prompt_via_toml(tmp_path):
    cfg = tmp_path / "ml_advisor.toml"
    cfg.write_text('[ml_advisor]\nsystem_prompt = "Você é um auditor."\n')
    s = load_settings(config_path=cfg)
    assert s.system_prompt == "Você é um auditor."


def test_system_prompt_default_none():
    s = load_settings()
    assert s.system_prompt is None


def test_system_prompt_sanitizado():
    s = load_settings(system_prompt="  Você   é\x00 um auditor.  ")
    assert s.system_prompt == "Você é um auditor."


def test_system_prompt_excede_tamanho():
    with pytest.raises(ValueError, match="máximo"):
        load_settings(system_prompt="x" * 1001)
