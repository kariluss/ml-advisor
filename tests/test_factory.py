"""Testes para a fábrica de agentes."""

from __future__ import annotations

import pytest

from ml_advisor.agents import AgentUnavailableError, OllamaAgent, create_agent
from ml_advisor.settings import Settings


def test_create_agent_default_is_ollama():
    agent = create_agent()
    assert isinstance(agent, OllamaAgent)
    assert agent.settings.provider == "ollama"


def test_create_agent_with_provider():
    agent = create_agent(provider="ollama")
    assert isinstance(agent, OllamaAgent)


def test_create_agent_with_settings():
    settings = Settings(
        provider="ollama",
        preset="minimal",
        model="phi3:mini",
        timeout=60.0,
        temperature=0.5,
        max_tokens=256,
    )
    agent = create_agent(settings=settings)
    assert agent.model == "phi3:mini"
    assert agent.timeout == 60.0


def test_create_agent_with_override():
    agent = create_agent(model="custom:model")
    assert agent.model == "custom:model"


def test_create_agent_planned_provider_raises():
    with pytest.raises(AgentUnavailableError, match="planejado"):
        create_agent(provider="openai")


def test_create_agent_unknown_provider_raises():
    with pytest.raises(Exception):
        create_agent(provider="nao-existe")
