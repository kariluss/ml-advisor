"""Testes para os agentes LLM."""

from __future__ import annotations

import httpx
import pytest
import respx

from ml_advisor.agents.base_agent import AgentError, AgentUnavailableError
from ml_advisor.agents.ollama_agent import OllamaAgent

HOST = "http://test-ollama:11434"
MODEL = "test-model:latest"


@pytest.fixture
def agent() -> OllamaAgent:
    return OllamaAgent(model=MODEL, host=HOST, timeout=5.0)


@respx.mock
def test_is_available_true(agent: OllamaAgent) -> None:
    respx.get(f"{HOST}/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": MODEL}]})
    )
    assert agent.is_available() is True


@respx.mock
def test_is_available_model_missing(agent: OllamaAgent) -> None:
    respx.get(f"{HOST}/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "other"}]})
    )
    assert agent.is_available() is False


@respx.mock
def test_is_available_connection_error(agent: OllamaAgent) -> None:
    respx.get(f"{HOST}/api/tags").mock(side_effect=httpx.ConnectError("boom"))
    assert agent.is_available() is False


@respx.mock
def test_generate_ok(agent: OllamaAgent) -> None:
    respx.post(f"{HOST}/api/generate").mock(
        return_value=httpx.Response(200, json={"response": "Olá, mundo."})
    )
    assert agent.generate("Diga oi") == "Olá, mundo."


@respx.mock
def test_generate_payload_includes_system_and_options(agent: OllamaAgent) -> None:
    route = respx.post(f"{HOST}/api/generate").mock(
        return_value=httpx.Response(200, json={"response": "ok"})
    )
    agent.generate(
        "teste",
        system="Você é um assistente.",
        temperature=0.2,
        max_tokens=50,
    )
    request_body = route.calls.last.request.content
    assert b'"system"' in request_body
    assert b'"temperature"' in request_body
    assert b'"num_predict"' in request_body


def test_generate_empty_prompt(agent: OllamaAgent) -> None:
    with pytest.raises(AgentError, match="vazio"):
        agent.generate("   ")


@respx.mock
def test_generate_connection_error(agent: OllamaAgent) -> None:
    respx.post(f"{HOST}/api/generate").mock(side_effect=httpx.ConnectError("boom"))
    with pytest.raises(AgentUnavailableError):
        agent.generate("Diga oi")


@respx.mock
def test_generate_http_error_status(agent: OllamaAgent) -> None:
    respx.post(f"{HOST}/api/generate").mock(
        return_value=httpx.Response(500, text="internal error")
    )
    with pytest.raises(AgentError, match="500"):
        agent.generate("Diga oi")


@respx.mock
def test_generate_empty_response(agent: OllamaAgent) -> None:
    respx.post(f"{HOST}/api/generate").mock(
        return_value=httpx.Response(200, json={"response": ""})
    )
    with pytest.raises(AgentError, match="vazia"):
        agent.generate("Diga oi")


def test_agent_name(agent: OllamaAgent) -> None:
    assert agent.name == f"ollama:{MODEL}"
