"""Agente LLM que fala com o Ollama via HTTP."""

from __future__ import annotations

import httpx

from ml_advisor.agents.base_agent import (
    AgentError,
    AgentUnavailableError,
    BaseAgent,
)
from ml_advisor.utils.config import get_ollama_host

DEFAULT_TIMEOUT = 300.0


class OllamaAgent(BaseAgent):
    """Agente que usa um modelo servido pelo Ollama.

    Exemplo:
        >>> agent = OllamaAgent(model="mistral:7b-instruct-q4_K_M")
        >>> agent.is_available()
        True
        >>> agent.generate("Explique overfitting em uma frase.")
        '...'
    """

    def __init__(
        self,
        model: str,
        host: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.model = model
        self.host = host or get_ollama_host()
        self.timeout = timeout

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    def is_available(self) -> bool:
        try:
            response = httpx.get(
                f"{self.host}/api/tags",
                timeout=5.0,
                trust_env=False,
            )
        except httpx.HTTPError:
            return False

        if response.status_code != 200:
            return False

        models = [m["name"] for m in response.json().get("models", [])]
        return self.model in models

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        if not prompt.strip():
            raise AgentError("prompt vazio")

        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system is not None:
            payload["system"] = system

        options: dict = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if options:
            payload["options"] = options

        try:
            response = httpx.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout,
                trust_env=False,
            )
        except httpx.ConnectError as exc:
            raise AgentUnavailableError(
                f"Não foi possível conectar em {self.host}. " "O Ollama está rodando?"
            ) from exc
        except httpx.HTTPError as exc:
            raise AgentError(f"Erro de rede ao chamar Ollama: {exc}") from exc

        if response.status_code != 200:
            raise AgentError(f"Ollama retornou {response.status_code}: {response.text[:200]}")

        text = response.json().get("response", "").strip()
        if not text:
            raise AgentError("Ollama retornou resposta vazia")
        return text
