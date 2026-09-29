"""Agente LLM que fala com o Ollama via HTTP."""

from __future__ import annotations

import httpx

from ml_advisor.agents.base_agent import (
    AgentError,
    AgentUnavailableError,
    BaseAgent,
)
from ml_advisor.settings import Settings, load_settings
from ml_advisor.utils.config import get_ollama_host


class OllamaAgent(BaseAgent):
    """Agente que usa um modelo servido pelo Ollama.

    Aceita configuração de três formas:

    1. ``OllamaAgent(settings=...)`` — passa um ``Settings`` pronto.
    2. ``OllamaAgent(model="...", timeout=...)`` — overrides pontuais; o
       restante é resolvido via ``load_settings()`` (env vars, TOML, preset).
    3. ``OllamaAgent()`` — usa apenas a cadeia padrão de configuração.
    """

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        timeout: float | None = None,
        settings: Settings | None = None,
    ) -> None:
        if settings is None:
            settings = load_settings(model=model, timeout=timeout)
        self.settings = settings
        self.host = host or get_ollama_host()

    @property
    def model(self) -> str:
        return self.settings.model

    @property
    def timeout(self) -> float:
        return self.settings.timeout

    @property
    def name(self) -> str:
        return f"ollama:{self.settings.model}"

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

        if temperature is None:
            temperature = self.settings.temperature
        if max_tokens is None:
            max_tokens = self.settings.max_tokens

        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system is not None:
            payload["system"] = system

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
