"""Interface base para agentes LLM."""

from __future__ import annotations
from abc import ABC, abstractmethod


class AgentError(Exception):
    """Erro genérico de agente."""


class AgentUnavailableError(AgentError):
    """Levantado quando o backend do agente não está acessível."""


class BaseAgent(ABC):
    """Interface comum para todos os agentes LLM.

    Agentes encapsulam a comunicação com um provedor de LLM (Ollama,
    OpenAI, etc.) atrás de uma API uniforme. Isso permite que o resto
    do ``ml_advisor`` não precise saber qual backend está em uso.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Identificador legível (ex.: ``ollama:mistral:7b-instruct``)."""

    @abstractmethod
    def is_available(self) -> bool:
        """Retorna ``True`` se o agente pode ser usado agora."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Gera uma resposta para ``prompt``.

        Args:
            prompt: Texto de entrada do usuário.
            system: Instrução de sistema opcional (papel do modelo).
            temperature: Amostragem. ``None`` usa o default do backend.
            max_tokens: Limite de tokens na resposta.

        Returns:
            Texto da resposta.

        Raises:
            AgentUnavailableError: Se o backend não estiver acessível.
            AgentError: Para outros erros de geração.
        """
