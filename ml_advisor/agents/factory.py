"""Fábrica de agentes LLM.

Resolve o provider a partir do ``Settings`` e instancia o agente correto.
Providers não implementados levantam ``AgentUnavailableError`` com
mensagem orientando o que fazer.
"""

from __future__ import annotations

from ml_advisor.agents.base_agent import AgentUnavailableError, BaseAgent
from ml_advisor.agents.ollama_agent import OllamaAgent
from ml_advisor.settings import (
    AVAILABLE_PROVIDERS,
    PLANNED_PROVIDERS,
    Settings,
    load_settings,
)


def create_agent(
    provider: str | None = None,
    settings: Settings | None = None,
    **settings_overrides,
) -> BaseAgent:
    """Cria um agente conforme a configuração.

    Args:
        provider: Nome do provider. Se ``None``, usa a cadeia de
            ``load_settings``.
        settings: Instância de ``Settings`` já resolvida. Se passada,
            ignora ``provider`` e ``settings_overrides``.
        **settings_overrides: Passados para ``load_settings`` quando
            ``settings`` é ``None`` (ex.: ``model="..."``).

    Returns:
        Uma instância de ``BaseAgent``.

    Raises:
        AgentUnavailableError: Se o provider não estiver implementado
            ou o SDK necessário não estiver instalado.
    """
    if settings is None:
        settings = load_settings(provider=provider, **settings_overrides)

    if settings.provider not in AVAILABLE_PROVIDERS:
        _raise_unavailable(settings.provider)

    if settings.provider == "ollama":
        return OllamaAgent(settings=settings)

    # Segurança: se chegar aqui, um provider foi adicionado a
    # AVAILABLE_PROVIDERS sem implementação correspondente.
    raise AgentUnavailableError(
        f"Provider {settings.provider!r} está marcado como disponível, "
        "mas não há implementação. Isso é um bug de configuração."
    )


def _raise_unavailable(provider: str) -> None:
    if provider in PLANNED_PROVIDERS:
        raise AgentUnavailableError(
            f"Provider {provider!r} está planejado mas ainda não foi "
            "implementado. Use 'ollama' por enquanto."
        )
    raise AgentUnavailableError(f"Provider desconhecido: {provider!r}")
