## Extensão: adicionando um novo provider de LLM

O projeto foi projetado para suportar múltiplos backends. Hoje só o
**Ollama** está implementado, mas a arquitetura já comporta OpenAI,
Anthropic e Gemini sem alterações em `core/*`.

### Passos para adicionar um provider

Exemplo: adicionar **OpenAI** (ou qualquer serviço OpenAI-compatible,
como Groq, Together, DeepSeek, OpenRouter, vLLM, LM Studio).

1. **Declarar o SDK como extra opcional** em `pyproject.toml`:

   ```toml
   [project.optional-dependencies]
   openai = ["openai>=1.50"]
   ```

2. **Escrever o agente** em `ml_advisor/agents/openai_agent.py`,
   herdando de `BaseAgent` e implementando `name`, `is_available` e
   `generate`. Importe o SDK **dentro** do `__init__` para permitir
   instalação seletiva.

3. **Marcar como disponível** em `ml_advisor/settings.py`:

   ```python
   AVAILABLE_PROVIDERS = frozenset({"ollama", "openai"})
   ```

   Os presets de `openai` já existem e serão usados imediatamente.

4. **Registrar na fábrica** em `ml_advisor/agents/factory.py`:

   ```python
   if settings.provider == "openai":
       return OpenAIAgent(settings=settings)
   ```

5. **Adicionar testes** em `tests/test_agents.py` (com `respx` ou mock
   do SDK) e rodar `uv run pytest`.

Nenhuma mudança em `core/*` é necessária. O restante do projeto
continua recebendo `BaseAgent` e chamando `.generate()`.

### Design do CLI

O comando `ml-advisor config show` sempre reflete a configuração
efetiva, incluindo o provider. Providers planejados aparecem com aviso
de "planejado" até serem implementados, evitando que o usuário tente
usar algo que não existe.