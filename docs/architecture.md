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

## Limitações conhecidas

Testes com o dataset Titanic (891 linhas, 12 colunas) revelaram
comportamentos do `mistral:7b-instruct-q4_K_M` que orientam decisões
futuras.

### O que o modelo faz bem

- Identifica problemas objetivos de qualidade (alta cardinalidade,
  valores ausentes, assimetria).
- Distingue classificação de regressão quando o alvo é reclassificado
  como categórico (ver mitigação abaixo).
- Mantém coerência estrutural do texto (seções pedidas são respeitadas).

### O que o modelo erra

- **Interpretação de correlações com semântica ambígua**: em
  `Pclass × Age` (Pclass 1 = superior), o modelo confunde a direção
  semântica ("classe superior tende a ser mais jovem" — o correto é o
  oposto).
- **Alucinação em pares desconexos**: em `SibSp × Parch`, o modelo
  inventou relação com `Fare`, que não está envolvido no par.
- **Números inventados**: ocasionalmente produz valores ("variância
  1.000") que não constam do resumo.
- **Tratamento de problema multi-alvo**: como o Titanic tem múltiplas
  colunas numéricas, o modelo propõe regressão para prever `Fare` ou
  `Age` — possível, mas não é o objetivo principal do dataset.

### Mitigações aplicadas

1. **Reclassificação de inteiros de baixa cardinalidade** em categóricos
   (≤ 10 valores únicos). Isso resolveu o caso mais grave: `Survived`
   0/1 era tratado como numérico, e o modelo não percebia que era
   classificação.
2. **Prompt com regras estritas**: instruções explícitas sobre sinal de
   correlação, proibição de sugerir pré-processamento como ML, e
   permissão para declarar "não há evidência suficiente".
3. **Limite de bullets por seção** para reduzir divagação.
4. **`max_tokens` ajustado** no preset `balanced` (1024 → 2048) para
   evitar truncamento.

### Recomendação para uso em produção

Para análises que dependem de interpretação correta, preferir o preset
`quality` (`llama3.1:8b-instruct-q4_K_M`) ou um provider comercial
(quando implementado). Modelos de 7B quantizados são viáveis para EDA
assistida, mas não substituem julgamento humano em interpretação
estatística.