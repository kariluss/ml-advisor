# Arquitetura

Visão geral dos módulos, princípios de design e decisões tomadas.

## Objetivo

`ml_advisor` é uma biblioteca Python modular que automatiza:

1. Análise exploratória de dados (EDA).
2. Testes de hipóteses estatísticas.
3. Recomendação de técnicas de Machine Learning.
4. Geração de relatórios interpretativos com LLMs locais.

O uso de **LLMs locais** (via Ollama) garante gratuidade, privacidade
dos dados e reprodutibilidade — três requisitos importantes para uso
acadêmico e em contextos com dados sensíveis.

## Estrutura de módulos

```
ml_advisor/
├── core/       Lógica pura (pandas, scipy, sklearn). Sem LLM.
├── agents/     Camada de LLM. Interface BaseAgent + implementações.
├── report/     Renderização final (markdown, HTML).
└── utils/      Infraestrutura (config, controle de serviços, texto).
```

### Princípio de separação

Os módulos `core/*` **não sabem** que existe um LLM. Eles recebem um
`BaseAgent` opcional via injeção de dependência e, se presente, usam
para enriquecer a saída com narrativa. Isso permite:

- Testar `core/*` sem precisar de Ollama rodando.
- Trocar o backend de LLM sem tocar em `core/*`.
- Usar `core/*` de forma "pura" (só estatística) quando o LLM é overkill.

### Fluxo de dados (planejado)

```
DataFrame
   │
   ▼
[core.DataAnalyzer] ──► resumo estruturado (dict/dataclass)
   │
   ├──► [agents.OllamaAgent] ──► narrativa interpretativa
   │
   ▼
[report.ReportGenerator] ──► markdown / HTML final
```

## Decisões de design

Cada decisão abaixo foi tomada durante o desenvolvimento e está
registrada aqui para consulta futura (metodologia do TCC).

### 1. WSL2 + Ollama no Windows

**Contexto:** GPU AMD RX 7600, Windows 11.

**Alternativas consideradas:**
- Ollama dentro do WSL2 com ROCm: suporte experimental, complicado no WSL.
- Ollama no Windows com CPU: estável, funciona, mas sem aceleração.

**Decisão:** Ollama no Windows, acessado do WSL2 via HTTP.

**Consequência:** ponte de rede exige configuração (`OLLAMA_HOST=0.0.0.0`
+ firewall). Mitigado com descoberta automática em `utils/config.py`.

### 2. Layout flat (não `src/`)

**Contexto:** projeto pessoal/TCC, não publicado no PyPI.

**Decisão:** pacote em `ml_advisor/` na raiz.

**Trade-off:** perde a "força de instalação" que o `src/` proporciona,
mas ganha simplicidade para quem está aprendendo.

### 3. `uv` em vez de `venv` + `pip`

**Contexto:** ambiente Python moderno.

**Decisão:** `uv` como gerenciador único (Python, venv, deps, lock).

**Motivos:** velocidade (10–100x), lockfile reprodutível, dispensa
pyenv/pip-tools.

### 4. Agente abstrato (`BaseAgent`)

**Contexto:** o projeto pode querer trocar de Ollama para outro backend.

**Decisão:** interface `BaseAgent` com `generate()` e `is_available()`.

**Benefício:** `core/*` recebe `BaseAgent`, não `OllamaAgent`. Trocar
backend vira adicionar uma subclasse.

### 5. Erros tipados

**Contexto:** distinguir "serviço fora do ar" de "resposta ruim".

**Decisão:**
- `AgentUnavailableError` — rede/serviço indisponível (retry faz sentido).
- `AgentError` — resposta inválida, payload errado, etc.

### 6. `trust_env=False` em todas as chamadas httpx

**Contexto:** httpx respeita variáveis de proxy por padrão.

**Decisão:** desabilitar globalmente no projeto.

**Motivo:** se algum dia `HTTP_PROXY` for definido no ambiente, as
chamadas ao Ollama passariam a ir para o proxy e quebrariam. Isolar é
mais seguro.

### 7. `uv.lock` versionado, `.venv` ignorado

**Decisão:** commitar `uv.lock`, ignorar `.venv/`.

**Motivo:** o lockfile garante builds idênticos; o `.venv` é reconstruível
via `uv sync`.

### 8. Configuração em presets por perfil de hardware

**Contexto:** o projeto precisa rodar em máquinas com capacidades muito
diferentes (8 GB vs 32 GB de RAM).

**Decisão:** três presets (`minimal`, `balanced`, `quality`) com modelos
e parâmetros calibrados. Configuráveis via:

1. Argumento no código.
2. Variável de ambiente (`ML_ADVISOR_*`).
3. Arquivo `ml_advisor.toml`.
4. Default do projeto.

**Benefício:** nenhum usuário precisa editar código para ajustar o
projeto à sua máquina.

### 9. Arquitetura multi-provider extensível

**Contexto:** o projeto pode querer usar OpenAI, Anthropic ou Gemini
além do Ollama.

**Decisão:**
- `Settings.provider` como conceito de primeira classe.
- `PRESETS` organizados por provider.
- `create_agent()` centraliza a instanciação.
- Providers planejados (`openai`, `anthropic`, `gemini`) aparecem na
  config mas levantam `AgentUnavailableError` claro se usados.
- SDKs declarados como extras opcionais no `pyproject.toml` — quem usa
  só Ollama não instala nada pago.

**Estado atual:** apenas `ollama` implementado. Ver seção "Extensão"
abaixo para o guia de adicionar um provider novo.

### 10. Customização de prompts em 3 níveis

**Contexto:** usuários podem querer informar contexto do domínio ou
mudar a persona do modelo.

**Decisão:** três níveis opcionais e combináveis:

1. **System prompt (global)** — via `load_settings(system_prompt=...)`,
   `ML_ADVISOR_SYSTEM_PROMPT` ou TOML. Limite: 1000 caracteres.
2. **Contexto do dataset (por instância)** — `DataAnalyzer(df,
   user_context="...")`. Limite: 2000 caracteres.
3. **Instruções adicionais (por chamada)** — `analyzer.report(
   extra_instructions="...")`. Limite: 1000 caracteres.

**Regras:**
- Sanitização automática (null bytes, espaços múltiplos, quebras).
- Delimitação explícita (`<CONTEXTO DO USUÁRIO>`).
- Regras estritas do sistema têm precedência sobre o contexto do usuário.
- Degradação graciosa: sem customização, comportamento idêntico ao anterior.

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
- Utiliza contexto customizado para desambiguar colunas com semântica
  não óbvia (ex.: `Pclass 1 = primeira classe`).

### O que o modelo erra

- **Alucinação numérica em colunas categóricas**: em algumas execuções
  o modelo atribui "média" e "assimetria" a colunas puramente
  categóricas como `Ticket` e `Cabin`, copiando valores de outras
  colunas (`Fare`). Erro grave e persistente.
- **Interpretação de correlações com semântica ambígua**: mesmo com
  contexto, o modelo às vezes confunde a direção semântica de pares
  como `Pclass × Age`.
- **Alucinação em pares desconexos**: em `SibSp × Parch`, o modelo já
  inventou relação com `Fare`, que não está envolvido no par.
- **Contradição interna**: em uma execução, o modelo afirmou "problema
  de classificação binária" na seção 4 e "não há evidência suficiente"
  na seção 5 do mesmo texto.
- **Duplicação de alertas**: seção 5 pode repetir bullets idênticos.
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
5. **Customização de prompt em 3 níveis** (system, contexto, instruções)
   para que o usuário possa fornecer semântica do domínio.

### Teste com contexto customizado (Titanic)

Após adicionar a feature de customização, testamos novamente com o
`train.csv`, passando:

- **Contexto:** definições semânticas das colunas (`Survived` 0/1,
  `Pclass 1 = primeira classe`, etc.).
- **Instruções adicionais:** "Priorize a discussão sobre sobrevivência".

**Resultado:**

- ✅ A interpretação de `Pclass × Fare` ficou **correta** após o
  contexto (antes o modelo invertia a semântica).
- ✅ O modelo passou a identificar `Survived` como alvo de classificação
  binária explicitamente.
- ❌ Alucinações numéricas persistiram: colunas categóricas (`Ticket`,
  `Cabin`) foram apresentadas com "média" e "assimetria".
- ❌ Contradição interna permaneceu na mesma execução.
- ❌ Alertas duplicados na seção 5.

**Conclusão:** o contexto customizado **mitiga** erros semânticos, mas
não elimina alucinações numéricas — limitação do modelo de 7B
quantizado, não do pipeline. Para análises críticas, recomendar preset
`quality` (`llama3.1:8b-instruct-q4_K_M`) ou provedor comercial.

### Recomendação para uso em produção

Para análises que dependem de interpretação correta, preferir o preset
`quality` ou um provider comercial (quando implementado). Modelos de
7B quantizados são viáveis para EDA assistida, mas não substituem
julgamento humano em interpretação estatística.

## Módulos implementados

### `core/data_analyzer.py`

Análise exploratória: sumário estatístico por coluna, alertas
automáticos, correlações, narrativa opcional via LLM.

### `core/hypothesis_tester.py`

Testes de hipóteses estatísticas com foco em métodos consolidados.

**Escopo deliberadamente restrito** — prioriza testes com base teórica
sólida e interpretação bem estabelecida:

| Teste | Aplicação | H0 | Tamanho de efeito |
|-------|-----------|-----|-------------------|
| Shapiro-Wilk | Uma numérica, n ≤ 5000 | Dados normais | — |
| Welch's t-test | Numérica × binária | Médias iguais | Cohen's d |
| Mann-Whitney U | Idem, complementar não-paramétrico | Distribuições iguais | rank-biserial r |
| Chi² independência | Categórica × categórica | Independência | Cramér's V |
| Pearson | Numérica × numérica | Correlação linear = 0 | r |
| Spearman | Idem, robusto | Correlação monotônica = 0 | ρ |

**Decisões de design:**

- `auto_suggest()` detecta automaticamente quais testes fazem sentido
  para cada par de variáveis (com ou sem `target`).
- Métodos individuais permanecem expostos para uso explícito.
- Welch's t-test **sempre roda com Mann-Whitney complementar**, e o
  resultado inclui nota quando Shapiro-Wilk falha nos grupos.
- Tamanhos de efeito são reportados sempre, com classificação
  qualitativa (negligenciável / pequeno / médio / grande).
- Prompt do LLM inclui regra explícita para **não confundir rejeitar H0
  com provar H1**, e para considerar tamanho de efeito junto com o
  p-valor.

**Fora de escopo (possível trabalho futuro):** ANOVA, Kruskal-Wallis,
teste t pareado, Fisher exato, Anderson-Darling.

### `core/ml_recommender.py`

Sugere algoritmos de ML baseado em perfil objetivo do dataset.

**Perfil detectado** (`DatasetProfile`):

- Tipo de problema: `classification_binary`, `classification_multiclass`,
  `regression` ou `clustering`.
- Amostras, features (numéricas vs categóricas), % de missing.
- Desbalanceamento de classes (razão majority:minority).
- Alta dimensionalidade (≥ 50 features), amostra pequena (< 200 linhas).
- Detecção de colunas tipo ID (inteiras com todos os valores únicos).

**Regras de recomendação por problema:**

| Problema | Baseline | Recomendado | Alternativa / Condicional |
|----------|----------|-------------|---------------------------|
| Classificação binária | Regressão Logística | Random Forest; Gradient Boosting (n ≥ 1000); pesos de classe (se imbalanced); seleção de features (se alta dim.) | — |
| Classificação multiclasse | Regressão Logística multinomial | Random Forest; Gradient Boosting (n ≥ 1000) | — |
| Regressão | Linear / Ridge / Lasso | Random Forest; Gradient Boosting (n ≥ 1000) | Regularização (se alta dim.) |
| Clustering | K-Means | — | DBSCAN; Hierárquico (n < 5000) |

**Prioridade:** `baseline` (sempre comece por aqui), `recomendado`
(escolha principal), `alternativa` (quando faz sentido mas não é a
primeira opção).

**Decisões de design:**

- Threshold de Gradient Boosting em n ≥ 1000: abaixo disso, árvores
  pequenas tendem a superar GB e treinam mais rápido.
- Detecção de ID evita que o usuário inclua colunas sem valor
  preditivo por descuido.
- `interpret()` produz 5 seções: diagnóstico, modelos, pré-processamento,
  cuidados, validação.

## Próximos módulos planejados

- `report/report_generator.py` — consolida todos os módulos de `core/`
  em um relatório único (markdown / HTML).
- `utils/visualization.py` — histogramas, boxplots, matriz de correlação.