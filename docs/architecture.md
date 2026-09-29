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
└── utils/      Infraestrutura (config, controle de serviços).
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

## Próximos módulos planejados

- `core/data_analyzer.py` — estatísticas descritivas + narrativa LLM.
- `core/hypothesis_tester.py` — testes estatísticos (Shapiro-Wilk, t-test,
  chi²) com interpretação.
- `core/ml_recommender.py` — sugere algoritmos com base nas características
  do dataset.
- `report/report_generator.py` — consolida tudo em markdown/HTML.