# Catálogo de Decisões Técnicas

Este documento registra as decisões técnicas do `ml_advisor` que vão
além do escopo da LLM — thresholds, escolha de testes, heurísticas de
recomendação e regras determinísticas em geral.

Cada decisão tem um **identificador estável** (DEC-XXX) que pode ser
citado no TCC, em issues e em commits.

## Como consultar

- **Onde está a justificativa de X?** Procure a constante no código e
  use o índice abaixo para encontrar o ID.
- **Banca pergunta "por que Y?"** Localize o ID e apresente a
  "Decisão", "Alternativas" e, quando existir, "Base".
- **Adicionando uma nova decisão?** Use o template no final.

## Índice

### Análise de dados (`core/data_analyzer.py`)
- DEC-001 — Threshold de 50% para alerta de missing
- DEC-002 — Amostra mínima de 30 para alerta
- DEC-003 — Razão de 0.8 para detecção de alta cardinalidade
- DEC-004 — Assimetria de 2.0 como limiar de alerta
- DEC-005 — Cinco valores como top-N de categóricas
- DEC-006 — Reclassificação de inteiros de baixa cardinalidade
- DEC-007 — Pearson como correlação padrão

### Testes de hipóteses (`core/hypothesis_tester.py`)
- DEC-008 — Alpha padrão de 0.05
- DEC-009 — Shapiro-Wilk limitado a 5000 observações
- DEC-010 — Welch's t-test como padrão
- DEC-011 — Mann-Whitney U como complemento
- DEC-012 — Correção de Yates no chi²
- DEC-013 — Aviso para células esperadas < 5
- DEC-014 — Cohen's d como tamanho de efeito
- DEC-015 — Cramér's V como tamanho de efeito
- DEC-016 — Rank-biserial como tamanho de efeito
- DEC-017 — Escopo restrito de testes

### Recomendação de ML (`core/ml_recommender.py`)
- DEC-018 — Gradient Boosting só com n ≥ 1000
- DEC-019 — Thresholds de desbalanceamento 0.7 / 0.1
- DEC-020 — Alta dimensionalidade a partir de 50 features
- DEC-021 — Amostra pequena abaixo de 200 linhas
- DEC-022 — Detecção automática de colunas tipo ID
- DEC-023 — Regra de classificação do tipo de problema
- DEC-024 — Priority como enum `baseline/recomendado/alternativa`

### Uso de LLM (prompts em `core/*`)
- DEC-025 — Estrutura hierárquica do prompt
- DEC-026 — Regra "use APENAS as informações abaixo"
- DEC-027 — Regra "não confundir rejeitar H0 com provar H1"
- DEC-028 — Limite de bullets por seção
- DEC-029 — Permissão explícita para "não há evidência suficiente"
- DEC-030 — Delimitação do contexto do usuário
- DEC-031 — Limites de tamanho para prompts do usuário

---

## Análise de dados

### DEC-001

**Threshold de 50% para alerta de valores ausentes**

**Onde:** `data_analyzer.py` (`HIGH_MISSING_THRESHOLD = 0.5`)

**Decisão:** Alertar quando ≥ 50% de uma coluna estiver ausente.

**Alternativas:** 30% geraria ruído em datasets com missing estrutural moderado; 70% deixa passar colunas já inviáveis.

**Base:** Little & Rubin (2019), *Statistical Analysis with Missing Data* — 50% é o ponto em que a coluna deixa de ser informativa sem intervenção. Convenção também adotada em ferramentas de EDA (ydata-profiling).

---

### DEC-002

**Amostra mínima de 30 para alerta**

**Onde:** `data_analyzer.py` (`SMALL_SAMPLE_THRESHOLD = 30`)

**Decisão:** Alertar quando o dataset tiver menos de 30 linhas.

**Alternativas:** 15 era permissivo; 50 alertaria em datasets legítimos.

**Base:** Regra clássica do Teorema Central do Limite, usada como piso prático em testes estatísticos. Kwak & Kim (2017), *Korean J. Anesthesiol.* — n ≥ 30 para aproximação normal razoável.

---

### DEC-003

**Razão de 0.8 para detecção de alta cardinalidade**

**Onde:** `data_analyzer.py` (`HIGH_CARDINALITY_RATIO = 0.8`)

**Decisão:** Alertar quando `n_unique / n_rows ≥ 0.8` em categóricas.

**Alternativas:** valor absoluto (ex.: > 100) falha em datasets pequenos; 1.0 perderia casos quase-únicos.

**Base:** Escolha prática. Durante testes com o Titanic, colunas como `Name` (891/891) e `Ticket` (681/891) se destacaram claramente com essa razão. O valor 0.8 é conservador o suficiente para não disparar em categóricas legítimas com alguma repetição.

---

### DEC-004

**Assimetria de 2.0 como limiar de alerta**

**Onde:** `data_analyzer.py` (`EXTREME_SKEW_THRESHOLD = 2.0`)

**Decisão:** Alertar quando |skewness| ≥ 2.0.

**Alternativas:** 1.0 alertaria em distribuições exponenciais legítimas; 3.0 deixaria passar casos que já afetam modelos lineares.

**Base:** Regra prática consolidada na literatura de modelagem. Hair et al. (2019), *Multivariate Data Analysis* — skewness > 2 é tratado como desvio substancial.

---

### DEC-005

**Cinco valores como top-N de categóricas**

**Onde:** `data_analyzer.py` (`MAX_TOP_VALUES = 5`)

**Decisão:** Reportar no máximo 5 valores mais frequentes.

**Alternativas:** 3 perde padrões de cauda; 10 domina o resumo em categóricas médias.

**Base:** Escolha prática. Alinhada ao conceito de "five-number summary" (Tukey, 1977) e ao formato de saída de ferramentas de EDA.

---

### DEC-006

**Reclassificação de inteiros de baixa cardinalidade**

**Onde:** `data_analyzer.py` (`_summarize_column`)

**Decisão:** Tratar colunas inteiras com ≤ 10 valores únicos como categóricas.

**Alternativas:** confiar no dtype falha em `Survived` (int64 0/1); reclassificar por nome seria frágil.

**Base:** Escolha prática motivada por observação direta. Durante testes com o Titanic, o `mistral:7b` tratava `Survived` como numérica e não percebia que o problema era classificação. Após reclassificar, passou a identificar corretamente.

---

### DEC-007

**Pearson como correlação padrão**

**Onde:** `data_analyzer.py` (`_compute_correlations`)

**Decisão:** Usar Pearson como default; Spearman disponível como opção.

**Alternativas:** Spearman seria mais robusto mas menos familiar em EDA clássico.

**Base:** Convenção dominante em análise exploratória. Pearson mede relação linear, o que é o padrão de reporte em relatórios descritivos.

---

## Testes de hipóteses

### DEC-008

**Alpha padrão de 0.05**

**Onde:** `hypothesis_tester.py` (`DEFAULT_ALPHA = 0.05`)

**Decisão:** Usar α = 0.05.

**Alternativas:** 0.01 (mais conservador, menos poder); 0.10 (mais liberal, útil em exploração).

**Base:** Convenção histórica dominante em estatística aplicada desde Fisher (1925). Ainda que criticada (Wasserstein & Lazar, 2016), continua sendo o valor padrão esperado pela banca e pela literatura.

---

### DEC-009

**Shapiro-Wilk limitado a 5000 observações**

**Onde:** `hypothesis_tester.py` (`SHAPIRO_MAX_N = 5000`)

**Decisão:** Truncar amostras a 5000 antes de rodar Shapiro-Wilk.

**Alternativas:** rodar sem limite geraria p-valores imprecisos; trocar para Anderson-Darling sairia do escopo restrito (DEC-017).

**Base:** Limite técnico documentado. Royston (1995), *J. Royal Stat. Soc. C* — estabelece n = 5000 como limite superior de precisão do W-test. Documentação oficial do `scipy.stats.shapiro` segue a mesma recomendação.

---

### DEC-010

**Welch's t-test como padrão**

**Onde:** `hypothesis_tester.py` (`test_mean_difference`)

**Decisão:** Usar Welch (variâncias desiguais) em vez de Student.

**Alternativas:** Student assume homocedasticidade, raramente verdadeira em dados reais e não testada automaticamente. Testar Levene antes adicionaria complexidade.

**Base:** Welch (1947), *Biometrika* — demonstra a generalização sem suposição de variâncias iguais. Delacre et al. (2017) defendem Welch como padrão em psicologia e áreas aplicadas.

---

### DEC-011

**Mann-Whitney U como complemento**

**Onde:** `hypothesis_tester.py` (`test_mean_difference`)

**Decisão:** Reportar Welch e Mann-Whitney sempre juntos, com nota quando Shapiro falha.

**Alternativas:** só t-test engana quando normalidade falha; só Mann-Whitney perde poder quando dados são normais; roteamento automático esconde informação.

**Base:** Mann & Whitney (1947) para o teste; Demšar (2006), *JMLR* — defende uso conjunto de paramétrico e não-paramétrico em comparações.

---

### DEC-012

**Correção de Yates no chi²**

**Onde:** `hypothesis_tester.py` (`correction=True`)

**Decisão:** Aplicar correção de continuidade por padrão.

**Alternativas:** sem correção, tabelas 2×2 superestimam chi².

**Base:** Yates (1934), *J. Royal Stat. Soc.* — correção original para tabelas de contingência pequenas. Prática conservadora recomendada em Agresti (2018).

---

### DEC-013

**Aviso para células esperadas < 5**

**Onde:** `hypothesis_tester.py` (`test_independence`)

**Decisão:** Avisar quando alguma célula esperada for < 5.

**Alternativas:** bloquear o teste seria restritivo demais; silenciar ocultaria limitação.

**Base:** Regra de Cochran (1954), *Biometrics* — piso prático para validade da aproximação chi².

---

### DEC-014

**Cohen's d como tamanho de efeito**

**Onde:** `hypothesis_tester.py` (`_cohens_d`)

**Decisão:** Reportar d com classificação 0.2 / 0.5 / 0.8.

**Alternativas:** Hedges' g corrige viés em amostras pequenas, mas é muito próximo do d para n razoável; Glass's Δ é menos comum.

**Base:** Cohen (1988), *Statistical Power Analysis*. Sullivan & Feinn (2012) — p-valor sozinho não indica relevância prática.

---

### DEC-015

**Cramér's V como tamanho de efeito**

**Onde:** `hypothesis_tester.py` (`_cramers_v`)

**Decisão:** Reportar V (normalizado em [0, 1]).

**Alternativas:** Phi só faz sentido em 2×2; contingency coefficient é limitado pelo tamanho da tabela.

**Base:** Cramér (1946), *Mathematical Methods of Statistics* — medida generalizada para qualquer tabela.

---

### DEC-016

**Rank-biserial como tamanho de efeito do Mann-Whitney**

**Onde:** `hypothesis_tester.py` (`_rank_biserial`)

**Decisão:** Reportar r baseado em ranks.

**Alternativas:** usar Cohen's d seria metodologicamente inconsistente (Mann-Whitney é sobre distribuições, não médias).

**Base:** Kerby (2014), *Comprehensive Psychology* — fórmula simples derivada de U. Cureton (1956) para a definição original.

---

### DEC-017

**Escopo restrito de testes**

**Onde:** `hypothesis_tester.py` (módulo inteiro)

**Decisão:** Apenas 6 testes: Shapiro, Welch, Mann-Whitney, chi², Pearson, Spearman.

**Alternativas:** cobrir todos os testes clássicos aumentaria escopo sem mudar arquitetura; API plugável complicaria design.

**Base:** Decisão de projeto. Prioriza solidez teórica sobre cobertura. Testes fora do escopo (ANOVA, Kruskal-Wallis, Fisher, Anderson-Darling) ficam como trabalho futuro.

---

## Recomendação de ML

### DEC-018

**Gradient Boosting só com n ≥ 1000**

**Onde:** `ml_recommender.py` (`GRADIENT_BOOSTING_MIN_SAMPLES = 1000`)

**Decisão:** Só recomendar GB com pelo menos 1000 amostras.

**Alternativas:** sempre recomendar ignora datasets acadêmicos (< 500); 500 é agressivo demais sem tuning.

**Base:** Friedman (2001), *Annals of Statistics* — GB é sensível a amostra. Hastie et al. (2009), *ESL* cap. 10 — boosting overfitta em n pequeno. Grinsztajn et al. (2022), *NeurIPS* — confirma superioridade em tabular com nuance de tamanho.

---

### DEC-019

**Thresholds de desbalanceamento 0.7 / 0.1**

**Onde:** `ml_recommender.py` (`IMBALANCE_MAJORITY_THRESHOLD = 0.7`, `MINORITY = 0.1`)

**Decisão:** Duplo critério — maioria ≥ 70% ou minoria ≤ 10%.

**Alternativas:** critério único de maioria perde casos multiclasse 60/10/30; critério único de minoria perde binários 70/30.

**Base:** Chawla et al. (2002), SMOTE — referência canônica de desbalanceamento. He & Garcia (2009), *IEEE TKDE* — discussão de limiares práticos.

---

### DEC-020

**Alta dimensionalidade a partir de 50 features**

**Onde:** `ml_recommender.py` (`HIGH_DIMENSIONALITY_MIN = 50`)

**Decisão:** Alertar a partir de 50 features.

**Alternativas:** limiar relativo a n (p > n) é mais preciso mas falha em features informativas; sem alerta omite limitação real.

**Base:** Escolha prática. Bellman (1961) introduz o termo "curse of dimensionality"; 50 é um piso prático onde regularização costuma começar a fazer diferença visível.

---

### DEC-021

**Amostra pequena abaixo de 200 linhas**

**Onde:** `ml_recommender.py` (`SMALL_SAMPLE_MAX = 200`)

**Decisão:** Alertar quando n < 200 para treino de modelos.

**Alternativas:** 100 é muito permissivo; 500 alertaria em metade dos datasets acadêmicos.

**Base:** Escolha prática. 200 é compatível com validação cruzada k=5 (folds de ~40 amostras), que é o piso operacional usual para estimativas razoáveis.

---

### DEC-022

**Detecção automática de colunas tipo ID**

**Onde:** `ml_recommender.py` (`_id_like_columns`)

**Decisão:** Detectar inteiras com `n_unique == n_rows`.

**Alternativas:** depender do usuário é frágil; heurística por nome ("id") falha em outros idiomas.

**Base:** Escolha prática. Colunas tipo ID não têm sinal preditivo — o modelo aprende a mapear IDs e falha em generalizar. Critério objetivo evita heurística linguística.

---

### DEC-023

**Regra de classificação do tipo de problema**

**Onde:** `ml_recommender.py` (`_classify_problem`)

**Decisão:**

| Condição | Tipo |
|----------|------|
| Sem target | `clustering` |
| Target numérico com > 20 únicos | `regression` |
| Target com 2 valores únicos | `classification_binary` |
| Target com > 2 e ≤ 20 únicos | `classification_multiclass` |

**Alternativas:** confiar no dtype falha em `Survived` (int64 binário); pedir ao usuário perde automação.

**Base:** Escolha prática alinhada com DEC-006. Consistente com a convenção de AutoML (auto-sklearn, TPOT).

---

### DEC-024

**Priority como enum qualitativo**

**Onde:** `ml_recommender.py` (`Recommendation.priority`)

**Decisão:** Três níveis — `baseline`, `recomendado`, `alternativa`.

**Alternativas:** score numérico implicaria precisão falsa; ordem implícita na lista seria ambígua.

**Base:** Decisão de design. Segue o padrão de guias de modelagem (scikit-learn "Choosing the right estimator") que organizam sugestões em camadas.

---

## Uso de LLM

### DEC-025

**Estrutura hierárquica do prompt**

**Onde:** `build_prompt` nos três módulos

**Decisão:** Ordem — persona → regras → contexto → dados → tarefa.

**Alternativas:** prompt linear sem separadores funcionou pior nos testes; JSON estruturado confunde o modelo.

**Base:** Empírica. A estrutura atual produziu melhor aderência às regras com `mistral:7b` do que as variações testadas. Alinhada com práticas documentadas de prompt engineering (Wei et al., 2022 — chain-of-thought).

---

### DEC-026

**Regra "use APENAS as informações abaixo"**

**Onde:** `build_prompt` nos três módulos

**Decisão:** Proibir explicitamente invenção de dados.

**Alternativas:** confiar no prompt implícito falha em `mistral:7b`; pós-processamento para detectar números seria complexo.

**Base:** Ji et al. (2023), *ACM Computing Surveys* — survey de alucinação em LLMs. Regra explícita mitiga mas não elimina; limitação documentada em `architecture.md`.

---

### DEC-027

**Regra "não confundir rejeitar H0 com provar H1"**

**Onde:** `hypothesis_tester.py` (`build_prompt`)

**Decisão:** Instruir o modelo sobre a distinção.

**Alternativas:** deixar implícito resulta em afirmações causais indevidas.

**Base:** Erro conceitual comum. Wasserstein & Lazar (2016), *The American Statistician* — crítica ao uso ritual de p-valores inclui essa confusão. Regra alinha o modelo à interpretação correta.

---

### DEC-028

**Limite de bullets por seção**

**Onde:** `build_prompt` (Máximo N bullets por seção)

**Decisão:** Limitar número de bullets no output.

**Alternativas:** sem limite, `mistral:7b` divaga e repete; limite por tokens é indireto.

**Base:** Empírica. Limites por seção reduzem divagação e truncamento. Observado durante testes com o Titanic.

---

### DEC-029

**Permissão explícita para "não há evidência suficiente"**

**Onde:** `build_prompt` nos três módulos

**Decisão:** Autorizar o modelo a declarar insuficiência.

**Alternativas:** sem permissão, o modelo preenche seções mesmo sem base.

**Base:** Lin, Hilton & Evans (2022), *TACL* — modelos treinados a expressar incerteza reduzem alucinação. Resultado prático: respostas mais honestas em datasets pobres.

---

### DEC-030

**Delimitação do contexto do usuário**

**Onde:** `_build_context_block` nos três módulos

**Decisão:** Envolver contexto em `<CONTEXTO DO USUÁRIO>`.

**Alternativas:** concatenação direta faz o modelo tratar contexto como dado.

**Base:** Convenção documentada pela Anthropic e OpenAI sobre uso de tags XML em prompts. Reduz ambiguidade entre instrução e dado.

---

### DEC-031

**Limites de tamanho para prompts do usuário**

**Onde:** `utils/text.py` (`MAX_*_CHARS`)

**Decisão:**

- `user_context`: 2000 caracteres.
- `extra_instructions`: 1000 caracteres.
- `system_prompt`: 1000 caracteres.

**Alternativas:** sem limite, usuário pode estourar a janela de contexto; limites baixos frustrariam casos legítimos.

**Base:** Escolha prática. Janela típica do `mistral:7b` é ~8k tokens (~6k caracteres em português), então os limites deixam folga para o resumo estatístico e a tarefa.
