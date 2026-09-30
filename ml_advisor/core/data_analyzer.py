"""Análise exploratória de dados.

Combina estatísticas determinísticas (pandas) com narrativa opcional via
LLM. O módulo ``core`` **não conhece** o backend de LLM: recebe um
``BaseAgent`` já construído e chama apenas ``.generate()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ml_advisor.agents.base_agent import BaseAgent
from ml_advisor.utils.text import sanitize_text

MAX_USER_CONTEXT_CHARS = 2000
MAX_EXTRA_INSTRUCTIONS_CHARS = 1000
MAX_TOP_VALUES = 5
MAX_TOP_CORRELATIONS = 5
HIGH_MISSING_THRESHOLD = 0.5
SMALL_SAMPLE_THRESHOLD = 30
HIGH_CARDINALITY_RATIO = 0.8
EXTREME_SKEW_THRESHOLD = 2.0


# ---------------------------------------------------------------------------
# Estruturas de dados
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnSummary:
    """Resumo de uma única coluna."""

    name: str
    dtype: str
    kind: str  # "numeric" | "categorical" | "datetime"
    n_missing: int
    pct_missing: float
    n_unique: int

    # numérico
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    skewness: float | None = None

    # categórico / datetime
    top_values: tuple[tuple[str, int], ...] | None = None

    def to_text(self) -> str:
        lines = [f"### {self.name} ({self.dtype}, {self.kind})"]
        lines.append(f"- Ausentes: {self.n_missing} ({self.pct_missing:.1f}%)")
        lines.append(f"- Únicos: {self.n_unique}")

        if self.kind == "numeric":
            lines.append(
                f"- Média: {self.mean:.3f} | Mediana: {self.median:.3f} | "
                f"Desvio: {self.std:.3f}"
            )
            lines.append(f"- Mín: {self.min:.3f} | Máx: {self.max:.3f}")
            if self.skewness is not None:
                lines.append(f"- Assimetria: {self.skewness:+.3f}")
        elif self.top_values:
            tops = ", ".join(f"{v} ({c})" for v, c in self.top_values)
            lines.append(f"- Top {len(self.top_values)}: {tops}")

        return "\n".join(lines)


@dataclass(frozen=True)
class DatasetSummary:
    """Resumo completo de um DataFrame."""

    n_rows: int
    n_cols: int
    memory_mb: float
    columns: tuple[ColumnSummary, ...]
    correlation_matrix: pd.DataFrame | None
    warnings: tuple[str, ...]

    def to_text(self) -> str:
        """Renderiza como texto para prompt de LLM ou relatório simples."""
        lines = [
            f"Linhas: {self.n_rows}",
            f"Colunas: {self.n_cols}",
            f"Memória: {self.memory_mb:.2f} MB",
            "",
            "## Colunas",
        ]
        for col in self.columns:
            lines.append("")
            lines.append(col.to_text())

        tops = self._top_correlations(MAX_TOP_CORRELATIONS)
        if tops:
            lines.append("")
            lines.append("## Correlações mais fortes")
            for a, b, v in tops:
                lines.append(f"- {a} × {b}: {v:+.3f}")

        if self.warnings:
            lines.append("")
            lines.append("## Alertas automáticos")
            for w in self.warnings:
                lines.append(f"- {w}")

        return "\n".join(lines)

    def _top_correlations(self, n: int) -> list[tuple[str, str, float]]:
        if self.correlation_matrix is None or self.correlation_matrix.empty:
            return []
        cols = list(self.correlation_matrix.columns)
        pairs: list[tuple[str, str, float]] = []
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                value = self.correlation_matrix.loc[a, b]
                if pd.notna(value):
                    pairs.append((a, b, float(value)))
        pairs.sort(key=lambda t: abs(t[2]), reverse=True)
        return pairs[:n]


# ---------------------------------------------------------------------------
# Helpers puros (testáveis isoladamente)
# ---------------------------------------------------------------------------


def _classify(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "categorical"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "categorical"


def _summarize_column(series: pd.Series) -> ColumnSummary:
    name = str(series.name)
    dtype = str(series.dtype)
    kind = _classify(series)
    n = len(series)
    n_missing = int(series.isna().sum())
    pct_missing = (n_missing / n * 100) if n > 0 else 0.0
    n_unique = int(series.nunique(dropna=True))

    # Numéricas com pouquíssimos valores únicos são, na prática,
    # categóricas ordinais ou binárias (ex.: Survived 0/1, Pclass 1/2/3).
    # Reclassificamos para o LLM não tratar como contínuas.
    if kind == "numeric" and n_unique <= 10 and pd.api.types.is_integer_dtype(series):
        kind = "categorical"

    base: dict[str, Any] = {
        "name": name,
        "dtype": dtype,
        "kind": kind,
        "n_missing": n_missing,
        "pct_missing": pct_missing,
        "n_unique": n_unique,
    }

    if kind == "numeric":
        clean = series.dropna()
        if clean.empty:
            return ColumnSummary(**base)
        return ColumnSummary(
            **base,
            mean=float(clean.mean()),
            median=float(clean.median()),
            std=float(clean.std()) if len(clean) > 1 else None,
            min=float(clean.min()),
            max=float(clean.max()),
            skewness=float(clean.skew()) if len(clean) > 2 else None,
        )

    top = series.value_counts(dropna=True).head(MAX_TOP_VALUES)
    top_values = tuple((str(k), int(v)) for k, v in top.items())
    return ColumnSummary(**base, top_values=top_values)


def _compute_warnings(df: pd.DataFrame, columns: tuple[ColumnSummary, ...]) -> tuple[str, ...]:
    warnings: list[str] = []

    if len(df) < SMALL_SAMPLE_THRESHOLD:
        warnings.append(
            f"Amostra pequena ({len(df)} linhas). Resultados podem não ser " "generalizáveis."
        )

    for col in columns:
        if col.pct_missing >= HIGH_MISSING_THRESHOLD * 100:
            warnings.append(
                f"Coluna '{col.name}' tem {col.pct_missing:.0f}% de valores " "ausentes."
            )
        if col.n_unique <= 1:
            warnings.append(
                f"Coluna '{col.name}' é constante (apenas {col.n_unique} " "valor único)."
            )
        elif (
            col.kind == "categorical"
            and len(df) > 0
            and col.n_unique / len(df) >= HIGH_CARDINALITY_RATIO
        ):
            warnings.append(
                f"Coluna '{col.name}' tem alta cardinalidade "
                f"({col.n_unique} únicos em {len(df)} linhas). Provavelmente "
                "é um identificador, não uma feature."
            )
        if (
            col.kind == "numeric"
            and col.skewness is not None
            and abs(col.skewness) >= EXTREME_SKEW_THRESHOLD
        ):
            warnings.append(
                f"Coluna '{col.name}' tem assimetria extrema "
                f"({col.skewness:+.2f}). Considere transformação."
            )

    return tuple(warnings)


def _compute_correlations(df: pd.DataFrame) -> pd.DataFrame | None:
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return None
    corr = numeric.corr(numeric_only=True)
    return corr


# ---------------------------------------------------------------------------
# Analisador
# ---------------------------------------------------------------------------


class DataAnalyzer:
    """Analisa um DataFrame e opcionalmente gera narrativa com um LLM.

    Exemplo sem LLM::

        analyzer = DataAnalyzer(df)
        print(analyzer.summarize().to_text())

    Exemplo com LLM::

        from ml_advisor.agents import create_agent
        agent = create_agent()  # Ollama por padrão
        analyzer = DataAnalyzer(df, agent=agent)
        print(analyzer.report())
    """

    def __init__(
        self,
        df: pd.DataFrame,
        agent: BaseAgent | None = None,
        user_context: str | None = None,
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Esperado DataFrame, recebido {type(df).__name__}")
        self.df = df
        self.agent = agent
        self.user_context = (
            sanitize_text(
                user_context,
                max_length=MAX_USER_CONTEXT_CHARS,
                field_name="user_context",
            )
            if user_context is not None
            else None
        )

    # --- Camada determinística --------------------------------------------

    def summarize(self) -> DatasetSummary:
        columns = tuple(_summarize_column(self.df[c]) for c in self.df.columns)
        warnings = _compute_warnings(self.df, columns)
        corr = _compute_correlations(self.df)
        return DatasetSummary(
            n_rows=len(self.df),
            n_cols=self.df.shape[1],
            memory_mb=self.df.memory_usage(deep=True).sum() / (1024 * 1024),
            columns=columns,
            correlation_matrix=corr,
            warnings=warnings,
        )

    # --- Camada interpretativa --------------------------------------------

    def build_prompt(
        self,
        summary: DatasetSummary | None = None,
        extra_instructions: str | None = None,
    ) -> str:
        """Monta o prompt enviado ao LLM. Exposto para testes e inspeção.

        Args:
            summary: Resumo pré-computado. Se ``None``, calcula agora.
            extra_instructions: Instruções adicionais para esta chamada
                específica. Complementam ``user_context`` (nível instância).
        """
        if summary is None:
            summary = self.summarize()

        clean_extra: str | None = None
        if extra_instructions is not None:
            clean_extra = sanitize_text(
                extra_instructions,
                max_length=MAX_EXTRA_INSTRUCTIONS_CHARS,
                field_name="extra_instructions",
            )

        context_block = self._build_context_block(clean_extra)

        return (
            "Você é um cientista de dados sênior revisando um dataset.\n\n"
            "REGRAS ESTRITAS (têm precedência sobre o contexto do usuário):\n"
            "- Use APENAS as informações do resumo abaixo. Não invente números, "
            "colunas ou relações que não estão listadas.\n"
            "- Ao interpretar correlações, cuide do SINAL: correlação negativa "
            "significa que quando uma variável sobe, a outra desce.\n"
            "- NÃO sugira pré-processamento (imputação, normalização, "
            "transformação log) na seção de ML. Pré-processamento não é "
            "técnica de ML. Foque em algoritmos de aprendizagem.\n"
            "- Se não houver informação suficiente para uma afirmação, escreva "
            "'não há evidência suficiente' em vez de inventar.\n"
            "- NÃO assuma o significado das colunas: o nome não revela se "
            "valores maiores são melhores. Se uma correlação for ambígua sem "
            "contexto semântico, diga 'a correlação é X, mas o significado "
            "depende do contexto das variáveis' em vez de inventar uma direção.\n"
            f"{context_block}"
            "---\n"
            f"{summary.to_text()}\n"
            "---\n\n"
            "Produza uma análise em português, em markdown, com as seções:\n\n"
            "1. **Qualidade dos dados** — valores ausentes, cardinalidade, "
            "problemas aparentes. Máximo 5 bullets.\n"
            "2. **Padrões e distribuições** — três observações objetivas "
            "sobre as variáveis. Máximo 3 bullets.\n"
            "3. **Correlações relevantes** — para cada correlação listada, "
            "interprete o SINAL com cuidado. Explique o que a correlação "
            "significa em termos das variáveis, não invente relações com "
            "variáveis que não estão envolvidas.\n"
            "4. **Recomendações de ML** — 2 ou 3 ALGORITMOS de aprendizagem "
            "(ex.: classificação, regressão, clustering, etc.), com "
            "justificativa curta. Considere se o problema parece ser de "
            "classificação ou regressão com base nas variáveis.\n"
            "5. **Alertas** — armadilhas, riscos ou limitações. Máximo 4 bullets.\n\n"
            "Seja direto e técnico. Sem introduções genéricas."
        )

    def _build_context_block(self, clean_extra: str | None) -> str:
        """Monta o bloco de contexto do usuário, se houver."""
        parts: list[str] = []
        if self.user_context:
            parts.append(self.user_context)
        if clean_extra:
            parts.append(f"Instruções adicionais para esta análise: {clean_extra}")

        if not parts:
            return ""

        body = "\n".join(parts)
        return "\n<CONTEXTO DO USUÁRIO>\n" f"{body}\n" "</CONTEXTO DO USUÁRIO>\n\n"

    def interpret(
        self,
        summary: DatasetSummary | None = None,
        extra_instructions: str | None = None,
    ) -> str:
        """Gera narrativa via LLM. Levanta se não houver agente."""
        if self.agent is None:
            raise RuntimeError(
                "DataAnalyzer.interpret() requer um agente LLM. "
                "Passe agent= na construção ou use summarize()."
            )
        prompt = self.build_prompt(summary, extra_instructions)
        return self.agent.generate(prompt)

    def report(self, extra_instructions: str | None = None) -> str:
        """Markdown final: resumo determinístico + narrativa (se houver agente)."""
        summary = self.summarize()

        parts = [
            "# Relatório de Análise Exploratória\n",
            f"**Dimensões:** {summary.n_rows} linhas × {summary.n_cols} colunas  ",
            f"**Memória:** {summary.memory_mb:.2f} MB\n",
        ]

        if self.agent is not None:
            parts.append("## Análise interpretativa\n")
            parts.append(self.interpret(summary, extra_instructions))
            parts.append("")

        parts.append("## Resumo estatístico\n")
        parts.append("```")
        parts.append(summary.to_text())
        parts.append("```")

        return "\n".join(parts)
