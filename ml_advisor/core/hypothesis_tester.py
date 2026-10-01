"""Testes de hipóteses estatísticas.

Combina testes determinísticos (scipy) com narrativa opcional via LLM.
Escopo focado em testes de uso consolidado: normalidade, diferença de
médias, independência e correlação.

Princípios:
- Todos os testes retornam ``TestResult`` uniformes.
- Missing values são dropados linha a linha (pairwise) por teste.
- Tamanhos de efeito são sempre reportados quando aplicáveis.
- O módulo não conhece LLM: recebe ``BaseAgent`` opcional.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from ml_advisor.agents.base_agent import BaseAgent
from ml_advisor.utils.text import sanitize_text

DEFAULT_ALPHA = 0.05
SHAPIRO_MAX_N = 5000
MAX_EXTRA_INSTRUCTIONS_CHARS = 1000
MAX_USER_CONTEXT_CHARS = 2000


# ---------------------------------------------------------------------------
# Estruturas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TestResult:
    """Resultado uniforme de um teste estatístico."""

    __test__ = False

    test_name: str
    variables: tuple[str, ...]
    n: int
    statistic: float
    p_value: float
    alpha: float
    effect_size: float | None = None
    effect_size_name: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def reject_h0(self) -> bool:
        return self.p_value < self.alpha

    @property
    def decision(self) -> str:
        return "rejeitar H0" if self.reject_h0 else "não rejeitar H0"

    def summary(self) -> str:
        lines = [
            f"### {self.test_name}",
            f"- Variáveis: {' × '.join(self.variables)}",
            f"- n: {self.n}",
            f"- Estatística: {self.statistic:.4f}",
            f"- p-valor: {_format_pvalue(self.p_value)}",
            f"- Decisão (α = {self.alpha}): {self.decision}",
        ]
        if self.effect_size is not None and self.effect_size_name:
            magnitude = _describe_effect(self.effect_size, self.effect_size_name)
            lines.append(
                f"- Tamanho de efeito: {self.effect_size_name} = "
                f"{self.effect_size:+.4f} ({magnitude})"
            )
        for note in self.notes:
            lines.append(f"- ⚠ {note}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers de formatação/interpretação
# ---------------------------------------------------------------------------


def _format_pvalue(p: float) -> str:
    if p < 1e-15:
        return "< 1e-15"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def _describe_effect(value: float, name: str) -> str:
    v = abs(value)
    if "V" in name:  # Cramér's V
        if v < 0.1:
            return "negligenciável"
        if v < 0.3:
            return "pequeno"
        if v < 0.5:
            return "médio"
        return "grande"
    if "d" in name:  # Cohen's d
        if v < 0.2:
            return "negligenciável"
        if v < 0.5:
            return "pequeno"
        if v < 0.8:
            return "médio"
        return "grande"
    # Correlações (r)
    if v < 0.1:
        return "negligenciável"
    if v < 0.3:
        return "fraco"
    if v < 0.5:
        return "moderado"
    if v < 0.7:
        return "forte"
    return "muito forte"


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)
    pooled = ((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2)
    if pooled == 0:
        return 0.0
    return (a.mean() - b.mean()) / np.sqrt(pooled)


def _cramers_v(chi2: float, n: int, table_shape: tuple[int, int]) -> float:
    r, k = table_shape
    denominator = n * min(r - 1, k - 1)
    if denominator == 0:
        return 0.0
    return float(np.sqrt(chi2 / denominator))


def _rank_biserial(u_stat: float, n1: int, n2: int) -> float:
    return (2 * u_stat) / (n1 * n2) - 1


# ---------------------------------------------------------------------------
# Analisador
# ---------------------------------------------------------------------------


class HypothesisTester:
    """Executa testes de hipóteses estatísticos com interpretação opcional.

    Exemplo simples (sem LLM)::

        tester = HypothesisTester(df)
        results = tester.auto_suggest(target="Survived")
        print(tester.report(results))

    Exemplo com LLM::

        tester = HypothesisTester(df, agent=agent)
        results = tester.auto_suggest(target="Survived")
        print(tester.report(results))
    """

    def __init__(
        self,
        df: pd.DataFrame,
        agent: BaseAgent | None = None,
        alpha: float = DEFAULT_ALPHA,
        user_context: str | None = None,
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Esperado DataFrame, recebido {type(df).__name__}")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha deve estar em (0, 1), recebido: {alpha}")

        self.df = df
        self.agent = agent
        self.alpha = alpha
        self.user_context = (
            sanitize_text(
                user_context,
                max_length=MAX_USER_CONTEXT_CHARS,
                field_name="user_context",
            )
            if user_context is not None
            else None
        )

    # --- Helpers internos -------------------------------------------------

    def _numeric_columns(self) -> list[str]:
        return list(self.df.select_dtypes(include="number").columns)

    def _categorical_columns(self) -> list[str]:
        cols = []
        for c in self.df.columns:
            if c in self._numeric_columns():
                continue
            if self.df[c].nunique(dropna=True) <= 20:
                cols.append(c)
        return cols

    def _binary_columns(self) -> list[str]:
        return [c for c in self.df.columns if self.df[c].nunique(dropna=True) == 2]

    def _clean_pair(self, col_a: str, col_b: str) -> pd.DataFrame:
        return self.df[[col_a, col_b]].dropna()

    def _clean_single(self, col: str) -> pd.Series:
        return self.df[col].dropna()

    # --- Testes individuais -----------------------------------------------

    def test_normality(self, column: str) -> TestResult:
        """Shapiro-Wilk. H0: a amostra segue distribuição normal."""
        series = self._clean_single(column)
        n = len(series)

        if n < 3:
            raise ValueError(f"n insuficiente para Shapiro-Wilk: {n}")

        notes: tuple[str, ...] = ()
        data = series.to_numpy()
        if n > SHAPIRO_MAX_N:
            data = data[:SHAPIRO_MAX_N]
            notes = (
                f"amostra limitada a {SHAPIRO_MAX_N} observações "
                f"(original: {n}) para o teste de Shapiro-Wilk",
            )
            n = SHAPIRO_MAX_N

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, p = stats.shapiro(data)

        return TestResult(
            test_name="Shapiro-Wilk (normalidade)",
            variables=(column,),
            n=n,
            statistic=float(stat),
            p_value=float(p),
            alpha=self.alpha,
            notes=notes,
        )

    def test_mean_difference(self, numeric: str, group: str) -> TestResult:
        """Welch's t-test + Mann-Whitney U. H0: as duas populações têm
        a mesma média / distribuição."""
        if numeric == group:
            raise ValueError(f"'{numeric}' e '{group}' devem ser colunas diferentes")
        df = self._clean_pair(numeric, group)
        if df[group].nunique() != 2:
            raise ValueError(
                f"'{group}' deve ter exatamente 2 valores únicos, " f"tem {df[group].nunique()}"
            )

        levels = sorted(df[group].unique(), key=str)
        a = df.loc[df[group] == levels[0], numeric].to_numpy()
        b = df.loc[df[group] == levels[1], numeric].to_numpy()

        if len(a) < 2 or len(b) < 2:
            raise ValueError("Cada grupo precisa de pelo menos 2 observações")

        t_stat, t_p = stats.ttest_ind(a, b, equal_var=False)
        d = _cohens_d(a, b)

        # Se a normalidade falha em algum grupo, o t-test pode não ser adequado.
        # Roda Mann-Whitney como complemento e adiciona nota.
        u_stat, u_p = stats.mannwhitneyu(a, b, alternative="two-sided")
        rb = _rank_biserial(float(u_stat), len(a), len(b))

        notes = []
        shapiro_ok = True
        for label, arr in [(levels[0], a), (levels[1], b)]:
            if len(arr) >= 3:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _, sw_p = stats.shapiro(arr[:SHAPIRO_MAX_N])
                if sw_p < self.alpha:
                    shapiro_ok = False
                    notes.append(
                        f"grupo '{label}' não passa em Shapiro-Wilk; " "considere Mann-Whitney U"
                    )

        return TestResult(
            test_name=f"Welch's t-test ({numeric} por {group})",
            variables=(numeric, group),
            n=len(a) + len(b),
            statistic=float(t_stat),
            p_value=float(t_p),
            alpha=self.alpha,
            effect_size=d,
            effect_size_name="Cohen's d",
            notes=tuple(notes)
            + (
                f"Mann-Whitney U complementar: U = {u_stat:.2f}, "
                f"p = {_format_pvalue(u_p)}, rank-biserial r = {rb:+.4f}"
                + ("" if shapiro_ok else " (preferível dado que Shapiro falhou)"),
            ),
        )

    def test_independence(self, cat_a: str, cat_b: str) -> TestResult:
        """Chi² de independência. H0: as variáveis são independentes."""
        df = self._clean_pair(cat_a, cat_b)
        table = pd.crosstab(df[cat_a], df[cat_b])
        if table.shape[0] < 2 or table.shape[1] < 2:
            raise ValueError(f"Tabela de contingência degenerada: {table.shape}")

        chi2, p, dof, _ = stats.chi2_contingency(table, correction=True)
        v = _cramers_v(chi2, int(table.values.sum()), table.shape)

        notes = []
        if table.values.min() < 5:
            notes.append(
                "algumas células têm frequência esperada < 5; "
                "resultado do chi² pode ser impreciso"
            )

        return TestResult(
            test_name=f"Chi² de independência ({cat_a} × {cat_b})",
            variables=(cat_a, cat_b),
            n=int(table.values.sum()),
            statistic=float(chi2),
            p_value=float(p),
            alpha=self.alpha,
            effect_size=v,
            effect_size_name="Cramér's V",
            notes=tuple(notes) + (f"graus de liberdade: {dof}",),
        )

    def test_correlation(self, num_a: str, num_b: str, method: str = "pearson") -> TestResult:
        """Pearson ou Spearman. H0: correlação = 0."""
        df = self._clean_pair(num_a, num_b)
        if len(df) < 3:
            raise ValueError(f"n insuficiente para correlação: {len(df)}")

        x = df[num_a].to_numpy()
        y = df[num_b].to_numpy()

        if method == "pearson":
            coef, p = stats.pearsonr(x, y)
            name = "r (Pearson)"
        elif method == "spearman":
            coef, p = stats.spearmanr(x, y)
            name = "ρ (Spearman)"
        else:
            raise ValueError(f"method inválido: {method}")

        return TestResult(
            test_name=f"Correlação {method.capitalize()} ({num_a} × {num_b})",
            variables=(num_a, num_b),
            n=len(df),
            statistic=float(coef),
            p_value=float(p),
            alpha=self.alpha,
            effect_size=float(coef),
            effect_size_name=name,
        )

    # --- Auto-descoberta --------------------------------------------------

    def auto_suggest(self, target: str | None = None) -> list[TestResult]:
        """Sugere e roda automaticamente os testes relevantes.

        Sem ``target``: explora pares (numérico×numérico, categórico×categórico,
        numérico×binário).

        Com ``target``: prioriza pares que envolvem o alvo.
        """
        numeric = self._numeric_columns()
        categorical = self._categorical_columns()
        binary = self._binary_columns()

        # Checa o tipo do target ANTES de removê-lo das listas
        target_is_binary = target in binary if target is not None else False
        target_is_numeric = target in numeric if target is not None else False

        # Remove o target das listas de variáveis dependentes se ele aparecer
        if target is not None:
            numeric = [c for c in numeric if c != target]
            categorical = [c for c in categorical if c != target]
            binary = [c for c in binary if c != target]

        results: list[TestResult] = []

        if target is None:
            # Exploração livre
            for i, a in enumerate(numeric):
                for b in numeric[i + 1 :]:
                    try:
                        results.append(self.test_correlation(a, b, "pearson"))
                    except ValueError:
                        pass
            for i, a in enumerate(categorical):
                for b in categorical[i + 1 :]:
                    try:
                        results.append(self.test_independence(a, b))
                    except ValueError:
                        pass
            for n in numeric:
                for g in binary:
                    try:
                        results.append(self.test_mean_difference(n, g))
                    except ValueError:
                        pass
        else:
            if target_is_binary:
                for n in numeric:
                    try:
                        results.append(self.test_mean_difference(n, target))
                    except ValueError:
                        pass
                for c in categorical:
                    if c == target:
                        continue
                    try:
                        results.append(self.test_independence(c, target))
                    except ValueError:
                        pass
            elif target_is_numeric:
                for n in numeric:
                    if n == target:
                        continue
                    for method in ("pearson", "spearman"):
                        try:
                            results.append(self.test_correlation(target, n, method))
                        except ValueError:
                            pass
                for b in binary:
                    try:
                        results.append(self.test_mean_difference(target, b))
                    except ValueError:
                        pass

        return results

    # --- Camada interpretativa -------------------------------------------

    def build_prompt(
        self,
        results: list[TestResult],
        extra_instructions: str | None = None,
    ) -> str:
        """Monta o prompt enviado ao LLM."""
        clean_extra: str | None = None
        if extra_instructions is not None:
            clean_extra = sanitize_text(
                extra_instructions,
                max_length=MAX_EXTRA_INSTRUCTIONS_CHARS,
                field_name="extra_instructions",
            )

        body = "\n\n".join(r.summary() for r in results)
        context_block = self._build_context_block(clean_extra)

        return (
            "Você é um estatístico sênior interpretando testes de hipóteses.\n\n"
            "REGRAS ESTRITAS (têm precedência sobre o contexto do usuário):\n"
            "- Use APENAS os resultados abaixo. Não invente p-valores, "
            "tamanhos de efeito ou variáveis que não estão listadas.\n"
            "- NÃO confunda rejeitar H0 com 'provar H1'. Rejeitar H0 "
            "significa que há evidência contra H0, não prova da alternativa.\n"
            "- Considere sempre o TAMANHO DE EFEITO além do p-valor. "
            "Um p-valor pequeno com efeito negligenciável não é "
            "cientificamente relevante.\n"
            "- Se um teste tem nota de advertência, mencione-a na "
            "interpretação.\n"
            "- Se um resultado é ambíguo, diga 'conclusão não conclusiva' em "
            "vez de inventar.\n"
            f"{context_block}"
            "---\n"
            f"{body}\n"
            "---\n\n"
            "Produza uma interpretação em português, em markdown, com as "
            "seções:\n\n"
            "1. **Resumo executivo** — 3 bullets principais.\n"
            "2. **Resultados por teste** — para cada teste, uma frase "
            "ligando estatística, p-valor, tamanho de efeito e conclusão.\n"
            "3. **Alertas metodológicos** — limitações, advertências, "
            "cuidados com múltiplas comparações.\n"
            "4. **Próximos passos** — o que investigar a seguir.\n\n"
            "Seja conciso e técnico. Sem introduções genéricas."
        )

    def _build_context_block(self, clean_extra: str | None) -> str:
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
        results: list[TestResult] | None = None,
        extra_instructions: str | None = None,
    ) -> str:
        """Gera narrativa via LLM. Levanta se não houver agente."""
        if self.agent is None:
            raise RuntimeError("HypothesisTester.interpret() requer um agente LLM.")
        if results is None:
            results = self.auto_suggest()
        prompt = self.build_prompt(results, extra_instructions)
        return self.agent.generate(prompt)

    def report(
        self,
        results: list[TestResult] | None = None,
        extra_instructions: str | None = None,
    ) -> str:
        """Markdown: resultados determinísticos + narrativa (se houver agente)."""
        if results is None:
            results = self.auto_suggest()

        parts = [
            "# Relatório de Testes de Hipóteses\n",
            f"**Testes executados:** {len(results)}  ",
            f"**α:** {self.alpha}\n",
        ]

        if self.agent is not None and results:
            parts.append("## Interpretação\n")
            parts.append(self.interpret(results, extra_instructions))
            parts.append("")

        parts.append("## Resultados detalhados\n")
        if not results:
            parts.append("_Nenhum teste aplicável ao dataset._")
        else:
            for r in results:
                parts.append(r.summary())
                parts.append("")

        return "\n".join(parts)
