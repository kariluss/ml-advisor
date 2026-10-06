"""Testes para o DataAnalyzer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_advisor.agents.base_agent import BaseAgent
from ml_advisor.core.data_analyzer import (
    DataAnalyzer,
    _classify,
    _compute_warnings,
    _summarize_column,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df_simples() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "idade": rng.integers(18, 70, 100),
            "renda": rng.normal(5000, 1500, 100).round(2),
            "categoria": rng.choice(["A", "B", "C"], 100),
        }
    )


@pytest.fixture
def df_com_nulos() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1, 2, np.nan, 4, 5],
            "b": [np.nan, np.nan, np.nan, 1, 2],
            "c": ["x", "y", None, "x", "x"],
        }
    )


@pytest.fixture
def df_constante() -> pd.DataFrame:
    return pd.DataFrame({"k": [1, 1, 1, 1], "v": [1, 2, 3, 4]})


class _FakeAgent(BaseAgent):
    """Agente de teste que devolve uma resposta fixa."""

    def __init__(self, response: str = "análise de teste") -> None:
        self._response = response
        self.last_prompt: str | None = None

    @property
    def name(self) -> str:
        return "fake:test"

    def is_available(self) -> bool:
        return True

    def generate(self, prompt, *, system=None, temperature=None, max_tokens=None):
        self.last_prompt = prompt
        return self._response


# ---------------------------------------------------------------------------
# Helpers de baixo nível
# ---------------------------------------------------------------------------

def test_classify_numeric():
    assert _classify(pd.Series([1, 2, 3])) == "numeric"


def test_classify_categorical():
    assert _classify(pd.Series(["a", "b", "a"])) == "categorical"


def test_classify_bool_as_categorical():
    assert _classify(pd.Series([True, False, True])) == "categorical"


def test_summarize_column_numeric():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name="x")
    summary = _summarize_column(s)
    assert summary.kind == "numeric"
    assert summary.mean == pytest.approx(3.0)
    assert summary.median == pytest.approx(3.0)
    assert summary.min == 1.0
    assert summary.max == 5.0
    assert summary.n_missing == 0
    assert summary.top_values is None


def test_summarize_column_categorical():
    s = pd.Series(["a", "b", "a", "a", "c"], name="cat")
    summary = _summarize_column(s)
    assert summary.kind == "categorical"
    assert summary.n_unique == 3
    assert summary.top_values is not None
    assert summary.top_values[0] == ("a", 3)


def test_summarize_column_all_missing():
    s = pd.Series([np.nan, np.nan], name="empty")
    summary = _summarize_column(s)
    assert summary.n_missing == 2
    assert summary.pct_missing == 100.0
    assert summary.mean is None


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------

def test_warnings_small_sample():
    df = pd.DataFrame({"x": range(5)})
    columns = tuple(_summarize_column(df[c]) for c in df.columns)
    warnings = _compute_warnings(df, columns)
    assert any("Amostra pequena" in w for w in warnings)


def test_warnings_high_missing():
    df = pd.DataFrame({"x": [1, np.nan, np.nan, np.nan, np.nan]})
    columns = tuple(_summarize_column(df[c]) for c in df.columns)
    warnings = _compute_warnings(df, columns)
    assert any("ausentes" in w for w in warnings)


def test_warnings_constant_column():
    df = pd.DataFrame({"k": [1, 1, 1, 1, 1]})
    columns = tuple(_summarize_column(df[c]) for c in df.columns)
    warnings = _compute_warnings(df, columns)
    assert any("constante" in w for w in warnings)


def test_warnings_high_cardinality():
    df = pd.DataFrame({"id": [f"u{i}" for i in range(50)]})
    columns = tuple(_summarize_column(df[c]) for c in df.columns)
    warnings = _compute_warnings(df, columns)
    assert any("cardinalidade" in w for w in warnings)


# ---------------------------------------------------------------------------
# DataAnalyzer
# ---------------------------------------------------------------------------

def test_analyzer_requires_dataframe():
    with pytest.raises(TypeError, match="DataFrame"):
        DataAnalyzer([1, 2, 3])


def test_summarize_basico(df_simples):
    analyzer = DataAnalyzer(df_simples)
    s = analyzer.summarize()
    assert s.n_rows == 100
    assert s.n_cols == 3
    assert len(s.columns) == 3
    names = {c.name for c in s.columns}
    assert names == {"idade", "renda", "categoria"}


def test_summarize_tem_correlacao(df_simples):
    analyzer = DataAnalyzer(df_simples)
    s = analyzer.summarize()
    assert s.correlation_matrix is not None
    # idade, renda (categoria é string)
    assert set(s.correlation_matrix.columns) == {"idade", "renda"}


def test_summarize_sem_correlacao():
    df = pd.DataFrame({"a": ["x", "y", "z"], "b": ["p", "q", "r"]})
    s = DataAnalyzer(df).summarize()
    assert s.correlation_matrix is None


def test_to_text_contem_secoes(df_simples):
    text = DataAnalyzer(df_simples).summarize().to_text()
    assert "## Colunas" in text
    assert "### idade" in text
    assert "## Correlações mais fortes" in text


def test_build_prompt_inclui_regras(df_simples):
    prompt = DataAnalyzer(df_simples).build_prompt()
    assert "Use APENAS as informações" in prompt
    assert "Qualidade dos dados" in prompt
    assert "Recomendações de ML" in prompt


def test_interpret_sem_agente_levanta(df_simples):
    with pytest.raises(RuntimeError, match="agente LLM"):
        DataAnalyzer(df_simples).interpret()


def test_interpret_com_agente(df_simples):
    agent = _FakeAgent("resposta do modelo")
    analyzer = DataAnalyzer(df_simples, agent=agent)
    result = analyzer.interpret()
    assert result == "resposta do modelo"
    assert agent.last_prompt is not None
    assert "cientista de dados" in agent.last_prompt


def test_report_sem_agente(df_simples):
    report = DataAnalyzer(df_simples).report()
    assert "# Relatório de Análise Exploratória" in report
    assert "## Resumo estatístico" in report
    assert "## Análise interpretativa" not in report


def test_report_com_agente(df_simples):
    analyzer = DataAnalyzer(df_simples, agent=_FakeAgent("narrativa"))
    report = analyzer.report()
    assert "## Análise interpretativa" in report
    assert "narrativa" in report


def test_summarize_int_low_cardinality_as_categorical():
    """Inteiro com poucos valores únicos vira categórica (ex.: Survived 0/1)."""
    s = pd.Series([0, 1, 0, 1, 1, 0], name="target")
    summary = _summarize_column(s)
    assert summary.kind == "categorical"
    assert summary.top_values is not None


def test_summarize_int_high_cardinality_stays_numeric():
    s = pd.Series(range(100), name="id")
    summary = _summarize_column(s)
    assert summary.kind == "numeric"


def test_user_context_injetado_no_prompt(df_simples):
    analyzer = DataAnalyzer(df_simples, user_context="Base de crédito.")
    prompt = analyzer.build_prompt()
    assert "<CONTEXTO DO USUÁRIO>" in prompt
    assert "Base de crédito." in prompt
    assert "</CONTEXTO DO USUÁRIO>" in prompt


def test_extra_instructions_injetadas(df_simples):
    analyzer = DataAnalyzer(df_simples)
    prompt = analyzer.build_prompt(extra_instructions="Foque em renda.")
    assert "Instruções adicionais" in prompt
    assert "Foque em renda." in prompt


def test_user_context_e_extra_juntos(df_simples):
    analyzer = DataAnalyzer(df_simples, user_context="Base de crédito.")
    prompt = analyzer.build_prompt(extra_instructions="Foque em renda.")
    assert "Base de crédito." in prompt
    assert "Foque em renda." in prompt
    # Contexto vem antes das instruções adicionais
    assert prompt.index("Base de crédito.") < prompt.index("Foque em renda.")


def test_sem_contexto_bloco_ausente(df_simples):
    prompt = DataAnalyzer(df_simples).build_prompt()
    assert "<CONTEXTO DO USUÁRIO>" not in prompt


def test_user_context_vazio_levanta(df_simples):
    with pytest.raises(ValueError, match="vazio"):
        DataAnalyzer(df_simples, user_context="   ")


def test_user_context_excede_tamanho(df_simples):
    with pytest.raises(ValueError, match="máximo"):
        DataAnalyzer(df_simples, user_context="x" * 2001)


def test_extra_instructions_excede_tamanho(df_simples):
    analyzer = DataAnalyzer(df_simples)
    with pytest.raises(ValueError, match="máximo"):
        analyzer.build_prompt(extra_instructions="x" * 1001)


def test_interpret_repassa_extra(df_simples):
    agent = _FakeAgent("ok")
    analyzer = DataAnalyzer(df_simples, agent=agent)
    analyzer.interpret(extra_instructions="Só analise renda.")
    assert "Só analise renda." in agent.last_prompt
