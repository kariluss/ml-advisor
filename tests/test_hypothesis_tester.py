"""Testes para o HypothesisTester."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_advisor.agents.base_agent import BaseAgent
from ml_advisor.core.hypothesis_tester import (
    HypothesisTester,
    TestResult,
    _cohens_d,
    _cramers_v,
    _describe_effect,
    _format_pvalue,
    _rank_biserial,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df_normal() -> pd.DataFrame:
    """Dataset com duas normais independentes e um grupo binário."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame(
        {
            "x": rng.normal(0, 1, n),
            "y": rng.normal(0, 1, n),
            "grupo": rng.choice(["A", "B"], n),
        }
    )


@pytest.fixture
def df_associated() -> pd.DataFrame:
    """Dataset com forte dependência entre variáveis."""
    rng = np.random.default_rng(7)
    n = 300
    x = rng.normal(0, 1, n)
    y = 2 * x + rng.normal(0, 0.3, n)  # quase determinístico
    grupo = np.where(x > 0, "alto", "baixo")
    return pd.DataFrame({"x": x, "y": y, "grupo": grupo})


@pytest.fixture
def df_categorical() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 400
    a = rng.choice(["A", "B", "C"], n)
    b = np.where(a == "A", rng.choice(["X", "Y"], n), rng.choice(["X", "Y"], n))
    return pd.DataFrame({"a": a, "b": b})


class _FakeAgent(BaseAgent):
    def __init__(self, response: str = "análise fake") -> None:
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
# Helpers internos
# ---------------------------------------------------------------------------

def test_format_pvalue_muito_pequeno():
    assert _format_pvalue(1e-20) == "< 1e-15"


def test_format_pvalue_cientifico():
    assert "e" in _format_pvalue(1e-5)


def test_format_pvalue_normal():
    assert _format_pvalue(0.03) == "0.0300"


def test_describe_effect_correlacao():
    assert _describe_effect(0.05, "r (Pearson)") == "negligenciável"
    assert _describe_effect(0.2, "r (Pearson)") == "fraco"
    assert _describe_effect(0.4, "r (Pearson)") == "moderado"
    assert _describe_effect(0.6, "r (Pearson)") == "forte"
    assert _describe_effect(0.9, "r (Pearson)") == "muito forte"


def test_describe_effect_cohens_d():
    assert _describe_effect(0.1, "Cohen's d") == "negligenciável"
    assert _describe_effect(0.3, "Cohen's d") == "pequeno"
    assert _describe_effect(0.6, "Cohen's d") == "médio"
    assert _describe_effect(1.2, "Cohen's d") == "grande"


def test_cohens_d_identicas():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert _cohens_d(a, a) == pytest.approx(0.0)


def test_cohens_d_diferentes():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    assert _cohens_d(a, b) < 0  # a média de a é menor


def test_cramers_v_quadrado_2x2():
    # chi2=10, n=100, min(1,1)=1 -> sqrt(10/100) ≈ 0.316
    assert _cramers_v(10.0, 100, (2, 2)) == pytest.approx(0.3162, abs=1e-3)


def test_cramers_v_degenerado():
    assert _cramers_v(0.0, 0, (2, 2)) == 0.0


def test_rank_biserial_empate():
    # U = n1*n2/2 -> r = 0
    assert _rank_biserial(50.0, 10, 10) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Validação do construtor
# ---------------------------------------------------------------------------

def test_requer_dataframe():
    with pytest.raises(TypeError, match="DataFrame"):
        HypothesisTester([1, 2, 3])


def test_alpha_invalido():
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError, match="alpha"):
        HypothesisTester(df, alpha=1.5)


def test_user_context_vazio_levanta():
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError, match="vazio"):
        HypothesisTester(df, user_context="   ")


# ---------------------------------------------------------------------------
# test_normality
# ---------------------------------------------------------------------------

def test_normality_normal(df_normal):
    tester = HypothesisTester(df_normal)
    result = tester.test_normality("x")
    # x é normal -> não rejeita H0
    assert result.reject_h0 is False
    assert result.variables == ("x",)
    assert result.n > 0


def test_normality_nao_normal():
    df = pd.DataFrame({"x": np.arange(100) ** 2})
    tester = HypothesisTester(df)
    result = tester.test_normality("x")
    assert result.reject_h0 is True


def test_normality_n_insuficiente():
    df = pd.DataFrame({"x": [1.0, 2.0]})
    tester = HypothesisTester(df)
    with pytest.raises(ValueError, match="insuficiente"):
        tester.test_normality("x")


def test_normality_amostra_grande_limita():
    df = pd.DataFrame({"x": np.random.default_rng(0).normal(0, 1, 8000)})
    tester = HypothesisTester(df)
    result = tester.test_normality("x")
    assert result.n == 5000
    assert any("limitada" in n for n in result.notes)


# ---------------------------------------------------------------------------
# test_mean_difference
# ---------------------------------------------------------------------------

def test_mean_difference_sem_diferenca(df_normal):
    tester = HypothesisTester(df_normal)
    result = tester.test_mean_difference("x", "grupo")
    assert result.variables == ("x", "grupo")
    assert result.effect_size_name == "Cohen's d"
    # x e grupo foram gerados independentes -> não deve rejeitar H0
    assert result.reject_h0 is False


def test_mean_difference_com_diferenca(df_associated):
    tester = HypothesisTester(df_associated)
    result = tester.test_mean_difference("x", "grupo")
    # x define o grupo -> diferença enorme -> rejeita H0
    assert result.reject_h0 is True


def test_mean_difference_grupo_invalido(df_normal):
    tester = HypothesisTester(df_normal)
    with pytest.raises(ValueError, match="colunas diferentes"):
        tester.test_mean_difference("x", "x")


# ---------------------------------------------------------------------------
# test_independence
# ---------------------------------------------------------------------------

def test_independence_executa(df_categorical):
    tester = HypothesisTester(df_categorical)
    result = tester.test_independence("a", "b")
    assert result.effect_size_name == "Cramér's V"
    assert result.n == len(df_categorical)


def test_independence_tabela_degenerada():
    df = pd.DataFrame({"a": ["x", "x", "x"], "b": ["y", "y", "y"]})
    tester = HypothesisTester(df)
    with pytest.raises(ValueError, match="degenerada"):
        tester.test_independence("a", "b")


# ---------------------------------------------------------------------------
# test_correlation
# ---------------------------------------------------------------------------

def test_correlation_pearson_forte(df_associated):
    tester = HypothesisTester(df_associated)
    result = tester.test_correlation("x", "y", "pearson")
    assert result.effect_size is not None
    assert result.effect_size > 0.9
    assert result.reject_h0 is True


def test_correlation_spearman(df_associated):
    tester = HypothesisTester(df_associated)
    result = tester.test_correlation("x", "y", "spearman")
    assert "Spearman" in result.test_name
    assert result.effect_size is not None
    assert result.effect_size > 0.9


def test_correlation_method_invalido(df_associated):
    tester = HypothesisTester(df_associated)
    with pytest.raises(ValueError, match="method inválido"):
        tester.test_correlation("x", "y", "kendall")


def test_correlation_n_insuficiente():
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    tester = HypothesisTester(df)
    with pytest.raises(ValueError, match="insuficiente"):
        tester.test_correlation("x", "y")


# ---------------------------------------------------------------------------
# auto_suggest
# ---------------------------------------------------------------------------

def test_auto_suggest_sem_target(df_normal):
    tester = HypothesisTester(df_normal)
    results = tester.auto_suggest()
    assert len(results) > 0
    # Deve incluir correlação x-y e comparações com grupo
    test_names = {r.test_name for r in results}
    assert any("Pearson" in n for n in test_names)


def test_auto_suggest_com_target_binario(df_associated):
    df = df_associated.copy()
    df["alvo"] = (df["x"] > 0).astype(int)
    tester = HypothesisTester(df)
    results = tester.auto_suggest(target="alvo")
    assert len(results) > 0
    # Deve gerar testes comparando x, y por alvo
    assert all("alvo" in r.variables for r in results)


def test_auto_suggest_com_target_numerico(df_associated):
    tester = HypothesisTester(df_associated)
    results = tester.auto_suggest(target="y")
    assert len(results) > 0
    assert all("y" in r.variables for r in results)


def test_auto_suggest_dataframe_vazio():
    df = pd.DataFrame({"a": [1.0]})
    tester = HypothesisTester(df)
    results = tester.auto_suggest()
    assert results == []


# ---------------------------------------------------------------------------
# build_prompt / interpret / report
# ---------------------------------------------------------------------------

def test_build_prompt_contem_regras(df_normal):
    tester = HypothesisTester(df_normal)
    results = tester.auto_suggest()
    prompt = tester.build_prompt(results)
    assert "Use APENAS" in prompt
    assert "tamanho de efeito" in prompt
    assert "Resumo executivo" in prompt


def test_build_prompt_com_user_context(df_normal):
    tester = HypothesisTester(df_normal, user_context="Dados clínicos.")
    results = tester.auto_suggest()
    prompt = tester.build_prompt(results)
    assert "Dados clínicos." in prompt
    assert "<CONTEXTO DO USUÁRIO>" in prompt


def test_build_prompt_com_extra(df_normal):
    tester = HypothesisTester(df_normal)
    results = tester.auto_suggest()
    prompt = tester.build_prompt(results, extra_instructions="Foque em x.")
    assert "Foque em x." in prompt


def test_interpret_sem_agente_levanta(df_normal):
    tester = HypothesisTester(df_normal)
    with pytest.raises(RuntimeError, match="agente LLM"):
        tester.interpret()


def test_interpret_com_agente(df_normal):
    agent = _FakeAgent("resposta")
    tester = HypothesisTester(df_normal, agent=agent)
    out = tester.interpret()
    assert out == "resposta"
    assert agent.last_prompt is not None
    assert "estatístico sênior" in agent.last_prompt


def test_report_sem_agente(df_normal):
    tester = HypothesisTester(df_normal)
    report = tester.report()
    assert "# Relatório de Testes de Hipóteses" in report
    assert "## Resultados detalhados" in report
    assert "## Interpretação" not in report


def test_report_com_agente(df_normal):
    tester = HypothesisTester(df_normal, agent=_FakeAgent("narrativa"))
    report = tester.report()
    assert "## Interpretação" in report
    assert "narrativa" in report


def test_report_sem_resultados():
    df = pd.DataFrame({"a": [1]})
    tester = HypothesisTester(df)
    report = tester.report()
    assert "Nenhum teste aplicável" in report


# ---------------------------------------------------------------------------
# TestResult
# ---------------------------------------------------------------------------

def test_testresult_decision_rejeitar():
    r = TestResult(
        test_name="X",
        variables=("a", "b"),
        n=100,
        statistic=5.0,
        p_value=0.001,
        alpha=0.05,
    )
    assert r.reject_h0 is True
    assert r.decision == "rejeitar H0"


def test_testresult_decision_nao_rejeitar():
    r = TestResult(
        test_name="X",
        variables=("a",),
        n=100,
        statistic=0.5,
        p_value=0.4,
        alpha=0.05,
    )
    assert r.reject_h0 is False
    assert r.decision == "não rejeitar H0"
