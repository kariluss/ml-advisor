"""Testes para o MLRecommender."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_advisor.agents.base_agent import BaseAgent
from ml_advisor.core.ml_recommender import (
    DatasetProfile,
    MLRecommender,
    Recommendation,
    _class_balance,
    _classify_problem,
    _id_like_columns,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df_binario() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 500
    return pd.DataFrame(
        {
            "idade": rng.integers(18, 70, n),
            "renda": rng.normal(5000, 1500, n),
            "alvo": rng.choice([0, 1], n),
        }
    )


@pytest.fixture
def df_multiclasse() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 300
    return pd.DataFrame(
        {
            "x": rng.normal(0, 1, n),
            "y": rng.normal(0, 1, n),
            "classe": rng.choice(["A", "B", "C"], n),
        }
    )


@pytest.fixture
def df_regressao() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 400
    return pd.DataFrame(
        {
            "x": rng.normal(0, 1, n),
            "y": rng.normal(0, 1, n),
            "alvo": rng.normal(50, 10, n),
        }
    )


@pytest.fixture
def df_imbalanced() -> pd.DataFrame:
    rng = np.random.default_rng(13)
    n = 1000
    return pd.DataFrame(
        {
            "x": rng.normal(0, 1, n),
            "alvo": rng.choice([0, 1], n, p=[0.95, 0.05]),
        }
    )


@pytest.fixture
def df_sem_target() -> pd.DataFrame:
    rng = np.random.default_rng(17)
    return pd.DataFrame(
        {"a": rng.normal(0, 1, 100), "b": rng.normal(0, 1, 100)}
    )


class _FakeAgent(BaseAgent):
    def __init__(self, response: str = "sugestão fake") -> None:
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

def test_classify_problem_binario(df_binario):
    assert _classify_problem(df_binario, "alvo") == "classification_binary"


def test_classify_problem_multiclasse(df_multiclasse):
    assert _classify_problem(df_multiclasse, "classe") == "classification_multiclass"


def test_classify_problem_regressao(df_regressao):
    assert _classify_problem(df_regressao, "alvo") == "regression"


def test_classify_problem_clustering():
    df = pd.DataFrame({"x": [1, 2, 3]})
    assert _classify_problem(df, None) == "clustering"


def test_classify_problem_alvo_constante():
    df = pd.DataFrame({"x": [1, 2, 3], "alvo": [5, 5, 5]})
    with pytest.raises(ValueError, match="único"):
        _classify_problem(df, "alvo")


def test_id_like_columns():
    df = pd.DataFrame({"id": [1, 2, 3, 4], "x": [1, 1, 2, 2]})
    ids = _id_like_columns(df)
    assert ids == ("id",)


def test_id_like_columns_nenhuma():
    df = pd.DataFrame({"x": [1, 1, 2, 2], "y": [1, 2, 3, 4]})
    ids = _id_like_columns(df)
    assert "x" not in ids


def test_class_balance():
    s = pd.Series([0] * 90 + [1] * 10)
    counts, is_imbalanced, ratio = _class_balance(s)
    assert counts == {"0": 90, "1": 10}
    assert is_imbalanced is True
    assert ratio == pytest.approx(9.0)


def test_class_balance_equilibrado():
    s = pd.Series([0] * 50 + [1] * 50)
    _, is_imbalanced, _ = _class_balance(s)
    assert is_imbalanced is False


# ---------------------------------------------------------------------------
# Construtor
# ---------------------------------------------------------------------------

def test_requer_dataframe():
    with pytest.raises(TypeError, match="DataFrame"):
        MLRecommender([1, 2, 3])


def test_target_inexistente(df_binario):
    with pytest.raises(ValueError, match="não encontrada"):
        MLRecommender(df_binario, target="nao_existe")


def test_user_context_vazio_levanta(df_binario):
    with pytest.raises(ValueError, match="vazio"):
        MLRecommender(df_binario, target="alvo", user_context="   ")


# ---------------------------------------------------------------------------
# Perfil
# ---------------------------------------------------------------------------

def test_perfil_binario(df_binario):
    p = MLRecommender(df_binario, target="alvo").profile()
    assert p.n_samples == 500
    assert p.n_features == 2  # idade + renda (alvo excluído)
    assert p.problem_type == "classification_binary"
    assert p.target == "alvo"
    assert p.class_counts is not None


def test_perfil_regressao(df_regressao):
    p = MLRecommender(df_regressao, target="alvo").profile()
    assert p.problem_type == "regression"
    assert p.class_counts is None


def test_perfil_clustering(df_sem_target):
    p = MLRecommender(df_sem_target).profile()
    assert p.problem_type == "clustering"
    assert p.target is None


def test_perfil_detecta_imbalance(df_imbalanced):
    p = MLRecommender(df_imbalanced, target="alvo").profile()
    assert p.is_imbalanced is True
    assert p.imbalanced_ratio is not None
    assert p.imbalanced_ratio > 5


def test_perfil_alta_dimensionalidade():
    rng = np.random.default_rng(0)
    n = 100
    data = {f"f{i}": rng.normal(0, 1, n) for i in range(60)}
    data["alvo"] = rng.choice([0, 1], n)
    p = MLRecommender(pd.DataFrame(data), target="alvo").profile()
    assert p.has_high_dimensionality is True


def test_perfil_amostra_pequena():
    df = pd.DataFrame(
        {"x": np.arange(50), "alvo": np.random.default_rng(0).choice([0, 1], 50)}
    )
    p = MLRecommender(df, target="alvo").profile()
    assert p.has_small_sample is True


# ---------------------------------------------------------------------------
# Recomendações
# ---------------------------------------------------------------------------

def test_recomendacoes_binario(df_binario):
    recs = MLRecommender(df_binario, target="alvo").recommend()
    algorithms = [r.algorithm for r in recs]
    assert any("Logística" in a for a in algorithms)
    assert any("Random Forest" in a for a in algorithms)


def test_recomendacoes_binario_imbalanced(df_imbalanced):
    recs = MLRecommender(df_imbalanced, target="alvo").recommend()
    algorithms = [r.algorithm for r in recs]
    assert any("Pesos de classe" in a for a in algorithms)
    assert any("Gradient Boosting" in a for a in algorithms)  # n=1000


def test_recomendacoes_multiclasse(df_multiclasse):
    recs = MLRecommender(df_multiclasse, target="classe").recommend()
    algorithms = [r.algorithm for r in recs]
    assert any("multinomial" in a for a in algorithms)
    assert any("XGBoost" in a or "LightGBM" in a for a in algorithms)


def test_recomendacoes_regressao(df_regressao):
    recs = MLRecommender(df_regressao, target="alvo").recommend()
    algorithms = [r.algorithm for r in recs]
    assert any("Linear" in a for a in algorithms)
    assert any("Random Forest" in a for a in algorithms)


def test_recomendacoes_clustering(df_sem_target):
    recs = MLRecommender(df_sem_target).recommend()
    algorithms = [r.algorithm for r in recs]
    assert any("K-Means" in a for a in algorithms)
    assert any("DBSCAN" in a for a in algorithms)


def test_recomendacoes_tem_priority_valida(df_binario):
    recs = MLRecommender(df_binario, target="alvo").recommend()
    valid = {"baseline", "recomendado", "alternativa"}
    for r in recs:
        assert r.priority in valid


def test_recomendacoes_tem_rationale(df_binario):
    recs = MLRecommender(df_binario, target="alvo").recommend()
    for r in recs:
        assert r.rationale
        assert isinstance(r.caveats, tuple)


# ---------------------------------------------------------------------------
# build_prompt / interpret / report
# ---------------------------------------------------------------------------

def test_build_prompt_contem_regras(df_binario):
    rec = MLRecommender(df_binario, target="alvo")
    prompt = rec.build_prompt()
    assert "Use APENAS" in prompt
    assert "Diagnóstico" in prompt
    assert "classification_binary" in prompt


def test_build_prompt_com_user_context(df_binario):
    rec = MLRecommender(
        df_binario, target="alvo", user_context="Base de crédito."
    )
    prompt = rec.build_prompt()
    assert "Base de crédito." in prompt
    assert "<CONTEXTO DO USUÁRIO>" in prompt


def test_build_prompt_com_extra(df_binario):
    rec = MLRecommender(df_binario, target="alvo")
    prompt = rec.build_prompt(extra_instructions="Priorize interpretabilidade.")
    assert "Priorize interpretabilidade." in prompt


def test_interpret_sem_agente_levanta(df_binario):
    rec = MLRecommender(df_binario, target="alvo")
    with pytest.raises(RuntimeError, match="agente LLM"):
        rec.interpret()


def test_interpret_com_agente(df_binario):
    agent = _FakeAgent("recomendação narrativa")
    rec = MLRecommender(df_binario, target="alvo", agent=agent)
    out = rec.interpret()
    assert out == "recomendação narrativa"
    assert agent.last_prompt is not None
    assert "Machine Learning" in agent.last_prompt


def test_report_sem_agente(df_binario):
    rec = MLRecommender(df_binario, target="alvo")
    report = rec.report()
    assert "# Recomendações de Machine Learning" in report
    assert "## Perfil do dataset" in report
    assert "## Recomendações determinísticas" in report
    assert "## Análise interpretativa" not in report


def test_report_com_agente(df_binario):
    rec = MLRecommender(df_binario, target="alvo", agent=_FakeAgent("narrativa"))
    report = rec.report()
    assert "## Análise interpretativa" in report
    assert "narrativa" in report


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

def test_dataset_profile_summary():
    p = DatasetProfile(
        n_samples=100,
        n_features=5,
        n_numeric=3,
        n_categorical=2,
        pct_missing=0.0,
        problem_type="classification_binary",
        target="alvo",
    )
    s = p.summary()
    assert "Amostras: 100" in s
    assert "classification_binary" in s


def test_recommendation_summary():
    r = Recommendation(
        family="tree_ensemble",
        algorithm="Random Forest",
        priority="recomendado",
        rationale="Robusto.",
    )
    s = r.summary()
    assert "Random Forest" in s
    assert "recomendado" in s
    assert "Robusto." in s
