"""Testes para o ReportGenerator."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_advisor.agents.base_agent import BaseAgent
from ml_advisor.report import Report, ReportGenerator, ReportSection
from ml_advisor.settings import Settings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df_sintetico() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 400
    return pd.DataFrame(
        {
            "idade": rng.integers(18, 70, n),
            "renda": rng.normal(5000, 1500, n),
            "alvo": rng.choice([0, 1], n),
        }
    )


class _FakeAgent(BaseAgent):
    def __init__(self, response: str = "conteúdo narrativo") -> None:
        self._response = response
        self.calls: list[str] = []
        self.settings = Settings(
            provider="ollama",
            preset="balanced",
            model="fake:model",
            timeout=30.0,
            temperature=0.2,
            max_tokens=256,
        )

    @property
    def name(self) -> str:
        return "fake:test"

    def is_available(self) -> bool:
        return True

    def generate(self, prompt, *, system=None, temperature=None, max_tokens=None):
        self.calls.append(prompt)
        return self._response


# ---------------------------------------------------------------------------
# Estruturas
# ---------------------------------------------------------------------------

def test_report_section_dataclass():
    s = ReportSection(title="X", content="Y")
    assert s.title == "X"
    assert s.content == "Y"


def test_report_to_markdown_minimo():
    report = Report(
        title="Teste",
        generated_at=pd.Timestamp("2026-01-01").to_pydatetime(),
        metadata={"Chave": "valor"},
    )
    md = report.to_markdown()
    assert "# Teste" in md
    assert "| Chave | valor |" in md


def test_report_to_html_valido():
    report = Report(
        title="Teste",
        generated_at=pd.Timestamp("2026-01-01").to_pydatetime(),
        metadata={},
    )
    html = report.to_html()
    assert html.startswith("<!DOCTYPE html>")
    assert "<title>Teste</title>" in html
    assert "</html>" in html


def test_report_save_md(tmp_path: Path):
    report = Report(
        title="X",
        generated_at=pd.Timestamp("2026-01-01").to_pydatetime(),
        metadata={},
    )
    path = report.save(tmp_path / "out.md")
    assert path.exists()
    assert path.read_text().startswith("# X")


def test_report_save_html(tmp_path: Path):
    report = Report(
        title="X",
        generated_at=pd.Timestamp("2026-01-01").to_pydatetime(),
        metadata={},
    )
    path = report.save(tmp_path / "out.html")
    assert path.exists()
    assert "<!DOCTYPE html>" in path.read_text()


def test_report_save_extensao_invalida(tmp_path: Path):
    report = Report(
        title="X",
        generated_at=pd.Timestamp("2026-01-01").to_pydatetime(),
        metadata={},
    )
    with pytest.raises(ValueError, match="Extensão"):
        report.save(tmp_path / "out.pdf")


# ---------------------------------------------------------------------------
# Construtor
# ---------------------------------------------------------------------------

def test_requer_dataframe():
    with pytest.raises(TypeError, match="DataFrame"):
        ReportGenerator([1, 2, 3])


def test_title_vazio_levanta(df_sintetico):
    with pytest.raises(ValueError, match="vazio"):
        ReportGenerator(df_sintetico, title="   ")


def test_title_muito_longo(df_sintetico):
    with pytest.raises(ValueError, match="máximo"):
        ReportGenerator(df_sintetico, title="x" * 201)


# ---------------------------------------------------------------------------
# Geração determinística
# ---------------------------------------------------------------------------

def test_generate_gera_3_secoes(df_sintetico):
    gen = ReportGenerator(df_sintetico, target="alvo", verbose=False)
    report = gen.generate()
    titles = [s.title for s in report.sections]
    assert "Análise exploratória" in titles
    assert "Testes de hipóteses" in titles
    assert "Recomendações de Machine Learning" in titles


def test_generate_sem_eda(df_sintetico):
    gen = ReportGenerator(
        df_sintetico, target="alvo", include_eda=False, verbose=False
    )
    report = gen.generate()
    titles = [s.title for s in report.sections]
    assert "Análise exploratória" not in titles


def test_generate_sem_testes(df_sintetico):
    gen = ReportGenerator(
        df_sintetico, target="alvo", include_hypotheses=False, verbose=False
    )
    report = gen.generate()
    titles = [s.title for s in report.sections]
    assert "Testes de hipóteses" not in titles


def test_generate_sem_ml(df_sintetico):
    gen = ReportGenerator(
        df_sintetico, target="alvo", include_ml=False, verbose=False
    )
    report = gen.generate()
    titles = [s.title for s in report.sections]
    assert "Recomendações de Machine Learning" not in titles


def test_generate_metadata_sem_agente(df_sintetico):
    gen = ReportGenerator(df_sintetico, target="alvo", verbose=False)
    report = gen.generate()
    assert report.metadata["Provider"] == "(nenhum — modo determinístico)"
    assert report.metadata["Alvo"] == "alvo"


def test_generate_metadata_com_agente(df_sintetico):
    gen = ReportGenerator(
        df_sintetico, target="alvo", agent=_FakeAgent(), verbose=False
    )
    report = gen.generate()
    assert report.metadata["Provider"] == "ollama"
    assert report.metadata["Modelo"] == "fake:model"


# ---------------------------------------------------------------------------
# Com agente
# ---------------------------------------------------------------------------

def test_generate_com_agente_inclui_sumario(df_sintetico):
    agent = _FakeAgent("sumário executivo fake")
    gen = ReportGenerator(
        df_sintetico, target="alvo", agent=agent, verbose=False
    )
    report = gen.generate()
    assert report.executive_summary == "sumário executivo fake"
    # Chamadas: 1 EDA + 1 Testes + 1 ML + 1 Sumário = 4 mínimo
    assert len(agent.calls) >= 4


def test_generate_sem_sumario_executivo(df_sintetico):
    agent = _FakeAgent()
    gen = ReportGenerator(
        df_sintetico,
        target="alvo",
        agent=agent,
        include_executive_summary=False,
        verbose=False,
    )
    report = gen.generate()
    assert report.executive_summary is None


def test_sumario_no_markdown(df_sintetico):
    agent = _FakeAgent("SUMARIO_UNICO")
    gen = ReportGenerator(
        df_sintetico, target="alvo", agent=agent, verbose=False
    )
    report = gen.generate()
    md = report.to_markdown()
    assert "SUMARIO_UNICO" in md
    assert "## Sumário executivo" in md


# ---------------------------------------------------------------------------
# Integração com Titanic (dados reais)
# ---------------------------------------------------------------------------

def test_titanic_report_gera_3_secoes(titanic_df):
    gen = ReportGenerator(
        titanic_df,
        target="Survived",
        title="Titanic",
        verbose=False,
    )
    report = gen.generate()
    assert len(report.sections) == 3


def test_titanic_eda_detecta_cabin_alto_missing(titanic_df):
    gen = ReportGenerator(
        titanic_df,
        target="Survived",
        include_hypotheses=False,
        include_ml=False,
        verbose=False,
    )
    report = gen.generate()
    eda_section = report.sections[0]
    assert "Cabin" in eda_section.content
    # Na amostra de 50, ~78% de Cabin é nulo
    assert "ausentes" in eda_section.content.lower()


def test_titanic_recomenda_classificacao_binaria(titanic_df):
    gen = ReportGenerator(
        titanic_df,
        target="Survived",
        include_eda=False,
        include_hypotheses=False,
        verbose=False,
    )
    report = gen.generate()
    ml_section = report.sections[0]
    # Survived é binário -> Logistic Regression deve ser baseline
    assert "Regressão Logística" in ml_section.content
    assert "Random Forest" in ml_section.content


def test_titanic_report_save_md_e_html(titanic_df, tmp_path: Path):
    gen = ReportGenerator(
        titanic_df, target="Survived", title="Titanic", verbose=False
    )
    report = gen.generate()

    md_path = report.save(tmp_path / "titanic.md")
    assert md_path.exists()
    assert "Titanic" in md_path.read_text()

    html_path = report.save(tmp_path / "titanic.html")
    assert html_path.exists()
    html = html_path.read_text()
    assert "<table>" in html  # metadata vira tabela no HTML
    assert "<style>" in html


def test_titanic_html_tem_estrutura(titanic_df):
    gen = ReportGenerator(
        titanic_df, target="Survived", title="Titanic", verbose=False
    )
    html = gen.generate().to_html()
    # CSS embutido
    assert "font-family" in html
    # Hierarquia renderizada (markdown adiciona id= para âncoras)
    assert re.search(r"<h1[\s>]", html)
    assert re.search(r"<h2[\s>]", html)
