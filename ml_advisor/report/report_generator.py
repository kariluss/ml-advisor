"""Gerador de relatório consolidado.

Amara os três módulos de ``core/`` (EDA, testes de hipóteses,
recomendação de ML) em um único documento, opcionalmente com um sumário
executivo gerado por LLM.
"""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import markdown as md_lib
import pandas as pd

from ml_advisor.agents.base_agent import BaseAgent
from ml_advisor.core.data_analyzer import DataAnalyzer, DatasetSummary
from ml_advisor.core.hypothesis_tester import HypothesisTester, TestResult
from ml_advisor.core.ml_recommender import (
    DatasetProfile,
    MLRecommender,
    Recommendation,
)
from ml_advisor.settings import Settings
from ml_advisor.utils.text import sanitize_text

MAX_TITLE_CHARS = 200
MAX_USER_CONTEXT_CHARS = 2000

DEFAULT_CSS = """
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  max-width: 900px;
  margin: 2rem auto;
  padding: 0 1rem;
  line-height: 1.6;
  color: #24292e;
}
h1 { border-bottom: 2px solid #444; padding-bottom: .3rem; }
h2 { border-bottom: 1px solid #ddd; padding-bottom: .2rem; margin-top: 2.5rem; }
h3 { margin-top: 1.5rem; }
code { background: #f6f8fa; padding: 2px 5px; border-radius: 3px;
       font-family: "SF Mono", Menlo, Consolas, monospace; font-size: .9em; }
pre { background: #f6f8fa; padding: 1rem; border-radius: 6px;
      overflow-x: auto; }
pre code { background: transparent; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
th, td { border: 1px solid #d0d7de; padding: .5rem .75rem; text-align: left; }
th { background: #f6f8fa; }
blockquote { border-left: 4px solid #d0d7de; margin-left: 0;
             padding-left: 1rem; color: #57606a; }
hr { border: 0; border-top: 1px solid #d0d7de; margin: 2rem 0; }
"""


# ---------------------------------------------------------------------------
# Estruturas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReportSection:
    """Uma seção do relatório final."""

    title: str
    content: str


@dataclass
class Report:
    """Relatório consolidado, pronto para exportação."""

    title: str
    generated_at: datetime
    metadata: dict[str, str]
    sections: list[ReportSection] = field(default_factory=list)
    executive_summary: str | None = None

    def to_markdown(self) -> str:
        parts: list[str] = [f"# {self.title}\n"]

        parts.append("| Campo | Valor |")
        parts.append("|-------|-------|")
        for key, value in self.metadata.items():
            parts.append(f"| {key} | {value} |")
        parts.append("")

        if self.executive_summary:
            parts.append("## Sumário executivo\n")
            parts.append(self.executive_summary)
            parts.append("")

        for section in self.sections:
            parts.append(f"## {section.title}\n")
            parts.append(section.content)
            parts.append("")

        return "\n".join(parts)

    def to_html(self) -> str:
        md_body = self.to_markdown()
        html_body = md_lib.markdown(
            md_body,
            extensions=["tables", "fenced_code", "toc"],
        )
        return (
            "<!DOCTYPE html>\n"
            '<html lang="pt-BR">\n'
            "<head>\n"
            '<meta charset="UTF-8">\n'
            f"<title>{self.title}</title>\n"
            f"<style>{DEFAULT_CSS}</style>\n"
            "</head>\n"
            "<body>\n"
            f"{html_body}\n"
            "</body>\n"
            "</html>\n"
        )

    def save(self, path: str | Path) -> Path:
        """Salva em disco. Extensão decide o formato (.md ou .html)."""
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix in {".md", ".markdown"}:
            content = self.to_markdown()
        elif suffix in {".html", ".htm"}:
            content = self.to_html()
        else:
            raise ValueError(f"Extensão não suportada: {suffix!r}. Use .md ou .html.")
        path.write_text(content, encoding="utf-8")
        return path


# ---------------------------------------------------------------------------
# Gerador
# ---------------------------------------------------------------------------


class ReportGenerator:
    """Consolida EDA, testes de hipóteses e recomendação de ML em um relatório.

    Exemplo mínimo (determinístico)::

        gen = ReportGenerator(df, target="Survived", title="Titanic")
        report = gen.generate()
        report.save("titanic.md")

    Com LLM::

        gen = ReportGenerator(df, target="Survived", title="Titanic", agent=agent)
        report = gen.generate()   # inclui sumário executivo e narrativas
        report.save("titanic.html")
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target: str | None = None,
        title: str = "Relatório de Análise",
        agent: BaseAgent | None = None,
        settings: Settings | None = None,
        alpha: float = 0.05,
        user_context: str | None = None,
        include_eda: bool = True,
        include_hypotheses: bool = True,
        include_ml: bool = True,
        include_executive_summary: bool = True,
        verbose: bool = True,
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Esperado DataFrame, recebido {type(df).__name__}")

        self.df = df
        self.target = target
        self.agent = agent
        self.alpha = alpha
        self.include_eda = include_eda
        self.include_hypotheses = include_hypotheses
        self.include_ml = include_ml
        self.include_executive_summary = include_executive_summary
        self.verbose = verbose

        self.title = sanitize_text(title, max_length=MAX_TITLE_CHARS, field_name="title")
        self.user_context = (
            sanitize_text(
                user_context,
                max_length=MAX_USER_CONTEXT_CHARS,
                field_name="user_context",
            )
            if user_context is not None
            else None
        )

        self.settings = settings or self._extract_settings(agent)

    @staticmethod
    def _extract_settings(agent: BaseAgent | None) -> Settings | None:
        """Tenta extrair Settings do agente, se ele expuser."""
        if agent is None:
            return None
        return getattr(agent, "settings", None)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    # --- Geração ----------------------------------------------------------

    def generate(self) -> Report:
        sections: list[ReportSection] = []

        dataset_summary: DatasetSummary | None = None
        test_results: list[TestResult] | None = None
        profile: DatasetProfile | None = None
        recommendations: list[Recommendation] | None = None

        if self.include_eda:
            self._log("→ Análise exploratória...")
            dataset_summary, section = self._run_eda()
            sections.append(section)

        if self.include_hypotheses:
            self._log("→ Testes de hipóteses...")
            test_results, section = self._run_hypotheses()
            sections.append(section)

        if self.include_ml:
            self._log("→ Recomendações de ML...")
            profile, recommendations, section = self._run_ml()
            sections.append(section)

        executive: str | None = None
        if self.agent is not None and self.include_executive_summary and sections:
            self._log("→ Sumário executivo...")
            executive = self._build_executive_summary(
                dataset_summary, test_results, profile, recommendations
            )

        return Report(
            title=self.title,
            generated_at=datetime.now(),
            metadata=self._metadata(),
            sections=sections,
            executive_summary=executive,
        )

    # --- Runners internos -------------------------------------------------

    def _run_eda(self) -> tuple[DatasetSummary, ReportSection]:
        analyzer = DataAnalyzer(self.df, agent=self.agent, user_context=self.user_context)
        summary = analyzer.summarize()

        if self.agent is not None:
            content = analyzer.interpret(summary) + "\n\n"
        else:
            content = ""

        content += "### Resumo estatístico\n\n```\n" + summary.to_text() + "\n```"
        return summary, ReportSection(title="Análise exploratória", content=content)

    def _run_hypotheses(self) -> tuple[list[TestResult], ReportSection]:
        tester = HypothesisTester(
            self.df,
            agent=self.agent,
            alpha=self.alpha,
            user_context=self.user_context,
        )
        results = tester.auto_suggest(target=self.target)

        if not results:
            content = "_Nenhum teste aplicável ao dataset._"
            return results, ReportSection(title="Testes de hipóteses", content=content)

        if self.agent is not None:
            content = tester.interpret(results) + "\n\n"
        else:
            content = ""

        content += "### Resultados detalhados\n\n"
        content += "\n\n".join(r.summary() for r in results)
        return results, ReportSection(title="Testes de hipóteses", content=content)

    def _run_ml(
        self,
    ) -> tuple[DatasetProfile, list[Recommendation], ReportSection]:
        rec = MLRecommender(
            self.df,
            target=self.target,
            agent=self.agent,
            user_context=self.user_context,
        )
        profile = rec.profile()
        recommendations = rec.recommend()

        if self.agent is not None:
            content = rec.interpret(profile, recommendations) + "\n\n"
        else:
            content = ""

        content += "### Recomendações determinísticas\n\n"
        if recommendations:
            content += "\n\n".join(r.summary() for r in recommendations)
        else:
            content += "_Nenhuma recomendação aplicável._"
        return (
            profile,
            recommendations,
            ReportSection(title="Recomendações de Machine Learning", content=content),
        )

    # --- Sumário executivo ------------------------------------------------

    def _build_executive_summary(
        self,
        dataset_summary: DatasetSummary | None,
        test_results: list[TestResult] | None,
        profile: DatasetProfile | None,
        recommendations: list[Recommendation] | None,
    ) -> str:
        assert self.agent is not None
        prompt = self._build_executive_prompt(
            dataset_summary, test_results, profile, recommendations
        )
        return self.agent.generate(prompt)

    def _build_executive_prompt(
        self,
        dataset_summary: DatasetSummary | None,
        test_results: list[TestResult] | None,
        profile: DatasetProfile | None,
        recommendations: list[Recommendation] | None,
    ) -> str:
        blocks: list[str] = []

        if dataset_summary is not None:
            blocks.append(
                f"**Dimensões:** {dataset_summary.n_rows} linhas × "
                f"{dataset_summary.n_cols} colunas"
            )
            if dataset_summary.warnings:
                top = dataset_summary.warnings[:3]
                blocks.append("**Alertas de dados:** " + "; ".join(top))

        if profile is not None:
            blocks.append(f"**Tipo de problema detectado:** {profile.problem_type}")
            if profile.is_imbalanced and profile.imbalanced_ratio:
                blocks.append(f"**Desbalanceamento:** razão " f"{profile.imbalanced_ratio:.1f}:1")

        if test_results:
            sig = [r for r in test_results if r.reject_h0]
            blocks.append(f"**Testes com p < {self.alpha}:** {len(sig)} de " f"{len(test_results)}")
            for r in sig[:3]:
                blocks.append(
                    f"- {r.test_name}: p = "
                    f"{r.p_value:.3e}"
                    + (
                        f", {r.effect_size_name} = {r.effect_size:+.2f}"
                        if r.effect_size is not None
                        else ""
                    )
                )

        if recommendations:
            top = [r for r in recommendations if r.priority != "alternativa"][:4]
            blocks.append("**Recomendações principais:**")
            for r in top:
                blocks.append(f"- [{r.priority}] {r.algorithm}")

        essentials = "\n".join(blocks)

        context_block = ""
        if self.user_context:
            context_block = (
                "\n<CONTEXTO DO USUÁRIO>\n" f"{self.user_context}\n" "</CONTEXTO DO USUÁRIO>\n\n"
            )

        return (
            "Você é um cientista de dados sênior escrevendo o sumário "
            "executivo de um relatório técnico.\n\n"
            "REGRAS ESTRITAS:\n"
            "- Use APENAS as informações abaixo. Não invente métricas, "
            "nomes de colunas ou conclusões que não estejam suportadas.\n"
            "- O sumário é para um leitor **não-técnico** que precisa "
            "entender o essencial em 30 segundos.\n"
            "- Mencione o tamanho do dataset e o tipo de problema "
            "detectado no primeiro bullet.\n"
            "- No último bullet, aponte o próximo passo prático "
            "(o que fazer com o dataset).\n"
            f"{context_block}"
            "---\n"
            f"{essentials}\n"
            "---\n\n"
            "Escreva em português, em markdown, entre 4 e 6 bullets. "
            "Sem introdução, sem título — apenas os bullets."
        )

    # --- Metadados --------------------------------------------------------

    def _metadata(self) -> dict[str, str]:
        meta: dict[str, str] = {
            "Gerado em": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Dimensões": f"{self.df.shape[0]} linhas × {self.df.shape[1]} colunas",
        }
        if self.target:
            meta["Alvo"] = self.target
        meta["Alpha"] = f"{self.alpha}"

        if self.settings is not None:
            meta["Provider"] = self.settings.provider
            meta["Modelo"] = self.settings.model
            meta["Preset"] = self.settings.preset
        else:
            meta["Provider"] = "(nenhum — modo determinístico)"

        try:
            version = importlib.metadata.version("ml-advisor")
        except importlib.metadata.PackageNotFoundError:
            version = "desconhecida"
        meta["ml-advisor"] = version

        return meta
