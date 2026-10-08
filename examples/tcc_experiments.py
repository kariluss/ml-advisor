"""Experimentos reprodutíveis para o TCC.

Roda os três módulos do `ml_advisor` sobre um dataset e salva os
resultados em `resultados/<dataset>/`. Determinístico por padrão;
use `--with-llm` para enriquecer com narrativa (mais lento).

Uso:
    uv run python examples/tcc_experiments.py
    uv run python examples/tcc_experiments.py --dataset data/raw/train.csv \\
        --target Survived --name titanic
    uv run python examples/tcc_experiments.py --with-llm
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from ml_advisor.core.data_analyzer import DataAnalyzer
from ml_advisor.core.hypothesis_tester import HypothesisTester
from ml_advisor.core.ml_recommender import MLRecommender
from ml_advisor.report import ReportGenerator


RESULTS_DIR = Path("resultados")


def load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, sep=None, engine="python")


def slugify(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text).lower()


def run_deterministic(
    df: pd.DataFrame,
    target: str | None,
    output_dir: Path,
    user_context: str | None = None,
) -> dict:
    """Roda os três módulos sem LLM. Retorna métricas para metadata.json."""
    metrics: dict = {}

    # --- EDA ---
    t0 = time.monotonic()
    analyzer = DataAnalyzer(df, user_context=user_context)
    summary = analyzer.summarize()
    metrics["eda_seconds"] = round(time.monotonic() - t0, 3)
    metrics["n_warnings"] = len(summary.warnings)

    (output_dir / "01_eda.md").write_text(
        "# Análise Exploratória\n\n```\n" + summary.to_text() + "\n```",
        encoding="utf-8",
    )

    # --- Testes de hipóteses ---
    t0 = time.monotonic()
    tester = HypothesisTester(df, user_context=user_context)
    test_results = tester.auto_suggest(target=target)
    metrics["hypotheses_seconds"] = round(time.monotonic() - t0, 3)
    metrics["n_tests"] = len(test_results)
    metrics["n_tests_significant"] = sum(1 for r in test_results if r.reject_h0)

    hyp_lines = ["# Testes de Hipóteses\n"]
    hyp_lines.append(f"**Total:** {len(test_results)}  ")
    hyp_lines.append(f"**Significativos (α = 0.05):** {metrics['n_tests_significant']}\n")
    for r in test_results:
        hyp_lines.append(r.summary())
        hyp_lines.append("")
    (output_dir / "02_hypotheses.md").write_text(
        "\n".join(hyp_lines), encoding="utf-8"
    )

    # --- ML ---
    t0 = time.monotonic()
    recommender = MLRecommender(df, target=target, user_context=user_context)
    profile = recommender.profile()
    recommendations = recommender.recommend()
    metrics["ml_seconds"] = round(time.monotonic() - t0, 3)
    metrics["problem_type"] = profile.problem_type
    metrics["n_recommendations"] = len(recommendations)

    ml_lines = ["# Recomendações de Machine Learning\n"]
    ml_lines.append(profile.summary())
    ml_lines.append("")
    for r in recommendations:
        ml_lines.append(r.summary())
        ml_lines.append("")
    (output_dir / "03_ml.md").write_text("\n".join(ml_lines), encoding="utf-8")

    return metrics


def run_with_llm(
    df: pd.DataFrame,
    target: str | None,
    output_dir: Path,
    title: str,
    user_context: str | None,
) -> dict:
    """Roda com LLM. Retorna métricas adicionais."""
    from ml_advisor.agents import create_agent
    from ml_advisor.utils.ollama_control import OllamaSession

    with OllamaSession(auto_close=True):
        agent = create_agent()
        t0 = time.monotonic()
        gen = ReportGenerator(
            df,
            target=target,
            title=title,
            agent=agent,
            user_context=user_context,
            verbose=True,
        )
        report = gen.generate()
        elapsed = round(time.monotonic() - t0, 3)

        report.save(output_dir / "04_report.md")
        report.save(output_dir / "04_report.html")

        return {
            "llm_seconds": elapsed,
            "llm_model": agent.name,
            "llm_provider": agent.settings.provider,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/raw/train.csv"),
        help="Caminho do dataset. Default: Titanic (train.csv).",
    )
    parser.add_argument("--target", default="Survived")
    parser.add_argument("--name", default=None, help="Nome do experimento.")
    parser.add_argument("--context", default=None, help="Contexto do domínio.")
    parser.add_argument("--with-llm", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Diretório de saída. Default: resultados/<name>/",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"✗ Dataset não encontrado: {args.dataset}", file=sys.stderr)
        sys.exit(1)

    df = load_dataset(args.dataset)
    name = args.name or args.dataset.stem
    output_dir = args.output_dir or (RESULTS_DIR / slugify(name))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {args.dataset}")
    print(f"Dimensões: {df.shape[0]} linhas × {df.shape[1]} colunas")
    print(f"Alvo: {args.target!r}")
    print(f"Saída: {output_dir.resolve()}\n")

    metadata: dict = {
        "dataset": str(args.dataset),
        "name": name,
        "target": args.target,
        "n_samples": int(df.shape[0]),
        "n_features": int(df.shape[1]),
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "with_llm": args.with_llm,
    }

    print("→ Rodando módulos determinísticos...")
    det_metrics = run_deterministic(
        df, args.target, output_dir, user_context=args.context
    )
    metadata.update(det_metrics)
    print(f"  ✓ concluído em "
          f"{det_metrics['eda_seconds'] + det_metrics['hypotheses_seconds'] + det_metrics['ml_seconds']:.1f}s")

    if args.with_llm:
        print("\n→ Rodando com LLM (pode demorar)...")
        llm_metrics = run_with_llm(
            df, args.target, output_dir, title=name, user_context=args.context
        )
        metadata.update(llm_metrics)
        print(f"  ✓ concluído em {llm_metrics['llm_seconds']:.1f}s")

    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n✓ Experimentos salvos em {output_dir.resolve()}")
    print("  Arquivos:")
    for f in sorted(output_dir.iterdir()):
        size = f.stat().st_size
        print(f"    {f.name} ({size} bytes)")


if __name__ == "__main__":
    main()