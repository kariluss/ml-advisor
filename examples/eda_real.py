"""Testa o DataAnalyzer com um CSV real.

Uso:
    uv run python examples/eda_real.py data/raw/meu_dataset.csv
    uv run python examples/eda_real.py data/raw/meu_dataset.csv --no-llm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from ml_advisor.core.data_analyzer import DataAnalyzer


def load(path: Path) -> pd.DataFrame:
    """Carrega CSV ou Parquet por extensão."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".csv", ".txt"}:
        # sep=None + engine="python" tenta detectar separador automaticamente
        return pd.read_csv(path, sep=None, engine="python")
    raise ValueError(f"Extensão não suportada: {path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="Caminho do dataset.")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--head", type=int, default=0, help="Mostrar N primeiras linhas antes.")
    parser.add_argument(
        "--context",
        default=None,
        help="Contexto do dataset (injetado no prompt).",
    )
    parser.add_argument(
        "--instructions",
        default=None,
        help="Instruções adicionais para esta análise.",
    )
    args = parser.parse_args()

    if not args.path.exists():
        print(f"✗ Arquivo não encontrado: {args.path}", file=sys.stderr)
        sys.exit(1)

    df = load(args.path)
    print(f"Dataset carregado: {df.shape[0]} linhas × {df.shape[1]} colunas\n")

    if args.head:
        print(df.head(args.head), "\n")

    if args.no_llm:
        analyzer = DataAnalyzer(df, user_context=args.context)
        print(analyzer.report())
        return

    from ml_advisor.agents import create_agent
    from ml_advisor.utils.ollama_control import OllamaSession

    with OllamaSession(auto_close=True):
        agent = create_agent()
        print(f"Agente: {agent.name}\n")
        analyzer = DataAnalyzer(df, agent=agent, user_context=args.context)
        print(analyzer.report(extra_instructions=args.instructions))


if __name__ == "__main__":
    main()
