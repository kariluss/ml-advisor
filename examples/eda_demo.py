"""Demonstração do DataAnalyzer com e sem LLM.

Uso:
    uv run python examples/eda_demo.py            # com LLM (Ollama)
    uv run python examples/eda_demo.py --no-llm   # só estatística (rápido)
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ml_advisor.core.data_analyzer import DataAnalyzer


def make_dataset(seed: int = 42, n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idade = rng.integers(18, 70, n)
    renda = 2000 + idade * 80 + rng.normal(0, 800, n)
    score = 300 + idade * 5 + rng.normal(0, 50, n)
    return pd.DataFrame(
        {
            "idade": idade,
            "renda": renda.round(2),
            "score_credito": score.round(0),
            "tempo_emprego": rng.exponential(5, n).round(1),
            "categoria": rng.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2]),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Pula a interpretação via LLM (rápido).",
    )
    args = parser.parse_args()

    df = make_dataset()
    print(f"Dataset: {df.shape[0]} linhas × {df.shape[1]} colunas\n")

    if args.no_llm:
        analyzer = DataAnalyzer(df)
        print(analyzer.report())
        return

    from ml_advisor.agents import create_agent
    from ml_advisor.utils.ollama_control import OllamaSession

    with OllamaSession(auto_close=True) as _host:
        agent = create_agent()
        print(f"Agente: {agent.name}\n")
        analyzer = DataAnalyzer(df, agent=agent)
        print(analyzer.report())


if __name__ == "__main__":
    main()
