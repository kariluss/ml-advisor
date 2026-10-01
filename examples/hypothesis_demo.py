"""Demonstração do HypothesisTester com e sem LLM.

Uso:
    uv run python examples/hypothesis_demo.py                # com LLM
    uv run python examples/hypothesis_demo.py --no-llm       # só estatística
    uv run python examples/hypothesis_demo.py --target grupo # foco em um alvo
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ml_advisor.core.hypothesis_tester import HypothesisTester


def make_dataset(seed: int = 42, n: int = 300) -> pd.DataFrame:
    """Cria um dataset com associações reais para demonstrar os testes."""
    rng = np.random.default_rng(seed)

    idade = rng.integers(18, 70, n)
    # renda depende da idade com ruído
    renda = 1000 + idade * 100 + rng.normal(0, 800, n)
    # sobrevive depende de grupo (efeito real)
    grupo = rng.choice(["controle", "tratamento"], n)
    desfecho_base = rng.normal(50, 10, n)
    desfecho = np.where(grupo == "tratamento", desfecho_base + 5, desfecho_base)
    # categoria independente (não deve ter efeito)
    regiao = rng.choice(["Norte", "Sul", "Leste"], n)

    return pd.DataFrame(
        {
            "idade": idade,
            "renda": renda.round(2),
            "grupo": grupo,
            "desfecho": desfecho.round(2),
            "regiao": regiao,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Pula a interpretação via LLM (rápido).",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Coluna alvo: foca os testes nela.",
    )
    args = parser.parse_args()

    df = make_dataset()
    print(f"Dataset: {df.shape[0]} linhas × {df.shape[1]} colunas")
    print(f"Colunas: {list(df.columns)}\n")

    if args.no_llm:
        tester = HypothesisTester(df)
        results = tester.auto_suggest(target=args.target)
        print(f"Testes executados: {len(results)}\n")
        for r in results:
            print(r.summary())
            print()
        return

    from ml_advisor.agents import create_agent
    from ml_advisor.utils.ollama_control import OllamaSession

    with OllamaSession(auto_close=True):
        agent = create_agent()
        print(f"Agente: {agent.name}\n")
        tester = HypothesisTester(
            df,
            agent=agent,
            user_context=(
                "Estudo com grupo controle e tratamento. "
                "'desfecho' é a variável resposta contínua."
            ),
        )
        results = tester.auto_suggest(target=args.target)
        print(tester.report(results))


if __name__ == "__main__":
    main()
