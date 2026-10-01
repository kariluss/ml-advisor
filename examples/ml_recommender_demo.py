"""Demonstração do MLRecommender com e sem LLM.

Uso:
    uv run python examples/ml_recommender_demo.py                  # com LLM
    uv run python examples/ml_recommender_demo.py --no-llm         # só estatística
    uv run python examples/ml_recommender_demo.py --target alvo    # foco no alvo
    uv run python examples/ml_recommender_demo.py --no-llm --no-target  # clustering
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ml_advisor.core.ml_recommender import MLRecommender


def make_dataset(seed: int = 42, n: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idade = rng.integers(18, 70, n)
    renda = rng.normal(5000, 1500, n)
    # alvo binário correlacionado com idade e renda (efeito sutil)
    logit = (idade - 44) * 0.05 + (renda - 5000) / 5000
    prob = 1 / (1 + np.exp(-logit))
    alvo = (rng.uniform(0, 1, n) < prob).astype(int)

    return pd.DataFrame(
        {
            "idade": idade,
            "renda": renda.round(2),
            "regiao": rng.choice(["Norte", "Sul", "Leste", "Oeste"], n),
            "alvo": alvo,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--target", default="alvo")
    parser.add_argument(
        "--no-target",
        action="store_true",
        help="Força modo clustering (sem alvo).",
    )
    args = parser.parse_args()

    df = make_dataset()
    target = None if args.no_target else args.target

    print(f"Dataset: {df.shape[0]} linhas × {df.shape[1]} colunas")
    print(f"Alvo: {target!r}\n")

    if args.no_llm:
        rec = MLRecommender(df, target=target)
        print(rec.report())
        return

    from ml_advisor.agents import create_agent
    from ml_advisor.utils.ollama_control import OllamaSession

    with OllamaSession(auto_close=True):
        agent = create_agent()
        print(f"Agente: {agent.name}\n")
        rec = MLRecommender(
            df,
            target=target,
            agent=agent,
            user_context=(
                "Dataset de clientes. Interessa prever 'alvo' (0/1) "
                "a partir de idade, renda e região."
            ),
        )
        print(rec.report())


if __name__ == "__main__":
    main()
