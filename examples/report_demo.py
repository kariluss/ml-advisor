"""Demonstração do ReportGenerator com e sem LLM.

Uso:
    uv run python examples/report_demo.py                          # sintético + LLM
    uv run python examples/report_demo.py --no-llm                 # só determinístico
    uv run python examples/report_demo.py --dataset data/raw/train.csv
    uv run python examples/report_demo.py --dataset data/raw/train.csv \
        --target Survived --context "Dataset do Titanic."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ml_advisor.report import ReportGenerator


def make_synthetic(seed: int = 42, n: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idade = rng.integers(18, 70, n)
    renda = 2000 + idade * 80 + rng.normal(0, 800, n)
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


def load_csv(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, sep=None, engine="python")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--target", default=None)
    parser.add_argument("--context", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.dataset is None:
        df = make_synthetic()
        target = args.target or "alvo"
        default_title = "Demo Sintético"
    else:
        if not args.dataset.exists():
            print(f"✗ Arquivo não encontrado: {args.dataset}", file=sys.stderr)
            sys.exit(1)
        df = load_csv(args.dataset)
        target = args.target
        default_title = args.dataset.stem

    print(f"Dataset: {df.shape[0]} linhas × {df.shape[1]} colunas")
    print(f"Alvo: {target!r}\n")

    if args.no_llm:
        gen = ReportGenerator(df, target=target, title=default_title, verbose=True)
        report = gen.generate()
    else:
        from ml_advisor.agents import create_agent
        from ml_advisor.utils.ollama_control import OllamaSession

        # IMPORTANTE: todo o trabalho pesado (generate) precisa acontecer
        # DENTRO do `with`. Se sair do bloco, o Ollama é encerrado.
        with OllamaSession(auto_close=True):
            agent = create_agent()
            print(f"Agente: {agent.name}\n")
            gen = ReportGenerator(
                df,
                target=target,
                title=default_title,
                agent=agent,
                user_context=args.context,
                verbose=True,
            )
            report = gen.generate()

    if args.out:
        path = report.save(args.out)
        print(f"\n✓ Salvo em {path}")
    else:
        print()
        print(report.to_markdown())


if __name__ == "__main__":
    main()
