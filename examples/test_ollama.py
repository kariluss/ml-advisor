"""Script de sanidade: valida conexão com o Ollama e uma geração simples."""

from __future__ import annotations

import sys
import httpx

from ml_advisor.utils.ollama_control import OllamaSession

MODEL = "mistral:7b-instruct-q4_K_M"


def list_models(host: str) -> list[str]:
    response = httpx.get(f"{host}/api/tags", timeout=10, trust_env=False)
    response.raise_for_status()
    return [m["name"] for m in response.json().get("models", [])]


def generate(host: str, model: str, prompt: str) -> str:
    response = httpx.post(
        f"{host}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=300,
        trust_env=False,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


def main() -> None:
    # auto_close=True: se o script iniciar o Ollama, ele encerra no fim.
    # Se já estava rodando, deixa intacto.
    with OllamaSession(auto_close=True) as host:
        models = list_models(host)
        print(f"✓ Modelos disponíveis: {models}")

        if MODEL not in models:
            print(f"✗ Modelo {MODEL} não encontrado.")
            print(f"  Rode no PowerShell: ollama pull {MODEL}")
            sys.exit(1)

        print(f"→ Gerando com {MODEL}...")
        text = generate(
            host,
            MODEL,
            "Responda em uma única frase: o que é análise exploratória de dados?",
        )
        print(f"✓ Resposta:\n  {text}")

    print("\n✅ Tudo funcionando.")


if __name__ == "__main__":
    main()
