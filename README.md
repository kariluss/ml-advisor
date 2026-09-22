# ML Advisor

Biblioteca Python modular para análise exploratória de dados, testes de
hipóteses estatísticas e geração de relatórios com recomendações de técnicas
de Machine Learning — usando **LLMs locais** via Ollama para garantir
gratuidade, privacidade e reprodutibilidade.

## Stack

- Python 3.12 (gerenciado via [uv](https://docs.astral.sh/uv/))
- pandas, numpy, scipy, scikit-learn
- matplotlib, seaborn
- [Ollama](https://ollama.com/) com `mistral:7b-instruct-q4_K_M`
- Jupyter via VS Code

## Pré-requisitos

- Windows 11 com WSL2 (Ubuntu)
- Ollama instalado no Windows, escutando em `0.0.0.0:11434`
- `uv` instalado dentro do WSL2
- Git configurado com chave SSH no GitHub

## Setup

```bash
# 1. Clonar
git clone git@github.com:<seu-usuario>/ml-advisor.git
cd ml-advisor

# 2. Sincronizar dependências
uv sync

# 3. Verificar conexão com Ollama
uv run python examples/test_ollama.py