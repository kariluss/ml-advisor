"""Demonstra as três formas de configurar o OllamaAgent."""

from ml_advisor.agents.ollama_agent import OllamaAgent
from ml_advisor.settings import load_settings
from ml_advisor.utils.ollama_control import OllamaSession

with OllamaSession(auto_close=True):
    # Forma 1: padrão (lê env/toml/preset balanced)
    agent = OllamaAgent()
    print(f"[default] {agent.name}")
    print("   ", agent.generate("Em uma frase, o que é PCA?"))

    # Forma 2: override de modelo
    # (descomente se tiver o phi3 baixado)
    # agent = OllamaAgent(model="phi3:mini-4k-instruct-q4_K_M")
    # print(f"[phi3] {agent.name}")

    # Forma 3: settings explícito
    settings = load_settings(temperature=0.0, max_tokens=128)
    agent = OllamaAgent(settings=settings)
    print(f"[settings] {agent.name} (temp={settings.temperature})")
