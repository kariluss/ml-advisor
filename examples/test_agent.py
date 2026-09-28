from ml_advisor.agents.ollama_agent import OllamaAgent
from ml_advisor.utils.ollama_control import OllamaSession

MODEL = "mistral:7b-instruct-q4_K_M"

with OllamaSession(auto_close=True):
    agent = OllamaAgent(model=MODEL)
    print("Disponível?", agent.is_available())
    print("Nome:", agent.name)
    print("Resposta:", agent.generate("Em uma frase, o que é PCA?"))