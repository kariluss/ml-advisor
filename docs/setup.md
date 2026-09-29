# Guia de Setup

Documento de reprodução do ambiente de desenvolvimento do **ML Advisor**.
Escrito para ser seguido em uma máquina limpa, do zero.

## Visão geral

O projeto roda em **WSL2 (Ubuntu)** sobre Windows, com o **Ollama rodando
no lado Windows** por questões de aceleração e simplicidade. A comunicação
entre os dois lados é feita via HTTP na porta `11434`.

## Requisitos

**Hardware mínimo:**
- 16 GB de RAM (para rodar o modelo `mistral:7b-instruct-q4_K_M` em CPU)
- ~20 GB de disco livre (Python + dependências + modelos Ollama)
- CPU x86_64 com suporte a AVX2

**Software:**
- Windows 11 (ou 10 com WSL2 habilitado)
- WSL2 com Ubuntu 22.04+ instalado
- Git
- VS Code com extensão WSL da Microsoft

## 1. WSL2 + Ubuntu

```powershell
# No PowerShell como Administrador
wsl --install -d Ubuntu
```

Reinicie a máquina. Defina usuário/senha do Linux quando solicitado.

Verifique:

```bash
# Dentro do Ubuntu
wsl --status
wsl --list --verbose
```

A distro deve aparecer com `VERSION 2`.

## 2. Python 3.12 e uv (dentro do WSL2)

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y curl build-essential python3.12 python3.12-venv python3.12-dev

curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv --version
```

## 3. Ollama no Windows

Instale via winget (ou baixe em <https://ollama.com/download>):

```powershell
winget install Ollama.Ollama
```

### 3.1. Configurar variáveis de ambiente

**No Windows**, em *Configurações → Sistema → Variáveis de Ambiente*,
adicione em **Variáveis do Sistema**:

- `OLLAMA_HOST` = `0.0.0.0`
- `OLLAMA_ORIGINS` = `*`

Feche o Ollama pela bandeja do sistema e reabra (ele só lê as variáveis na
inicialização). Confirme com:

```powershell
netstat -ano | findstr :11434
```

Deve mostrar `0.0.0.0:11434`. Se mostrar `127.0.0.1:11434`, o Ollama ignorou
a variável — desinstale, redefina as variáveis e reinstale.

### 3.2. Liberar no firewall

```powershell
# PowerShell como Administrador
New-NetFirewallRule `
  -DisplayName "Ollama WSL2" `
  -Direction Inbound `
  -LocalPort 11434 `
  -Protocol TCP `
  -Action Allow `
  -Profile Any
```

### 3.3. Baixar o modelo

```powershell
ollama pull mistral:7b-instruct-q4_K_M
```

## 4. Acesso do WSL2 ao Ollama do Windows

O IP do Windows **muda a cada reinício do WSL2**. Para não depender disso,
adicione ao `~/.bashrc` do WSL2:

```bash
export WIN_IP=$(ip route show default | awk '{print $3}')
export OLLAMA_HOST="http://$WIN_IP:11434"
```

Recarregue e teste:

```bash
source ~/.bashrc
curl --max-time 10 $OLLAMA_HOST/api/tags
```

Deve retornar `{"models":[...]}`.

## 5. Clone e sincronização do projeto

**Dentro do WSL2**, nunca em `/mnt/c/`:

```bash
mkdir -p ~/projects
cd ~/projects
git clone git@github.com:kariluss/ml-advisor.git
cd ml-advisor
uv sync
```

## 6. Kernel Jupyter

```bash
uv run python -m ipykernel install --user \
  --name ml_advisor \
  --display-name "ML Advisor (uv)"
```

## 7. Verificação

```bash
uv run python examples/test_ollama.py
```

Deve imprimir `✅ Tudo funcionando.` ao final.

## Desenvolvimento no VS Code

**Sempre** abra o VS Code conectado ao WSL:

```bash
# Dentro do WSL, na pasta do projeto
code .
```

Alternativa: `Ctrl+Shift+P` → `WSL: Connect to WSL` → abrir a pasta
`/home/<usuário>/projects/ml-advisor`.

**Nunca** abra pelo Explorer do Windows via `\\wsl.localhost\...` — isso
faz o VS Code rodar do lado Windows, perdendo acesso ao kernel do WSL,
ao `uv`, ao `pytest` e à descoberta automática do Ollama.

## Troubleshooting

| Sintoma | Causa | Solução |
|---------|-------|---------|
| `curl` do WSL2 dá timeout no IP do Windows | Firewall bloqueando | Aplicar regra do item 3.2 |
| `netstat` mostra `127.0.0.1:11434` | `OLLAMA_HOST` não aplicado | Reinstalar Ollama com variáveis já definidas |
| `git` reclama de "dubious ownership" | Comando rodado no Git Bash do Windows | Usar terminal do WSL2 |
| `uv sync` com warning de hardlink | Projeto em `/mnt/c/` | Mover para `~/projects/` |
| VS Code só lista Pythons do Windows | Aberto fora do WSL | `WSL: Connect to WSL` + reabrir pasta |
| `OllamaAgent` retorna vazio | Modelo não baixado | `ollama pull mistral:7b-instruct-q4_K_M` |
| Geração muito lenta | Modelo 7B em CPU | Normal (30–90s na 1ª chamada). Usar `phi3:mini` para testes |