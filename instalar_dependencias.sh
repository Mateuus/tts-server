#!/bin/bash
# Script para instalar dependências do TTS Server
# Detecta automaticamente se há GPU disponível e instala PyTorch adequadamente

set -e

echo "🚀 Instalando dependências do TTS Server..."
echo ""

# Verificar se está em ambiente virtual
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  AVISO: Ambiente virtual não detectado!"
    echo "   Execute: python3.11 -m venv tts_env && source tts_env/bin/activate"
    read -p "   Deseja continuar mesmo assim? (s/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Ss]$ ]]; then
        exit 1
    fi
fi

# Atualizar pip
echo "📦 Atualizando pip..."
pip install --upgrade pip

# Verificar se há GPU NVIDIA disponível
HAS_GPU=false
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        HAS_GPU=true
        echo "✅ GPU NVIDIA detectada!"
        CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
        echo "   Driver: $CUDA_VERSION"
    fi
fi

# Instalar PyTorch baseado na disponibilidade de GPU
echo ""
echo "📦 Instalando PyTorch..."

if [ "$HAS_GPU" = true ]; then
    echo "   Instalando versão com suporte CUDA..."
    pip install torch==2.3.1 torchaudio==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
else
    echo "   Instalando versão CPU (sem CUDA)..."
    pip install torch==2.3.1 torchaudio==2.3.1
fi

echo "✅ PyTorch instalado!"
echo ""

# Instalar outras dependências
echo "📦 Instalando outras dependências..."
pip install -r requirements.txt

echo ""
echo "✅ Todas as dependências foram instaladas com sucesso!"
echo ""
echo "📝 Próximos passos:"
echo "   1. Configure o arquivo .env: ./criar_env.sh"
echo "   2. Inicie a API: ./iniciar_api.sh"
echo ""

