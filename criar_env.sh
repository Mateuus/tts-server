#!/bin/bash
# Script para criar arquivo .env a partir do template

echo "🔧 Criando arquivo .env..."

# Verificar se .env já existe
if [ -f .env ]; then
    echo "⚠️  Arquivo .env já existe!"
    read -p "Deseja sobrescrever? (s/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Ss]$ ]]; then
        echo "❌ Operação cancelada."
        exit 1
    fi
fi

# Criar arquivo .env
cat > .env << 'EOF'
# ============================================
# SERVIDOR
# ============================================
# Host e porta do servidor
HOST=0.0.0.0
PORT=8000

# Modo de desenvolvimento (ativa reload automático)
DEBUG=true

# ============================================
# DIRETÓRIOS
# ============================================
# Diretório para uploads de áudio
UPLOAD_DIR=audio/uploads

# Diretório para áudios gerados
OUTPUT_DIR=audio/outputs

# Diretório padrão para referências de voz
VOICE_REF_DIR=audio

# ============================================
# MODELOS TTS
# ============================================
# Modelo TTS padrão (multilíngue com clonagem)
# Opções: 
#   - tts_models/multilingual/multi-dataset/xtts_v2 (recomendado, clonagem de voz)
#   - tts_models/pt/cv/vits (português, sem clonagem)
TTS_MODEL=tts_models/multilingual/multi-dataset/xtts_v2

# Idioma padrão
DEFAULT_LANGUAGE=pt

# Velocidade padrão de fala (0.5 a 2.0)
DEFAULT_SPEED=0.95

# Arquivo de voz de referência padrão (caminho relativo ou absoluto)
DEFAULT_VOICE_REF=audio/minha_voz.mp3

# ============================================
# WHISPER (Transcrição)
# ============================================
# Modelo Whisper para transcrição
# Opções: tiny, base, small, medium, large
# Menor = mais rápido, menos preciso
# Maior = mais lento, mais preciso
WHISPER_MODEL=base

# ============================================
# SEGURANÇA E FILTROS
# ============================================
# Palavras banidas (separadas por vírgula)
# Essas palavras serão substituídas por # no texto e por beeps no áudio
BANNED_WORDS=clonagem,Open Voice

# ============================================
# RECURSOS E PERFORMANCE
# ============================================
# Usar GPU se disponível (true/false)
# Requer CUDA e PyTorch com suporte CUDA instalado
USE_GPU=false

# Device CUDA (0, 1, 2, etc.)
# Use apenas se tiver múltiplas GPUs
CUDA_VISIBLE_DEVICES=0

# ============================================
# MODELOS DE IA (Opcional)
# ============================================
# Caminho para modelos GPT-2 customizados
# Usado no endpoint /generateAI
AI_MODELS_DIR=../models

# ============================================
# PYTHON / RUNTIME
# ============================================
# Desabilitar buffer de saída Python (útil para logs em tempo real)
PYTHONUNBUFFERED=1

# ============================================
# LOGS E DEBUG
# ============================================
# Nível de log (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO

# ============================================
# LIMITES E CONFIGURAÇÕES
# ============================================
# Tamanho máximo de arquivo de upload (em MB)
MAX_UPLOAD_SIZE_MB=50

# Tempo limite para geração de áudio (em segundos)
AUDIO_GENERATION_TIMEOUT=300
EOF

echo "✅ Arquivo .env criado com sucesso!"
echo ""
echo "📝 Próximos passos:"
echo "   1. Edite o arquivo .env conforme necessário"
echo "   2. Configure o arquivo de voz de referência em DEFAULT_VOICE_REF"
echo "   3. Execute ./iniciar_api.sh para iniciar o servidor"
echo ""

