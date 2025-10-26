#!/bin/bash

# Script para construir imagens Docker do Coqui TTS

echo "🔨 Construindo imagens Docker para Coqui TTS..."

# Verificar se Docker está instalado
if ! command -v docker &> /dev/null; then
    echo "❌ Docker não está instalado. Por favor, instale o Docker primeiro."
    exit 1
fi

# Menu de seleção
echo ""
echo "Escolha uma opção:"
echo "1) CPU Only"
echo "2) GPU (CUDA)"
echo "3) Ambos"
read -p "Opção [1-3]: " option

case $option in
    1)
        echo "🐢 Construindo imagem CPU..."
        docker-compose build tts-cpu
        echo "✅ Imagem CPU construída!"
        ;;
    2)
        echo "🚀 Construindo imagem GPU..."
        if ! docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
            echo "❌ GPU não detectada ou NVIDIA Docker runtime não configurado."
            echo "   Tente instalar nvidia-container-toolkit ou use a opção 1 (CPU)"
            exit 1
        fi
        docker-compose build tts-gpu
        echo "✅ Imagem GPU construída!"
        ;;
    3)
        echo "🐢 Construindo imagem CPU..."
        docker-compose build tts-cpu
        echo "✅ Imagem CPU construída!"
        
        echo "🚀 Construindo imagem GPU..."
        if ! docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
            echo "⚠️ GPU não detectada. Pulando construção de GPU."
        else
            docker-compose build tts-gpu
            echo "✅ Imagem GPU construída!"
        fi
        ;;
    *)
        echo "❌ Opção inválida"
        exit 1
        ;;
esac

echo ""
echo "✅ Construção concluída!"
echo ""
echo "Para iniciar os containers:"
echo "  CPU: docker-compose up -d tts-cpu"
echo "  GPU: docker-compose up -d tts-gpu"

