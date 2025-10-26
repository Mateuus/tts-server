#!/bin/bash
# Script para iniciar a API

echo "=========================================="
echo "🚀 Iniciando API de Clonagem de Voz"
echo "=========================================="

# Verificar se está no ambiente virtual
if [ -z "$VIRTUAL_ENV" ]; then
    echo "📦 Ativando ambiente virtual..."
    
    # Tentar encontrar ambiente virtual
    if [ -f "../tts_env/bin/activate" ]; then
        source ../tts_env/bin/activate
    elif [ -f "../../tts_env/bin/activate" ]; then
        source ../../tts_env/bin/activate
    else
        echo "⚠️ Ambiente virtual não encontrado"
        echo "Execute manualmente:"
        echo "  source ../tts_env/bin/activate"
        echo "  python app.py"
        exit 1
    fi
fi

# Instalar dependências se necessário
if ! python -c "import fastapi" 2>/dev/null; then
    echo "📦 Instalando dependências..."
    pip install -r api/requirements.txt
fi

echo ""
echo "🚀 Iniciando servidor..."
echo ""
echo "📖 Documentação: http://localhost:8000/docs"
echo "🎤 API: http://localhost:8000"
echo ""
echo "Pressione Ctrl+C para parar"
echo ""

python app.py

