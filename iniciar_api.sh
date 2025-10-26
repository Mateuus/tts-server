#!/bin/bash
# Iniciar API de Clonagem de Voz

echo "🚀 Iniciando API de Clonagem de Voz..."
echo ""

# Ativar ambiente virtual
echo "📦 Ativando ambiente virtual..."
source tts_env/bin/activate

# Ir para diretório da API
cd api

# Instalar dependências se necessário
echo "📦 Verificando dependências..."
pip install -q -r requirements.txt

echo ""
echo "🚀 Iniciando servidor..."
echo "📖 Documentação: http://localhost:8000/docs"
echo ""
echo "Pressione Ctrl+C para parar"
echo ""

# Iniciar
python app.py

