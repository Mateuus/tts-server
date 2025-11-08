#!/bin/bash

# Script para fechar processos na porta 8080
# Uso: ./closeports.sh

PORT=8000

echo "🔍 Procurando processos na porta $PORT..."

# Verificar se há processos na porta
PIDS=$(lsof -ti:$PORT 2>/dev/null)

if [ -z "$PIDS" ]; then
    echo "✅ Nenhum processo encontrado na porta $PORT"
    exit 0
fi

echo "📋 Processos encontrados na porta $PORT:"
lsof -i:$PORT 2>/dev/null | grep LISTEN

echo ""
echo "🛑 Encerrando processos..."

# Fechar cada processo
for PID in $PIDS; do
    echo "   ⚠️  Encerrando processo PID: $PID"
    kill -9 $PID 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "   ✅ Processo $PID encerrado com sucesso"
    else
        echo "   ❌ Erro ao encerrar processo $PID"
    fi
done

# Verificar se ainda há processos
sleep 1
REMAINING=$(lsof -ti:$PORT 2>/dev/null)

if [ -z "$REMAINING" ]; then
    echo ""
    echo "✅ Porta $PORT liberada com sucesso!"
else
    echo ""
    echo "⚠️  Ainda há processos na porta $PORT:"
    lsof -i:$PORT 2>/dev/null
    echo ""
    echo "Tente executar novamente ou verifique manualmente com: lsof -i:$PORT"
    exit 1
fi

