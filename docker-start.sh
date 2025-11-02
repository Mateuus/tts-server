#!/bin/bash
# ==============================================================================
# Script para Iniciar API de Clonagem de Voz no Docker
# ==============================================================================

set -e

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Funções auxiliares
print_header() {
    echo -e "\n${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Verificar Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker não está instalado!"
    exit 1
fi

# Menu principal
print_header "🚀 Iniciar API de Clonagem de Voz - EoPix"

echo "Escolha uma opção:"
echo "  1) Iniciar CPU (porta 8000)"
echo "  2) Iniciar GPU (porta 8001)"
echo "  3) Iniciar ambos (CPU + GPU)"
echo "  4) Parar todos os containers"
echo "  5) Ver logs do container CPU"
echo "  6) Ver logs do container GPU"
echo "  7) Restart container CPU"
echo "  8) Restart container GPU"
echo "  9) Status dos containers"
echo "  10) Sair"
echo ""

read -p "Opção [1-10]: " choice

case $choice in
    1)
        print_header "Iniciando API (CPU)"
        docker-compose up -d tts-api-cpu
        sleep 3
        docker-compose logs -f tts-api-cpu &
        print_success "API iniciada em http://localhost:8000"
        print_info "Docs: http://localhost:8000/docs"
        print_info "Health: http://localhost:8000/health"
        print_info "Pressione Ctrl+C para parar de seguir os logs"
        ;;
    2)
        print_header "Iniciando API (GPU)"
        
        # Verificar se NVIDIA Docker está disponível
        if ! docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
            print_warning "NVIDIA Docker não está configurado ou GPU não disponível"
            print_info "Instalando NVIDIA Container Toolkit..."
            print_info "Siga: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
            exit 1
        fi
        
        docker-compose up -d tts-api-gpu
        sleep 3
        docker-compose logs -f tts-api-gpu &
        print_success "API iniciada em http://localhost:8001"
        print_info "Docs: http://localhost:8001/docs"
        print_info "Health: http://localhost:8001/health"
        print_info "Pressione Ctrl+C para parar de seguir os logs"
        ;;
    3)
        print_header "Iniciando ambos (CPU + GPU)"
        docker-compose up -d
        sleep 3
        print_success "CPU API: http://localhost:8000"
        print_success "GPU API: http://localhost:8001"
        print_info "Para ver logs: docker-compose logs -f"
        ;;
    4)
        print_header "Parando todos os containers"
        docker-compose down
        print_success "Containers parados!"
        ;;
    5)
        print_header "Logs do container CPU"
        docker-compose logs -f tts-api-cpu
        ;;
    6)
        print_header "Logs do container GPU"
        docker-compose logs -f tts-api-gpu
        ;;
    7)
        print_header "Reiniciando container CPU"
        docker-compose restart tts-api-cpu
        print_success "Container reiniciado!"
        ;;
    8)
        print_header "Reiniciando container GPU"
        docker-compose restart tts-api-gpu
        print_success "Container reiniciado!"
        ;;
    9)
        print_header "Status dos containers"
        docker-compose ps
        echo ""
        print_info "Para mais detalhes: docker stats"
        ;;
    10)
        print_info "Saindo..."
        exit 0
        ;;
    *)
        print_error "Opção inválida!"
        exit 1
        ;;
esac

print_header "✨ Operação concluída!"

