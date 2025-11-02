#!/bin/bash
# ==============================================================================
# Script de Build para Docker - API de Clonagem de Voz
# ==============================================================================

set -e  # Parar em caso de erro

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Verificar se Docker está instalado
if ! command -v docker &> /dev/null; then
    print_error "Docker não está instalado!"
    exit 1
fi

# Verificar se Docker Compose está instalado
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    print_error "Docker Compose não está instalado!"
    exit 1
fi

# Função para build
build_image() {
    local TYPE=$1
    local DOCKERFILE=$2
    local TAG=$3
    
    print_header "Building ${TYPE} Image"
    
    docker build \
        -f ${DOCKERFILE} \
        -t ${TAG} \
        --progress=plain \
        .
    
    if [ $? -eq 0 ]; then
        print_success "${TYPE} image built successfully: ${TAG}"
    else
        print_error "Failed to build ${TYPE} image"
        exit 1
    fi
}

# Menu de opções
print_header "🐳 Docker Build - API de Clonagem de Voz EoPix"

echo "Escolha uma opção:"
echo "  1) Build CPU only (mais rápido, sem GPU)"
echo "  2) Build GPU (requer NVIDIA Docker)"
echo "  3) Build ambas (CPU + GPU)"
echo "  4) Build com Docker Compose (CPU)"
echo "  5) Build com Docker Compose (GPU)"
echo "  6) Limpar imagens antigas"
echo "  7) Sair"
echo ""

read -p "Opção [1-7]: " choice

case $choice in
    1)
        build_image "CPU" "Dockerfile" "eopix/tts-api:latest-cpu"
        print_success "Build CPU concluído!"
        print_info "Para iniciar: docker run -p 8000:8000 eopix/tts-api:latest-cpu"
        ;;
    2)
        build_image "GPU" "Dockerfile.gpu" "eopix/tts-api:latest-gpu"
        print_success "Build GPU concluído!"
        print_info "Para iniciar: docker run --gpus all -p 8000:8000 eopix/tts-api:latest-gpu"
        ;;
    3)
        build_image "CPU" "Dockerfile" "eopix/tts-api:latest-cpu"
        build_image "GPU" "Dockerfile.gpu" "eopix/tts-api:latest-gpu"
        print_success "Builds concluídos!"
        ;;
    4)
        print_header "Building com Docker Compose (CPU)"
        docker-compose build tts-api-cpu
        print_success "Build concluído!"
        print_info "Para iniciar: docker-compose up -d tts-api-cpu"
        ;;
    5)
        print_header "Building com Docker Compose (GPU)"
        docker-compose build tts-api-gpu
        print_success "Build concluído!"
        print_info "Para iniciar: docker-compose up -d tts-api-gpu"
        ;;
    6)
        print_header "Limpando imagens antigas"
        docker image prune -f
        docker system prune -f
        print_success "Limpeza concluída!"
        ;;
    7)
        print_info "Saindo..."
        exit 0
        ;;
    *)
        print_error "Opção inválida!"
        exit 1
        ;;
esac

print_header "🎉 Processo concluído!"
print_info "Documentação da API: http://localhost:8000/docs"
print_info "Health check: http://localhost:8000/health"
