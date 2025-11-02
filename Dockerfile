# ==============================================================================
# Dockerfile para API de Clonagem de Voz com Coqui TTS (CPU)
# ==============================================================================

# Stage 1: Builder - Instalar dependências
FROM python:3.11-slim as builder

# Variáveis de ambiente para otimização
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    cmake \
    git \
    curl \
    libsndfile1-dev \
    libsox-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Criar diretório de trabalho
WORKDIR /app

# Copiar requirements e instalar dependências Python
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# ==============================================================================
# Stage 2: Runtime - Imagem final otimizada
# ==============================================================================
FROM python:3.11-slim

# Variáveis de ambiente
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.local/bin:$PATH" \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Instalar apenas dependências de runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libgomp1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Criar usuário não-root para segurança
RUN useradd -m -u 1000 -s /bin/bash appuser && \
    mkdir -p /app/audio/uploads /app/audio/outputs && \
    chown -R appuser:appuser /app

# Copiar dependências Python do builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Configurar diretório de trabalho
WORKDIR /app

# Copiar código da aplicação
COPY --chown=appuser:appuser ./api ./api
COPY --chown=appuser:appuser ./audio ./audio

# Criar diretórios necessários
RUN mkdir -p /app/api/audio/uploads /app/api/audio/outputs && \
    chown -R appuser:appuser /app

# Mudar para usuário não-root
USER appuser

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expor porta da API
EXPOSE 8000

# Comando de inicialização
WORKDIR /app/api
CMD ["python", "app.py"]
