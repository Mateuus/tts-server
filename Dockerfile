# Dockerfile para Coqui TTS
FROM python:3.10-slim

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Configurar diretório de trabalho
WORKDIR /app

# Copiar requirements (se existir)
COPY requirements.txt ./

# Instalar Coqui TTS
RUN pip install --upgrade pip && \
    pip install TTS

# Criar diretório para armazenar arquivos de áudio
RUN mkdir -p /app/audio

# Expor porta para API (se necessário)
EXPOSE 5000

# Comando padrão - manter container vivo
CMD ["tail", "-f", "/dev/null"]

