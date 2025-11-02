# 🐳 Docker - API de Clonagem de Voz EoPix

Documentação completa para rodar a API de Clonagem de Voz usando Docker.

## 📋 Índice

- [Pré-requisitos](#pré-requisitos)
- [Quick Start](#quick-start)
- [Build](#build)
- [Executar](#executar)
- [Configuração](#configuração)
- [Troubleshooting](#troubleshooting)

---

## 🔧 Pré-requisitos

### Docker e Docker Compose

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install docker.io docker-compose

# Adicionar usuário ao grupo docker
sudo usermod -aG docker $USER
newgrp docker
```

### Para GPU (Opcional)

Se você tem uma GPU NVIDIA e quer aceleração por hardware:

```bash
# Instalar NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Testar
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

---

## 🚀 Quick Start

### Opção 1: Script Automatizado (Recomendado)

```bash
# 1. Build da imagem
chmod +x build.sh docker-start.sh
./build.sh
# Escolha opção 1 (CPU) ou 2 (GPU)

# 2. Iniciar API
./docker-start.sh
# Escolha opção 1 (CPU) ou 2 (GPU)
```

### Opção 2: Docker Compose Manual

```bash
# CPU (porta 8000)
docker-compose up -d tts-api-cpu

# GPU (porta 8001) 
docker-compose up -d tts-api-gpu

# Ver logs
docker-compose logs -f tts-api-cpu
```

### Opção 3: Docker Run Direto

```bash
# Build
docker build -t eopix/tts-api:latest-cpu -f Dockerfile .

# Run
docker run -d \
  --name tts-api \
  -p 8000:8000 \
  -v $(pwd)/audio:/app/audio \
  eopix/tts-api:latest-cpu
```

---

## 🏗️ Build

### Build CPU

```bash
# Usando script
./build.sh  # Opção 1

# Ou manualmente
docker build -f Dockerfile -t eopix/tts-api:latest-cpu .
```

### Build GPU

```bash
# Usando script
./build.sh  # Opção 2

# Ou manualmente
docker build -f Dockerfile.gpu -t eopix/tts-api:latest-gpu .
```

### Build Multi-Stage (Otimizado)

Os Dockerfiles já usam multi-stage build para:
- ✅ Imagens menores (apenas runtime no final)
- ✅ Build mais rápido (cache de dependências)
- ✅ Mais seguro (usuário não-root)

---

## ▶️ Executar

### Com Docker Compose (Recomendado)

```bash
# Iniciar
docker-compose up -d tts-api-cpu

# Parar
docker-compose down

# Ver logs
docker-compose logs -f tts-api-cpu

# Restart
docker-compose restart tts-api-cpu

# Status
docker-compose ps
```

### Com Docker Run

```bash
# CPU
docker run -d \
  --name tts-api-cpu \
  -p 8000:8000 \
  -v $(pwd)/audio:/app/audio \
  -v $(pwd)/api/audio/uploads:/app/api/audio/uploads \
  -v $(pwd)/api/audio/outputs:/app/api/audio/outputs \
  --restart unless-stopped \
  eopix/tts-api:latest-cpu

# GPU
docker run -d \
  --name tts-api-gpu \
  --gpus all \
  -p 8001:8000 \
  -v $(pwd)/audio:/app/audio \
  -v $(pwd)/api/audio/uploads:/app/api/audio/uploads \
  -v $(pwd)/api/audio/outputs:/app/api/audio/outputs \
  --restart unless-stopped \
  eopix/tts-api:latest-gpu
```

---

## ⚙️ Configuração

### Variáveis de Ambiente

Edite `docker-compose.yml` para configurar:

```yaml
environment:
  - PYTHONUNBUFFERED=1              # Logs em tempo real
  - TTS_HOME=/root/.local/share/tts # Cache de modelos TTS
  - HF_HOME=/root/.cache/huggingface # Cache Hugging Face
  - CUDA_VISIBLE_DEVICES=0          # GPU a usar (apenas GPU)
```

### Volumes Persistentes

```yaml
volumes:
  # Áudio (entrada/saída)
  - ./audio:/app/audio
  - ./api/audio/uploads:/app/api/audio/uploads
  - ./api/audio/outputs:/app/api/audio/outputs
  
  # Cache (evita re-download de modelos)
  - tts-models-cache:/root/.local/share/tts
  - huggingface-cache:/root/.cache/huggingface
```

### Limites de Recursos

```yaml
deploy:
  resources:
    limits:
      cpus: '4.0'      # Máximo de CPUs
      memory: 8G       # Máxima memória RAM
    reservations:
      cpus: '2.0'      # Mínimo garantido
      memory: 4G       # Mínimo garantido
```

---

## 🔍 Monitoramento

### Logs

```bash
# Tempo real
docker-compose logs -f tts-api-cpu

# Últimas 100 linhas
docker-compose logs --tail=100 tts-api-cpu

# Logs de um período
docker logs --since 1h tts-api-cpu
```

### Health Check

```bash
# Via curl
curl http://localhost:8000/health

# Via Docker
docker inspect --format='{{.State.Health.Status}}' tts-api-cpu
```

### Estatísticas de Recursos

```bash
# Todos os containers
docker stats

# Container específico
docker stats tts-api-cpu
```

---

## 🧪 Testes

### Testar API

```bash
# Health check
curl http://localhost:8000/health

# Gerar áudio
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Olá, este é um teste de clonagem de voz!",
    "voice_ref": "audio/minha_voz.mp3",
    "language": "pt"
  }'

# Ver documentação interativa
xdg-open http://localhost:8000/docs
```

---

## 🐛 Troubleshooting

### Container não inicia

```bash
# Ver logs
docker-compose logs tts-api-cpu

# Verificar status
docker-compose ps

# Recriar container
docker-compose up -d --force-recreate tts-api-cpu
```

### Erro "Address already in use"

```bash
# Verificar o que está usando a porta
sudo lsof -i :8000

# Matar processo
sudo kill -9 <PID>

# Ou mudar porta no docker-compose.yml
ports:
  - "8888:8000"  # Acesse em localhost:8888
```

### GPU não detectada

```bash
# Verificar NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Se falhar, reinstalar nvidia-docker2
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Modelos não baixam

```bash
# Entrar no container
docker exec -it tts-api-cpu bash

# Verificar conexão
ping huggingface.co

# Baixar manualmente
python -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')"
```

### Memória insuficiente

```bash
# Aumentar limites no docker-compose.yml
deploy:
  resources:
    limits:
      memory: 16G  # Aumentar de 8G para 16G
```

### Limpar tudo e recomeçar

```bash
# Parar e remover containers
docker-compose down

# Remover volumes
docker volume rm eopix-tts-models-cache eopix-huggingface-cache

# Remover imagens
docker rmi eopix/tts-api:latest-cpu eopix/tts-api:latest-gpu

# Rebuild
./build.sh
```

---

## 📊 Performance

### CPU vs GPU

| Operação | CPU (i7) | GPU (RTX 3060) |
|----------|----------|----------------|
| Carregamento inicial | ~30s | ~60s |
| Geração de 10s de áudio | ~15s | ~3s |
| Transcrição (Whisper) | ~20s | ~5s |

### Otimizações

1. **Use volumes para cache de modelos** - Evita re-download
2. **Aumente memória compartilhada** - Para processamento paralelo
3. **Use GPU se disponível** - 5x mais rápido
4. **Limite recursos** - Evita consumir todo o sistema

---

## 🔒 Segurança

✅ Container roda como usuário não-root (`appuser`)  
✅ Apenas portas necessárias expostas  
✅ Volumes com permissões restritas  
✅ Imagem otimizada (sem ferramentas de build no final)  

---

## 📝 Comandos Úteis

```bash
# Entrar no container
docker exec -it tts-api-cpu bash

# Copiar arquivo para dentro do container
docker cp minha_voz.mp3 tts-api-cpu:/app/audio/

# Copiar arquivo do container
docker cp tts-api-cpu:/app/api/audio/outputs/audio.wav ./

# Ver uso de disco
docker system df

# Limpar espaço
docker system prune -a --volumes
```

---

## 🆘 Suporte

- **Documentação da API**: http://localhost:8000/docs
- **Health check**: http://localhost:8000/health
- **GitHub Issues**: [Link para issues]

---

## 📄 Licença

Este projeto está sob a licença especificada no repositório principal.

