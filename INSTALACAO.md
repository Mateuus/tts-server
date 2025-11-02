# 🚀 Guia de Instalação - TTS Server

Este guia explica como instalar e configurar o servidor TTS (Text-to-Speech) usando Coqui TTS.

## 📋 Pré-requisitos

- Python 3.11 ou superior
- pip (gerenciador de pacotes Python)
- (Opcional) Docker e Docker Compose para uso via containers
- (Opcional) NVIDIA Docker runtime para suporte GPU

## 🔧 Instalação Local (Ambiente Virtual Python)

### 1. Criar e ativar ambiente virtual

```bash
cd tts-server
python3.11 -m venv tts_env
source tts_env/bin/activate  # Linux/Mac
# ou
tts_env\Scripts\activate  # Windows
```

### 2. Instalar dependências

**Opção A:** Use o script automatizado (recomendado - detecta GPU automaticamente):

```bash
./instalar_dependencias.sh
```

**Opção B:** Instalação manual:

Para **CPU** (sem GPU):
```bash
pip install --upgrade pip
pip install torch==2.3.1 torchaudio==2.3.1
pip install -r requirements.txt
```

Para **GPU com CUDA 12.1**:
```bash
pip install --upgrade pip
pip install torch==2.3.1 torchaudio==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 3. Configurar variáveis de ambiente

**Opção A:** Use o script automatizado (recomendado):

```bash
./criar_env.sh
```

**Opção B:** Crie manualmente o arquivo `.env`:

```bash
touch .env
```

E adicione o seguinte conteúdo mínimo:

```env
# Servidor
HOST=0.0.0.0
PORT=8000
DEBUG=true

# Diretórios
UPLOAD_DIR=audio/uploads
OUTPUT_DIR=audio/outputs

# Modelos TTS
TTS_MODEL=tts_models/multilingual/multi-dataset/xtts_v2
DEFAULT_LANGUAGE=pt
DEFAULT_SPEED=0.95
DEFAULT_VOICE_REF=audio/minha_voz.mp3

# Whisper
WHISPER_MODEL=base

# Segurança
BANNED_WORDS=clonagem,Open Voice

# Performance
USE_GPU=false
PYTHONUNBUFFERED=1
```

Edite o arquivo `.env` e ajuste as variáveis conforme necessário (veja seção [Configuração do .env](#configuração-do-env) para todas as opções disponíveis).

### 4. Baixar modelos (opcional)

Os modelos serão baixados automaticamente na primeira execução. Se quiser baixar antecipadamente:

```bash
python -c "from TTS.api import TTS; TTS('tts_models/pt/cv/vits')"
python -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')"
```

### 5. Iniciar a API

```bash
./iniciar_api.sh
```

Ou manualmente:

```bash
source tts_env/bin/activate
cd api
python app.py
```

A API estará disponível em `http://localhost:8000`

## 🐳 Instalação via Docker

### Opção 1: CPU Only

```bash
cd tts-server
docker-compose build tts-cpu
docker-compose up -d tts-cpu
```

A API estará disponível em `http://localhost:5000`

### Opção 2: Com suporte GPU

```bash
cd tts-server
docker-compose build tts-gpu
docker-compose up -d tts-gpu
```

A API estará disponível em `http://localhost:5001`

### Verificar logs

```bash
docker-compose logs -f tts-cpu
# ou
docker-compose logs -f tts-gpu
```

## ⚙️ Configuração do .env

Crie um arquivo `.env` na raiz do projeto com as seguintes variáveis:

```env
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
TTS_MODEL=tts_models/multilingual/multi-dataset/xtts_v2

# Idioma padrão
DEFAULT_LANGUAGE=pt

# Velocidade padrão de fala (0.5 a 2.0)
DEFAULT_SPEED=0.95

# Arquivo de voz de referência padrão
DEFAULT_VOICE_REF=audio/minha_voz.mp3

# ============================================
# WHISPER (Transcrição)
# ============================================
# Modelo Whisper para transcrição (tiny, base, small, medium, large)
WHISPER_MODEL=base

# ============================================
# SEGURANÇA E FILTROS
# ============================================
# Palavras banidas (separadas por vírgula)
BANNED_WORDS=clonagem,Open Voice

# ============================================
# RECURSOS E PERFORMANCE
# ============================================
# Usar GPU se disponível (true/false)
USE_GPU=false

# Device CUDA (0, 1, 2, etc.)
CUDA_VISIBLE_DEVICES=0

# ============================================
# MODELOS DE IA (Opcional)
# ============================================
# Caminho para modelos GPT-2 customizados
AI_MODELS_DIR=../models
```

### Variáveis Importantes

- **HOST e PORT**: Configuração do servidor (padrão: `0.0.0.0:8000`)
- **TTS_MODEL**: Modelo TTS a ser usado
- **DEFAULT_VOICE_REF**: Arquivo de referência de voz padrão
- **BANNED_WORDS**: Palavras que serão filtradas/censuradas
- **USE_GPU**: Ativar suporte a GPU (requer CUDA)

## ✅ Verificar Instalação

### Testar API

```bash
curl http://localhost:8000/health
```

Resposta esperada:
```json
{
  "status": "healthy",
  "tts_ready": true,
  "whisper_ready": true,
  "timestamp": "2024-01-01T12:00:00"
}
```

### Testar geração de áudio

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Olá, esta é uma mensagem de teste",
    "voice_ref": "audio/minha_voz.mp3",
    "language": "pt"
  }'
```

## 📚 Endpoints Disponíveis

- `GET /` - Informações da API
- `GET /health` - Status de saúde
- `GET /docs` - Documentação interativa (Swagger)
- `POST /generate` - Gerar áudio com clonagem de voz
- `POST /transcribe` - Transcrever áudio em texto
- `POST /filter` - Filtrar palavras banidas em áudio
- `GET /list` - Listar arquivos gerados
- `GET /audio/{filename}` - Download de arquivo

## 🔍 Troubleshooting

### Erro: "TTS não está pronto"

Os modelos estão sendo carregados. Aguarde alguns segundos e tente novamente. Verifique os logs para mais detalhes.

### Erro: "Arquivo de voz não encontrado"

Certifique-se de que o arquivo de referência de voz existe no caminho especificado em `voice_ref`.

### Erro de memória GPU

Reduza o tamanho do modelo ou use CPU:
- Edite `.env` e defina `USE_GPU=false`
- Reinicie a API

### Problemas com dependências

Recrie o ambiente virtual:

```bash
rm -rf tts_env
python3.11 -m venv tts_env
source tts_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 📖 Documentação Adicional

- Documentação da API: http://localhost:8000/docs
- [Coqui TTS Documentation](https://github.com/idiap/coqui-ai-TTS/) (mantido pelo Idiap Research Institute)
- [Whisper Documentation](https://github.com/openai/whisper)

## 🎯 Próximos Passos

1. Configure o arquivo `.env` com suas preferências
2. Adicione arquivos de voz de referência em `audio/`
3. Teste os endpoints usando a documentação interativa em `/docs`
4. Integre a API com seu aplicativo

