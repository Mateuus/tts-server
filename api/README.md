# 🎤 API de Clonagem de Voz

FastAPI para gerar áudios com clonagem de voz usando Coqui TTS.

## 🚀 Como Usar

### 1. Instalar Dependências

```bash
source ../../tts_env/bin/activate
pip install -r api/requirements.txt
```

### 2. Iniciar a API

```bash
cd api
python app.py
```

Ou use o script:
```bash
./api/start.sh
```

### 3. Acessar a Documentação

Abra no navegador: http://localhost:8000/docs

## 📝 Endpoints

### POST `/generate`
Gera áudio com clonagem de voz

**Request:**
```json
{
  "text": "Sua mensagem em português",
  "voice_ref": "../audio/minha_voz.mp3",
  "language": "pt",
  "speed": 0.95,
  "output_filename": "resultado.wav"
}
```

**Response:**
```json
{
  "success": true,
  "message": "✅ Áudio gerado com sucesso",
  "filename": "resultado.wav",
  "filepath": "../audio/outputs/resultado.wav",
  "size_kb": 123.45
}
```

### GET `/health`
Verifica status da API

### GET `/list`
Lista todos os áudios gerados

### GET `/audio/{filename}`
Download de arquivo de áudio

### POST `/transcribe`
Transcreve áudio em texto (usando Whisper)

**Request:**
```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.mp3" \
  -F "language=pt"
```

**Response:**
```json
{
  "success": true,
  "message": "✅ Áudio transcrito com sucesso",
  "text": "Este é o texto transcrito do áudio",
  "language": "pt",
  "duration": 5.23
}
```

**Parâmetros:**
- `file`: Arquivo de áudio (mp3, wav, m4a, flac, ogg, webm)
- `language`: Idioma do áudio (pt, en, es, etc.) - padrão: pt

## 🧪 Testar

```bash
# Executar testes
python api/test_api.py

# Ou manualmente - Gerar áudio
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Olá mundo!"}'

# Transcrever áudio
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.mp3" \
  -F "language=pt"
```

## 📁 Estrutura

```
tts-server/
├── api/
│   ├── app.py              # API principal
│   ├── requirements.txt     # Dependências
│   ├── test_api.py          # Testes
│   └── start.sh             # Script de start
└── audio/
    ├── minha_voz.mp3       # Voz de referência
    └── outputs/            # Áudios gerados
```

## 🔧 Configuração

Ajuste o modelo padrão em `app.py`:
```python
tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
```

Ou use outro modelo:
- `tts_models/pt/cv/vits` - Simples, sem clonagem
- `tts_models/multilingual/multi-dataset/your_tts` - Alternativa

## 💡 Exemplo de Uso com Python

```python
import requests

# Gerar áudio
response = requests.post(
    "http://localhost:8000/generate",
    json={
        "text": "Seu texto aqui",
        "voice_ref": "../audio/minha_voz.mp3",
        "speed": 0.95
    }
)
result = response.json()
print(f"Arquivo: {result['filename']}")

# Transcrever áudio
with open("audio.mp3", "rb") as f:
    response = requests.post(
        "http://localhost:8000/transcribe",
        files={"file": f},
        data={"language": "pt"}
    )
result = response.json()
print(f"Texto: {result['text']}")
```

## 📚 Documentação Completa

Acesse: http://localhost:8000/docs

