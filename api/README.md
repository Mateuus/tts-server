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

## 🧪 Testar

```bash
# Executar testes
python api/test_api.py

# Ou manualmente
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Olá mundo!"}'
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
```

## 📚 Documentação Completa

Acesse: http://localhost:8000/docs

