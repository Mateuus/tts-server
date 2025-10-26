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
  "output_filename": "resultado.wav",
  "return_base64": false,
  "banned_words": "clonagem,Open Voice"
}
```

**Response:**
```json
{
  "success": true,
  "message": "✅ Áudio gerado com sucesso",
  "filename": "resultado.wav",
  "filepath": "../audio/outputs/resultado.wav",
  "size_kb": 123.45,
  "base64": null,
  "filtered_words": ["clonagem"],
  "filtered_text": "Este é um teste de ######### de voz"
}
```

**Nota:** Quando `banned_words` é usado, o texto retornado em `filtered_text` terá `#` substituindo as palavras banidas, mas o TTS no áudio lerá "Hashtag" ao invés da palavra.

**Parâmetros:**
- `text`: Texto para gerar áudio
- `voice_ref`: Caminho para arquivo de voz de referência
- `language`: Idioma (pt, en, es, etc.)
- `speed`: Velocidade do áudio (0.95 = padrão)
- `output_filename`: Nome do arquivo de saída (opcional)
- `return_base64`: Retornar áudio em base64 (padrão: false)
- `banned_words`: Palavras banidas separadas por vírgula - serão substituídas por "Hashtag" no texto/áudio

**Response com base64 (quando `return_base64: true`):**
```json
{
  "success": true,
  "message": "✅ Áudio gerado com sucesso",
  "filename": "resultado.wav",
  "filepath": "../audio/outputs/resultado.wav",
  "size_kb": 123.45,
  "base64": "UklGRiQAAAAAABKAAAAA..."
}
```

### GET `/health`
Verifica status da API

### GET `/list`
Lista todos os áudios gerados

### GET `/audio/{filename}`
Download de arquivo de áudio

**Nota:** A partir de agora, os endpoints de geração (`/generate` e `/filter`) suportam retornar o áudio em **base64** diretamente na resposta, evitando a necessidade de salvar arquivos de output.

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

### POST `/filter`
Filtra palavras banidas em áudio

**Palavras banidas padrão:** `['clonagem', 'Open Voice']`

**Request:**
```bash
curl -X POST http://localhost:8000/filter \
  -F "file=@audio.mp3" \
  -F "language=pt"
```

**Response:**
```json
{
  "success": true,
  "message": "✅ Áudio filtrado com sucesso",
  "text": "Este é o texto com ###### no lugar de palavras banidas",
  "censored_words": ["clonagem"],
  "filename": "censored_20240101_120000.mp3",
  "filepath": "audio/outputs/censored_20240101_120000.mp3",
  "language": "pt"
}
```

**Parâmetros:**
- `file`: Arquivo de áudio (mp3, wav, m4a, flac, ogg, webm)
- `language`: Idioma do áudio (pt, en, es, etc.) - padrão: pt
- `banned_words`: Palavras banidas separadas por vírgula (opcional)
  - Exemplo: `"clonagem,Open Voice"`
  - Se não fornecido, usa as palavras padrão
- `return_base64`: Retornar áudio em base64 (padrão: false)

**Nota:** O áudio filtrado terá beeps adicionados onde palavras banidas foram detectadas e o texto terá `#` substituindo as palavras banidas.

**Exemplo com base64:**
```bash
curl -X POST http://localhost:8000/filter \
  -F "file=@audio.mp3" \
  -F "language=pt" \
  -F "return_base64=true"
```

**Response com base64:**
```json
{
  "success": true,
  "message": "✅ Áudio filtrado com sucesso",
  "text": "Texto com ######...",
  "censored_words": ["clonagem"],
  "filename": "censored_20240101_120000.mp3",
  "base64": "UklGRiQAAAAAABKAAAAA..." // Áudio em base64
}
```

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

# Filtrar áudio (usa palavras padrão)
curl -X POST http://localhost:8000/filter \
  -F "file=@audio.mp3" \
  -F "language=pt"

# Filtrar áudio com palavras personalizadas
curl -X POST http://localhost:8000/filter \
  -F "file=@audio.mp3" \
  -F "language=pt" \
  -F "banned_words=palavra1,palavra2,palavra3"
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

# Gerar áudio (sem base64)
response = requests.post(
    "http://localhost:8000/generate",
    json={
        "text": "Seu texto aqui",
        "voice_ref": "../audio/minha_voz.mp3",
        "speed": 0.95,
        "return_base64": False
    }
)
result = response.json()
print(f"Arquivo: {result['filename']}")

# Gerar áudio com base64
response = requests.post(
    "http://localhost:8000/generate",
    json={
        "text": "Seu texto aqui",
        "voice_ref": "../audio/minha_voz.mp3",
        "speed": 0.95,
        "return_base64": True
    }
)
result = response.json()
if result['base64']:
    import base64
    audio_data = base64.b64decode(result['base64'])
    with open('audio_output.wav', 'wb') as f:
        f.write(audio_data)
    print("Áudio salvo a partir de base64!")

# Transcrever áudio
with open("audio.mp3", "rb") as f:
    response = requests.post(
        "http://localhost:8000/transcribe",
        files={"file": f},
        data={"language": "pt"}
    )
result = response.json()
print(f"Texto: {result['text']}")

# Filtrar áudio (com palavras padrão)
with open("audio.mp3", "rb") as f:
    response = requests.post(
        "http://localhost:8000/filter",
        files={"file": f},
        data={"language": "pt"}
    )
result = response.json()
print(f"Texto filtrado: {result['text']}")

# Filtrar áudio com palavras personalizadas
with open("audio.mp3", "rb") as f:
    response = requests.post(
        "http://localhost:8000/filter",
        files={"file": f},
        data={
            "language": "pt",
            "banned_words": "palavra1,palavra2,palavra3"
        }
    )
result = response.json()
print(f"Texto filtrado: {result['text']}")
print(f"Palavras filtradas: {result['censored_words']}")
print(f"Arquivo filtrado: {result['filename']}")

# Filtrar áudio e obter base64
with open("audio.mp3", "rb") as f:
    response = requests.post(
        "http://localhost:8000/filter",
        files={"file": f},
        data={
            "language": "pt",
            "return_base64": "true"
        }
    )
result = response.json()
if result['base64']:
    import base64
    audio_data = base64.b64decode(result['base64'])
    with open('audio_filtrado.mp3', 'wb') as f:
        f.write(audio_data)
    print("Áudio filtrado salvo a partir de base64!")
```

## 📚 Documentação Completa

Acesse: http://localhost:8000/docs

