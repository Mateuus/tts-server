# 🐸 Coqui TTS Server

Servidor Docker para sintetização de texto em fala usando [Coqui TTS](https://github.com/idiap/coqui-ai-TTS/) (mantido pelo Idiap Research Institute, v0.27.2+).

## 📋 Pré-requisitos

- Docker instalado
- Docker Compose instalado
- (Opcional) NVIDIA Docker runtime para suporte GPU

**💡 Para instalação local (sem Docker):** Veja o guia completo em [`INSTALACAO.md`](INSTALACAO.md)

## 🚀 Instalação

### CPU Only

```bash
cd tts-server
docker-compose build tts-cpu
docker-compose up -d tts-cpu
```

### Com suporte GPU

```bash
cd tts-server
docker-compose build tts-gpu
docker-compose up -d tts-gpu
```

## 🎯 Uso

### Testar a instalação

```bash
docker exec -it coqui-tts-cpu python -c "from TTS.api import TTS; print('OK')"
```

### Exemplo de uso Python

```python
from TTS.api import TTS

# Inicializar TTS
tts = TTS("tts_models/pt/cv/vits")

# Gerar áudio
tts.tts_to_file(
    "Este é um teste de sintetização de fala em português.",
    file_path="output.wav"
)
```

### Executar TTS via terminal

```bash
docker exec -it coqui-tts-cpu tts --text "Olá, mundo!" --out_path /app/audio/output.wav
```

## 📁 Estrutura

```
tts-server/
├── Dockerfile          # Imagem CPU
├── Dockerfile.gpu      # Imagem GPU
├── docker-compose.yml  # Configuração Docker Compose
├── requirements.txt    # Dependências Python
├── INSTALACAO.md       # Guia completo de instalação
├── instalar_dependencias.sh  # Script para instalar dependências (detecta GPU)
├── criar_env.sh        # Script para criar arquivo .env
├── iniciar_api.sh      # Script para iniciar a API localmente
├── models/            # Modelos baixados (criado automaticamente)
├── audio/             # Arquivos de áudio gerados
└── api/               # Código da API FastAPI
    └── app.py         # Aplicação principal
```

## ⚙️ Configuração (.env)

Para instalação local, configure as variáveis de ambiente criando um arquivo `.env`:

```bash
./criar_env.sh
```

Ou consulte [`INSTALACAO.md`](INSTALACAO.md) para opções detalhadas de configuração.

## 🔧 Modelos Disponíveis

Listar modelos disponíveis:
```bash
docker exec -it coqui-tts-cpu tts --list_models
```

Alguns modelos em português:
- `tts_models/pt/cv/vits` - Português (Common Voice)
- `tts_models/multilingual/multi-dataset/xtts_v2` - Multilíngue com clonagem de voz
- `tts_models/multilingual/multi-dataset/your_tts` - Multilíngue avançado

## 🎤 Clonagem de Voz

```python
from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
tts.tts_to_file(
    text="Olá, esta é uma mensagem de teste.",
    speaker_wav="ref_speaker.wav",  # Arquivo de referência
    language="pt",
    file_path="output.wav"
)
```

## 📝 Exemplos de Comandos

### Texto para fala simples
```bash
docker exec -it coqui-tts-cpu tts \
  --text "Esta é uma mensagem de teste" \
  --out_path /app/audio/teste.wav
```

### Com modelo específico
```bash
docker exec -it coqui-tts-cpu tts \
  --model_name "tts_models/pt/cv/vits" \
  --text "Teste em português" \
  --out_path /app/audio/pt_test.wav
```

### Gerar áudio e reproduzir
```bash
docker exec -it coqui-tts-cpu tts \
  --text "Olá mundo" \
  --pipe_out \
  --out_path /app/audio/output.wav | aplay
```

## 🛠️ Troubleshooting

### Verificar GPU
```bash
docker exec -it coqui-tts-gpu nvidia-smi
```

### Ver logs
```bash
docker-compose logs -f tts-cpu
# ou
docker-compose logs -f tts-gpu
```

### Reiniciar container
```bash
docker-compose restart tts-cpu
```

## 📚 Recursos

- [Documentação Coqui TTS](https://github.com/coqui-ai/TTS)
- [Modelos Disponíveis](https://github.com/coqui-ai/TTS#list-models)
- [API Reference](https://github.com/coqui-ai/TTS/blob/dev/TTS/api.py)

## 🎓 Exemplo Completo

Veja o arquivo `example_usage.py` para um exemplo completo de uso.

