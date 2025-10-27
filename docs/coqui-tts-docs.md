# 🐸 Coqui TTS Documentation

## Visão Geral

**🐸TTS é uma biblioteca para geração avançada de Text-to-Speech.**

- 🚀 Modelos pré-treinados em +1100 idiomas
- 🛠️ Ferramentas para treinar novos modelos
- 📚 Utilitários para análise e curadoria de datasets

## Instalação

### Requisitos
- Python >= 3.9, < 3.12
- Ubuntu 18.04+

### Instalação via pip (recomendado)
```bash
pip install TTS
```

### Instalação via git
```bash
git clone https://github.com/coqui-ai/TTS
pip install -e .[all,dev,notebooks]
```

## Uso Básico - Python API

### Importação
```python
from TTS.api import TTS
```

### Sintetizar Fala

```python
# Inicializar TTS
tts = TTS("tts_models/pt/cv/vits")

# Gerar áudio
tts.tts_to_file(
    "Este é um teste de sintetização de fala em português.",
    file_path="output.wav"
)
```

## Modelos Disponíveis

### Listar Modelos
```python
from TTS.api import TTS

# Listar todos os modelos
models = TTS.list_models()

# Buscar modelos por idioma
portuguese_models = [m for m in models if 'pt' in m.lower()]
```

### Modelos Comuns

**Português:**
- `tts_models/pt/cv/vits` - Português (Common Voice)
- `tts_models/multilingual/multi-dataset/xtts_v2` - Multilíngue com clonagem de voz

**Multilíngue:**
- `tts_models/multilingual/multi-dataset/xtts_v2` - Clonagem de voz com 16 idiomas
- `tts_models/multilingual/multi-dataset/your_tts` - Multilíngue avançado

## Clonagem de Voz (Voice Cloning)

```python
from TTS.api import TTS

# Inicializar XTTS (suporta clonagem)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

# Clonar voz
tts.tts_to_file(
    text="Olá, esta é uma mensagem de teste.",
    speaker_wav="ref_speaker.wav",  # Arquivo de referência
    language="pt",
    file_path="output.wav"
)
```

## API Completa do TTS

### Inicialização
```python
TTS(
    model_name: str,           # Nome do modelo
    gpu: bool = False,         # Usar GPU
    progress_bar: bool = True  # Mostrar barra de progresso
)
```

### Métodos Principais

#### `tts_to_file()`
Sintetizar fala e salvar em arquivo.

```python
tts.tts_to_file(
    text: str,                    # Texto a ser sintetizado
    file_path: str,               # Caminho do arquivo de saída
    speaker_wav: str = None,      # Arquivo de voz de referência (clonagem)
    language: str = None,          # Idioma (para XTTS)
    **kwargs                      # Outros parâmetros específicos do modelo
)
```

#### `tts()`
Sintetizar fala e retornar array NumPy.

```python
wav = tts.tts(
    text="Olá, mundo!",
    speaker_wav="speaker.wav"
)
```

#### `speaker_manager`
Gerenciar speakers (vozes)

```python
# Listar speakers disponíveis
speakers = tts.speaker_manager.speakers

# Usar speaker específico
tts.tts_to_file(
    text="Olá",
    file_path="output.wav",
    speaker=speakers[0]  # ou speaker_name
)
```

## Parâmetros Comuns

### Velocidade
```python
tts.tts_to_file(
    text="Texto",
    file_path="output.wav",
    speed=1.2  # Mais rápido (padrão: 1.0)
)
```

### Temperature (XTTS)
```python
tts.tts_to_file(
    text="Texto",
    file_path="output.wav",
    speaker_wav="speaker.wav",
    temperature=0.7  # Controle criatividade (0.0-1.0)
)
```

### Top-p e Top-k (XTTS)
```python
tts.tts_to_file(
    text="Texto",
    file_path="output.wav",
    speaker_wav="speaker.wav",
    top_p=0.85,  # Nucleus sampling
    top_k=50     # Top-k sampling
)
```

## Streaming (Baixa Latência)

```python
from TTS.api import TTS
import numpy as np

# Inicializar com streaming
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

# Gerar com streaming (< 200ms latência)
for wav_chunk in tts.tts_stream(
    text="Este é um texto longo que será processado em chunks.",
    stream_chunk_size=100,  # Tamanho do chunk
    speaker_wav="speaker.wav",
    language="pt"
):
    # Processar chunks incrementalmente
    print(f"Chunk: {len(wav_chunk)} samples")
```

## Modelos Especializados

### Bark (Generation de Sons)
```python
from TTS.api import TTS

tts = TTS("tts_models/en/ljspeech/bark")
tts.tts_to_file(
    text="Can you generate: meow [LAUGHS], bark bark",
    file_path="output.wav"
)
```

### Tortoise (Alta Qualidade)
```python
tts = TTS("tts_models/en/ljspeech/tortoise-v2")
tts.tts_to_file(
    text="Texto com qualidade muito alta",
    file_path="output.wav",
    voice_samples=["voice1.wav", "voice2.wav", "voice3.wav"],
    conditioning_latents=None,
    k=1,
    use_deterministic_seed=42
)
```

## Suporte a Múltiplos Idiomas

### XTTS v2 (16 Idiomas)
```python
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

# Português
tts.tts_to_file("Olá", file_path="pt.wav", language="pt")

# Inglês
tts.tts_to_file("Hello", file_path="en.wav", language="en")

# Espanhol
tts.tts_to_file("Hola", file_path="es.wav", language="es")

# Alemão
tts.tts_to_file("Hallo", file_path="de.wav", language="de")
```

**Idiomas suportados pelo XTTS v2:**
- en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko

## Command Line (tts)

### Sintaxe Básica
```bash
tts --text "Texto" --out_path output.wav
```

### Com Modelo Específico
```bash
tts --text "Texto" \
    --model_name "tts_models/pt/cv/vits" \
    --out_path output.wav
```

### Com Speaker
```bash
tts --text "Texto" \
    --out_path output.wav \
    --model_name "modelo_multispeaker" \
    --speaker_idx 0
```

### Stream Output
```bash
tts --text "Texto" \
    --out_path output.wav \
    --pipe_out | aplay
```

## Troubleshooting

### Modelo não encontrado
```python
# Listar modelos disponíveis
from TTS.api import TTS
print(TTS.list_models())
```

### Erro de memória
```python
# Usar CPU ao invés de GPU
tts = TTS("modelo", gpu=False)

# Ou usar GPU específica
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

### Clonagem não funciona
- Certifique-se de usar XTTS ou YourTTS
- Audio de referência deve ter pelo menos 3 segundos
- Formato: WAV, 22050 Hz ou 24000 Hz

## Recursos Adicionais

- **GitHub:** https://github.com/coqui-ai/TTS
- **Documentação:** https://docs.coqui.ai/
- **Discord:** https://discord.gg/5eXr5seRrv
- **Coqui Studio:** https://coqui.ai/studio
- **Blog:** https://coqui.ai/blog

## Exemplos de Integração

### Com FastAPI
```python
from fastapi import FastAPI
from TTS.api import TTS

app = FastAPI()
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

@app.post("/synthesize")
def synthesize(text: str, speaker_wav: str):
    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_wav,
        file_path="output.wav",
        language="pt"
    )
    return {"status": "success", "file": "output.wav"}
```

### Com Streamlit
```python
import streamlit as st
from TTS.api import TTS

tts = TTS("tts_models/pt/cv/vits")

text = st.text_input("Digite o texto:")
if st.button("Sintetizar"):
    tts.tts_to_file(text, file_path="output.wav")
    st.audio("output.wav")
```

## Licença

A maioria dos modelos usa a licença MPL 2.0.

---
*Documentação gerada a partir de https://docs.coqui.ai/*

