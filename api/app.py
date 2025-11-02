#!/usr/bin/env python3
"""
FastAPI para Geração de Áudio com Clonagem de Voz
"""

import os
import sys
import base64
import warnings
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional
import uvicorn
import torch

# Suprimir warnings específicos do GPT2 e outros não críticos
warnings.filterwarnings("ignore", message=".*GPT2InferenceModel.*")
warnings.filterwarnings("ignore", message=".*does not have a tokenizer_class attribute.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# Adicionar paths necessários
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configurações
UPLOAD_DIR = Path("audio/uploads")
OUTPUT_DIR = Path("audio/outputs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Palavras banidas
BANNED_WORDS = ['clonagem', 'Open Voice']

# Variáveis globais para modelos (lazy loading)
import threading
_model_lock = threading.Lock()
_tts_model = None
_TTS_READY = False
_whisper_model = None
_WHISPER_READY = False
_ai_models = {}
_AI_READY = False
_MODELS_PRELOADED = False  # Flag para evitar carregamento múltiplo

def load_tts_model():
    """Carrega o modelo TTS apenas uma vez (thread-safe)"""
    global _tts_model, _TTS_READY
    
    with _model_lock:
        if _tts_model is not None:
            return _tts_model
        
        print("🔄 Carregando modelo TTS...")
        try:
            from TTS.api import TTS
            _tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            print("✅ Modelo TTS carregado!")
            _TTS_READY = True
        except Exception as e:
            print(f"⚠️ Erro ao carregar modelo TTS: {e}")
            _TTS_READY = False
        
        return _tts_model

def load_whisper_model():
    """Carrega o modelo Whisper para transcrição (lazy loading)"""
    global _whisper_model, _WHISPER_READY
    
    with _model_lock:
        if _whisper_model is not None:
            return _whisper_model
        
        print("🔄 Carregando modelo Whisper...")
        try:
            import whisper
            _whisper_model = whisper.load_model("base")
            print("✅ Modelo Whisper carregado!")
            _WHISPER_READY = True
        except Exception as e:
            print(f"⚠️ Erro ao carregar modelo Whisper: {e}")
            _WHISPER_READY = False
        
        return _whisper_model

def load_ai_model(model_name: str = "lula"):
    """Carrega modelos de IA para geração de texto (lazy loading)"""
    global _ai_models, _AI_READY
    
    with _model_lock:
        if model_name in _ai_models:
            return _ai_models[model_name]
        
        print(f"🔄 Carregando modelo de IA: {model_name}...")
        try:
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            
            # Mapear nomes para paths
            model_paths = {
                "lula": "../models/lulaoficial-ptbrasil",
                "lulaoficial-ptbrasil": "../models/lulaoficial-ptbrasil"
            }
            
            model_path = model_paths.get(model_name.lower(), model_name)
            
            # Carregar modelo e tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            model = GPT2LMHeadModel.from_pretrained(model_path)
            
            _ai_models[model_name] = {
                'model': model,
                'tokenizer': tokenizer
            }
            
            print(f"✅ Modelo de IA '{model_name}' carregado!")
            _AI_READY = True
        except Exception as e:
            print(f"⚠️ Erro ao carregar modelo de IA '{model_name}': {e}")
            _AI_READY = False
        
        return _ai_models.get(model_name)


# Lifespan event handler - carrega modelos no startup e limpa no shutdown
@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Gerencia o ciclo de vida da aplicação"""
    global _MODELS_PRELOADED
    
    # Startup: Carregar modelos
    if not _MODELS_PRELOADED:
        print("\n🔄 Carregando modelos no startup...")
        load_tts_model()
        load_whisper_model()
        _MODELS_PRELOADED = True
        print("✅ Modelos pré-carregados com sucesso!\n")
    else:
        print("⚠️ Modelos já foram carregados, pulando...")
    
    yield  # Aplicação rodando
    
    # Shutdown: Limpeza (opcional)
    print("\n🔄 Encerrando aplicação...")


# Criar app FastAPI com lifespan
app = FastAPI(
    title="🎤 API de Clonagem de Voz",
    description="Gere áudios com clonagem de voz usando Coqui TTS",
    version="1.0.0",
    lifespan=lifespan
)


class AudioRequest(BaseModel):
    """Request model para geração de áudio"""
    text: str
    voice_ref: Optional[str] = "audio/minha_voz.mp3"
    language: str = "pt"
    speed: Optional[float] = 0.95
    output_filename: Optional[str] = None
    return_base64: Optional[bool] = False
    banned_words: Optional[str] = None  # Palavras banidas separadas por vírgula


class AudioResponse(BaseModel):
    """Response model para áudio gerado"""
    success: bool
    message: str
    filename: Optional[str] = None
    filepath: Optional[str] = None
    size_kb: Optional[float] = None
    base64: Optional[str] = None
    filtered_words: Optional[list] = None  # Palavras filtradas
    filtered_text: Optional[str] = None  # Texto com # substituindo palavras banidas


class TranscribeResponse(BaseModel):
    """Response model para transcrição de áudio"""
    success: bool
    message: str
    text: Optional[str] = None
    language: Optional[str] = None
    duration: Optional[float] = None


class CensorResponse(BaseModel):
    """Response model para censura de áudio"""
    success: bool
    message: str
    text: Optional[str] = None
    censored_words: Optional[list] = None
    filename: Optional[str] = None
    filepath: Optional[str] = None
    language: Optional[str] = None
    base64: Optional[str] = None


class GenerateAIRequest(BaseModel):
    """Request model para geração de texto com IA"""
    prompt: str
    model_name: Optional[str] = "lula"
    max_length: Optional[int] = 150
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    do_sample: Optional[bool] = True
    generate_audio: Optional[bool] = True  # Gerar áudio automaticamente
    voice_ref: Optional[str] = "audio/minha_voz.mp3"  # Arquivo de voz para clonagem
    language: str = "pt"  # Idioma para TTS
    
    model_config = {"protected_namespaces": ()}  # Resolver avisos do Pydantic


class GenerateAIResponse(BaseModel):
    """Response model para geração de texto com IA"""
    success: bool
    message: str
    generated_text: Optional[str] = None
    prompt: Optional[str] = None
    ai_model_used: Optional[str] = None
    audio_filename: Optional[str] = None
    audio_filepath: Optional[str] = None
    length: Optional[int] = None
    
    model_config = {"protected_namespaces": ()}  # Resolver avisos do Pydantic


@app.get("/")
async def root():
    """Endpoint raiz"""
    return {
        "message": "🎤 API de Clonagem de Voz",
        "status": "ready" if _TTS_READY else "not ready",
            "endpoints": {
            "health": "/health",
            "generate": "/generate (POST) - Gerar áudio com clonagem",
            "generateAI": "/generateAI (POST) - Gerar texto com IA",
            "transcribe": "/transcribe (POST)",
            "filter": "/filter (POST)",
            "list": "/list",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health():
    """Verificar saúde da API"""
    return {
        "status": "healthy",
        "tts_ready": _TTS_READY,
        "whisper_ready": _WHISPER_READY,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/generate", response_model=AudioResponse)
async def generate_audio(request: AudioRequest):
    """
    Gerar áudio com clonagem de voz
    
    Args:
        request: AudioRequest com texto e configurações
    
    Returns:
        AudioResponse com informações do arquivo gerado
    """
    # Carregar modelo se necessário
    current_model = load_tts_model()
    
    if not _TTS_READY:
        raise HTTPException(
            status_code=503,
            detail="TTS não está pronto. Verifique os logs."
        )
    
    try:
        # Validar arquivo de voz
        if not os.path.exists(request.voice_ref):
            raise HTTPException(
                status_code=404,
                detail=f"Arquivo de voz não encontrado: {request.voice_ref}"
            )
        
        # Filtrar palavras banidas do texto se fornecido
        # Criar duas versões: uma para retornar (com #) e outra para TTS (com "Hashtag")
        text_for_response = request.text  # Versão com # para resposta
        text_to_generate = request.text   # Versão com "Hashtag" para TTS
        filtered_words_found = []
        
        if request.banned_words:
            import re
            # Converter string separada por vírgulas em lista
            words_to_filter = [word.strip() for word in request.banned_words.split(",") if word.strip()]
            
            # Filtrar palavras banidas - DUAS versões diferentes
            for banned_word in words_to_filter:
                pattern = re.compile(re.escape(banned_word), re.IGNORECASE)
                matches_response = list(pattern.finditer(text_for_response))
                matches_generate = list(pattern.finditer(text_to_generate))
                
                if matches_response:
                    # Versão 1: Substituir por # (para resposta)
                    for match in reversed(matches_response):
                        text_for_response = (
                            text_for_response[:match.start()] + 
                            "#" * len(match.group()) + 
                            text_for_response[match.end():]
                        )
                    
                    # Versão 2: Substituir por "Hashtag" (para TTS)
                    for match in reversed(matches_generate):
                        text_to_generate = (
                            text_to_generate[:match.start()] + 
                            "Hashtag" + 
                            text_to_generate[match.end():]
                        )
                        filtered_words_found.append(match.group())
            
            if filtered_words_found:
                print(f"   🚫 Palavras filtradas: {filtered_words_found}")
                print(f"   📝 Texto para retorno: {text_for_response[:50]}...")
                print(f"   🎤 Texto para TTS: {text_to_generate[:50]}...")
        
        # Gerar nome de arquivo
        if request.output_filename:
            filename = request.output_filename
            if not filename.endswith('.wav'):
                filename += '.wav'
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"audio_{timestamp}.wav"
        
        filepath = OUTPUT_DIR / filename
        
        print(f"\n🎤 Gerando áudio...")
        print(f"   Texto: {text_to_generate[:50]}...")
        print(f"   Voz: {request.voice_ref}")
        
        # Gerar áudio
        current_model.tts_to_file(
            text=text_to_generate,
            speaker_wav=request.voice_ref,
            language=request.language,
            file_path=str(filepath),
            speed=request.speed
        )
        
        # Verificar se foi criado
        if filepath.exists():
            size_kb = filepath.stat().st_size / 1024
            
            # Converter para base64 e limpar arquivo se solicitado
            base64_audio = None
            save_file = True
            
            if request.return_base64:
                with open(filepath, "rb") as audio_file:
                    audio_bytes = audio_file.read()
                    base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
                
                # Se foi pedido base64, não precisa manter o arquivo
                # (delete para economizar espaço)
                filepath.unlink()
                save_file = False
                print(f"   📦 Áudio convertido para base64 (arquivo não salvo)")
            
            return AudioResponse(
                success=True,
                message=f"✅ Áudio gerado com sucesso",
                filename=filename if save_file else None,
                filepath=str(filepath) if save_file else None,
                size_kb=round(size_kb, 2) if save_file else None,
                base64=base64_audio,
                filtered_words=filtered_words_found if filtered_words_found else [],
                filtered_text=text_for_response if filtered_words_found else None
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="Erro ao gerar áudio"
            )
    
    except Exception as e:
        print(f"❌ Erro: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao gerar áudio: {str(e)}"
        )


@app.get("/list")
async def list_audio_files():
    """Listar arquivos de áudio gerados"""
    files = []
    
    if OUTPUT_DIR.exists():
        for file in OUTPUT_DIR.glob("*.wav"):
            stat = file.stat()
            files.append({
                "filename": file.name,
                "path": str(file),
                "size_kb": round(stat.st_size / 1024, 2),
                "created": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
    
    return {
        "count": len(files),
        "files": sorted(files, key=lambda x: x["created"], reverse=True)
    }


@app.get("/audio/{filename}")
async def download_audio(filename: str):
    """Download de arquivo de áudio"""
    from fastapi.responses import FileResponse
    
    filepath = OUTPUT_DIR / filename
    
    if not filepath.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Arquivo não encontrado: {filename}"
        )
    
    return FileResponse(
        path=filepath,
        filename=filename,
        media_type="audio/wav"
    )


@app.post("/filter", response_model=CensorResponse)
async def filter_audio(
    file: UploadFile = File(...),
    language: Optional[str] = Form("pt"),
    banned_words: Optional[str] = Form(None),
    return_base64: Optional[bool] = Form(False)
):
    """
    Filtrar palavras banidas em áudio
    
    Args:
        file: Arquivo de áudio para filtrar
        language: Idioma do áudio (pt, en, es, etc.)
        banned_words: Palavras banidas separadas por vírgula (opcional, usa padrão se não fornecido)
                      Exemplo: "clonagem,Open Voice"
        return_base64: Retornar áudio em base64 (padrão: False)
    
    Returns:
        CensorResponse com o áudio filtrado e texto com # substituindo palavras banidas
    """
    # Processar palavras banidas
    if banned_words:
        # Converter string separada por vírgulas em lista
        words_to_censor = [word.strip() for word in banned_words.split(",") if word.strip()]
    else:
        words_to_censor = BANNED_WORDS
    
    # Carregar modelo se necessário
    current_model = load_whisper_model()
    
    if not _WHISPER_READY:
        raise HTTPException(
            status_code=503,
            detail="Whisper não está pronto. Verifique os logs."
        )
    
    try:
        # Validar tipo de arquivo
        allowed_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.webm'}
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Formato de arquivo não suportado: {file_ext}. Use: {allowed_extensions}"
            )
        
        # Salvar arquivo temporário original
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_filename = f"upload_{timestamp}{file_ext}"
        temp_filepath = UPLOAD_DIR / temp_filename
        
        # Salvar conteúdo do arquivo
        with open(temp_filepath, "wb") as f:
            content = await file.read()
            f.write(content)
        
        print(f"\n🚫 Filtrando áudio...")
        print(f"   Arquivo: {file.filename}")
        print(f"   Palavras banidas: {words_to_censor}")
        
        # Transcrever áudio com segmentos
        result = current_model.transcribe(
            str(temp_filepath),
            language=language if language != "auto" else None,
            word_timestamps=True
        )
        
        # Detectar palavras banidas com timestamps
        censored_info = []
        censored_timestamps = []  # Para armazenar timestamps das palavras banidas
        text = result["text"].strip()
        
        # Buscar palavras banidas nos segmentos usando timestamps
        segments = result.get("segments", [])
        for segment in segments:
            words = segment.get("words", [])
            for word_info in words:
                word_text = word_info.get("word", "").strip().lower()
                
                # Verificar se a palavra está na lista de palavras banidas
                for banned_word in words_to_censor:
                    if banned_word.lower() in word_text:
                        censored_timestamps.append({
                            'start': word_info.get("start", 0),
                            'end': word_info.get("end", 0),
                            'word': word_info.get("word", "")
                        })
                        break
        
        # Buscar por palavras banidas no texto para censura de texto
        for word in words_to_censor:
            import re
            # Criar regex para encontrar a palavra (case insensitive)
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            matches = list(pattern.finditer(text))
            
            if matches:
                for match in matches:
                    censored_info.append({
                        'word': match.group(),
                        'start_pos': match.start(),
                        'end_pos': match.end()
                    })
        
        # Substituir palavras banidas por # no texto
        censored_text = text
        censored_words_found = []
        offset = 0
        
        for info in sorted(censored_info, key=lambda x: x['start_pos']):
            # Ajustar posições com offset
            start = info['start_pos'] + offset
            end = info['end_pos'] + offset
            
            # Substituir por #
            censored_text = censored_text[:start] + '#' + censored_text[end:]
            censored_words_found.append(info['word'].lower())
            offset += 1 - (end - start)  # Ajustar offset
            
            # Adicionar #s extras para cada letra a mais
            for i in range(len(info['word']) - 1):
                censored_text = censored_text[:start + 1 + i] + '#' + censored_text[start + 1 + i:]
                offset += 1
        
        # Criar áudio com beeps nas posições das palavras banidas
        from pydub import AudioSegment
        from pydub.generators import Sine
        
        # Carregar áudio original
        audio = AudioSegment.from_file(str(temp_filepath))
        
        # Gerar beep de 0.5s (1000Hz)
        beep = Sine(1000).to_audio_segment(duration=500).apply_gain(-5)
        
        # Adicionar beeps nas posições exatas das palavras banidas
        if censored_timestamps:
            # Ordenar timestamps por posição no tempo (do fim para o início para evitar problemas de offset)
            censored_timestamps.sort(key=lambda x: x['start'], reverse=True)
            
            for ts_info in censored_timestamps:
                start_ms = int(ts_info['start'] * 1000)  # Converter segundos para milissegundos
                end_ms = int(ts_info['end'] * 1000)
                duration_ms = end_ms - start_ms
                
                # Garantir que não ultrapasse o tamanho do áudio
                if start_ms < len(audio) and end_ms <= len(audio):
                    # Dividir áudio em: início, palavra banida, fim
                    audio_before = audio[:start_ms]
                    audio_after = audio[end_ms:]
                    
                    # Substituir palavra banida por beep
                    audio = audio_before + beep + AudioSegment.silent(duration=max(0, duration_ms - len(beep))) + audio_after
                    
                    print(f"   🚫 Beep inserido em {ts_info['start']:.2f}s para '{ts_info['word']}'")
        
        # Salvar áudio censurado
        censored_filename = f"censored_{timestamp}{file_ext}"
        censored_filepath = OUTPUT_DIR / censored_filename
        audio.export(str(censored_filepath), format=file_ext[1:])
        
        # Limpar arquivo temporário original
        if temp_filepath.exists():
            temp_filepath.unlink()
        
        detected_language = result.get("language", language)
        
        # Converter para base64 e limpar arquivo se solicitado
        base64_audio = None
        save_file = True
        
        if return_base64:
            with open(censored_filepath, "rb") as audio_file:
                audio_bytes = audio_file.read()
                base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
            
            # Se foi pedido base64, não precisa manter o arquivo
            censored_filepath.unlink()
            save_file = False
            print(f"   📦 Áudio convertido para base64 (arquivo não salvo)")
        
        return CensorResponse(
            success=True,
            message=f"✅ Áudio filtrado com sucesso" if censored_words_found else "✅ Áudio processado (nenhuma palavra banida encontrada)",
            text=censored_text,
            censored_words=censored_words_found if censored_words_found else [],
            filename=censored_filename if save_file else None,
            filepath=str(censored_filepath) if save_file else None,
            language=detected_language,
            base64=base64_audio
        )
    
    except Exception as e:
        print(f"❌ Erro: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao filtrar áudio: {str(e)}"
        )


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = "pt"
):
    """
    Transcrever áudio em texto
    
    Args:
        file: Arquivo de áudio para transcrever
        language: Idioma do áudio (pt, en, es, etc.)
    
    Returns:
        TranscribeResponse com o texto transcrito
    """
    # Carregar modelo se necessário
    current_model = load_whisper_model()
    
    if not _WHISPER_READY:
        raise HTTPException(
            status_code=503,
            detail="Whisper não está pronto. Verifique os logs."
        )
    
    try:
        # Validar tipo de arquivo
        allowed_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.webm'}
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Formato de arquivo não suportado: {file_ext}. Use: {allowed_extensions}"
            )
        
        # Salvar arquivo temporário
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_filename = f"upload_{timestamp}{file_ext}"
        temp_filepath = UPLOAD_DIR / temp_filename
        
        # Salvar conteúdo do arquivo
        with open(temp_filepath, "wb") as f:
            content = await file.read()
            f.write(content)
        
        print(f"\n📝 Transcrevendo áudio...")
        print(f"   Arquivo: {file.filename}")
        print(f"   Idioma: {language}")
        
        # Transcrever áudio
        result = current_model.transcribe(
            str(temp_filepath),
            language=language if language != "auto" else None
        )
        
        # Limpar arquivo temporário
        if temp_filepath.exists():
            temp_filepath.unlink()
        
        # Extrair informações
        transcribed_text = result["text"].strip()
        detected_language = result.get("language", language)
        duration = sum(segment.get("end", 0) - segment.get("start", 0) 
                      for segment in result.get("segments", []))
        
        return TranscribeResponse(
            success=True,
            message="✅ Áudio transcrito com sucesso",
            text=transcribed_text,
            language=detected_language,
            duration=round(duration, 2)
        )
    
    except Exception as e:
        print(f"❌ Erro: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao transcrever áudio: {str(e)}"
        )


@app.post("/generateAI", response_model=GenerateAIResponse)
async def generate_ai_text(request: GenerateAIRequest):
    """
    Gerar texto usando IA (GPT-2 customizado)
    
    Args:
        request: GenerateAIRequest com prompt e configurações
    
    Returns:
        GenerateAIResponse com o texto gerado
    """
    try:
        # Carregar modelo
        model_data = load_ai_model(request.model_name)
        
        if model_data is None:
            raise HTTPException(
                status_code=503,
                detail=f"Modelo '{request.model_name}' não está pronto. Verifique os logs."
            )
        
        model = model_data['model']
        tokenizer = model_data['tokenizer']
        
        print(f"\n🤖 Gerando texto com IA...")
        print(f"   Modelo: {request.model_name}")
        print(f"   Prompt: {request.prompt[:50]}...")
        
        # Tokenizar o prompt
        inputs = tokenizer.encode(request.prompt, return_tensors="pt")
        
        # Gerar texto
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=request.do_sample,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decodificar a resposta
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remover o prompt original do texto gerado
        generated_text = generated_text[len(request.prompt):].strip()
        
        # Gerar áudio se solicitado
        audio_filename = None
        audio_filepath = None
        
        if request.generate_audio:
            print(f"\n🎤 Gerando áudio do texto...")
            print(f"   Texto: {generated_text[:50]}...")
            
            # Carregar modelo TTS se necessário
            current_tts_model = load_tts_model()
            
            if _TTS_READY:
                try:
                    # Validar arquivo de voz
                    if not os.path.exists(request.voice_ref):
                        print(f"⚠️ Arquivo de voz não encontrado: {request.voice_ref}")
                    else:
                        # Gerar nome de arquivo
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        audio_filename = f"ai_generated_{timestamp}.wav"
                        audio_filepath = str(OUTPUT_DIR / audio_filename)
                        
                        # Gerar áudio com TTS
                        current_tts_model.tts_to_file(
                            text=generated_text,
                            speaker_wav=request.voice_ref,
                            language=request.language,
                            file_path=audio_filepath,
                            speed=1.0
                        )
                        
                        print(f"   ✅ Áudio gerado: {audio_filename}")
                except Exception as e:
                    print(f"⚠️ Erro ao gerar áudio: {e}")
        
        return GenerateAIResponse(
            success=True,
            message="✅ Texto e áudio gerados com sucesso" if audio_filename else "✅ Texto gerado com sucesso",
            generated_text=generated_text,
            prompt=request.prompt,
            ai_model_used=request.model_name,
            audio_filename=audio_filename,
            audio_filepath=audio_filepath,
            length=len(generated_text)
        )
    
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao gerar texto: {str(e)}"
        )


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Iniciando API de Clonagem de Voz")
    print("="*60)
    print("\n📖 Documentação: http://localhost:8000/docs")
    print("🎤 Health check: http://localhost:8000/health")
    print("\n" + "="*60)
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

