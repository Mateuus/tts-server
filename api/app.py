#!/usr/bin/env python3
"""
FastAPI para Geração de Áudio com Clonagem de Voz
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional
import uvicorn

# Adicionar paths necessários
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

app = FastAPI(
    title="🎤 API de Clonagem de Voz",
    description="Gere áudios com clonagem de voz usando Coqui TTS",
    version="1.0.0"
)

# Configurações
UPLOAD_DIR = Path("audio/uploads")
OUTPUT_DIR = Path("audio/outputs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Carregar TTS uma vez (lazy loading)
import threading
_model_lock = threading.Lock()
_tts_model = None
_TTS_READY = False
_whisper_model = None
_WHISPER_READY = False

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

# Pre-carregar modelos
tts_model = load_tts_model()
TTS_READY = _TTS_READY
whisper_model = load_whisper_model()
WHISPER_READY = _WHISPER_READY


class AudioRequest(BaseModel):
    """Request model para geração de áudio"""
    text: str
    voice_ref: Optional[str] = "audio/minha_voz.mp3"
    language: str = "pt"
    speed: Optional[float] = 0.95
    output_filename: Optional[str] = None


class AudioResponse(BaseModel):
    """Response model para áudio gerado"""
    success: bool
    message: str
    filename: Optional[str] = None
    filepath: Optional[str] = None
    size_kb: Optional[float] = None


class TranscribeResponse(BaseModel):
    """Response model para transcrição de áudio"""
    success: bool
    message: str
    text: Optional[str] = None
    language: Optional[str] = None
    duration: Optional[float] = None


@app.get("/")
async def root():
    """Endpoint raiz"""
    return {
        "message": "🎤 API de Clonagem de Voz",
        "status": "ready" if TTS_READY else "not ready",
        "endpoints": {
            "health": "/health",
            "generate": "/generate (POST)",
            "transcribe": "/transcribe (POST)",
            "list": "/list",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health():
    """Verificar saúde da API"""
    return {
        "status": "healthy",
        "tts_ready": TTS_READY,
        "whisper_ready": WHISPER_READY,
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
    
    if not TTS_READY:
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
        print(f"   Texto: {request.text[:50]}...")
        print(f"   Voz: {request.voice_ref}")
        
        # Gerar áudio
        current_model.tts_to_file(
            text=request.text,
            speaker_wav=request.voice_ref,
            language=request.language,
            file_path=str(filepath),
            speed=request.speed
        )
        
        # Verificar se foi criado
        if filepath.exists():
            size_kb = filepath.stat().st_size / 1024
            
            return AudioResponse(
                success=True,
                message=f"✅ Áudio gerado com sucesso",
                filename=filename,
                filepath=str(filepath),
                size_kb=round(size_kb, 2)
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
    
    if not WHISPER_READY:
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

