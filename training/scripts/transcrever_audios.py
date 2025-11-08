#!/usr/bin/env python3
"""
Script para transcrever arquivos de áudio e gerar metadata.csv automaticamente
Uso: python transcrever_audios.py <nome_streamer> [opções]
Exemplo: python transcrever_audios.py skipnho --model base --language pt
"""

import os
import sys
import argparse
import csv
from pathlib import Path
from typing import List, Tuple

# Cores para terminal
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_success(msg: str):
    print(f"{Colors.GREEN}✅ {msg}{Colors.RESET}")

def print_warning(msg: str):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.RESET}")

def print_error(msg: str):
    print(f"{Colors.RED}❌ {msg}{Colors.RESET}")

def print_info(msg: str):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.RESET}")

def print_header(msg: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{msg}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.RESET}\n")

def get_audio_files(wavs_dir: Path) -> List[Path]:
    """Retorna lista de arquivos de áudio ordenados"""
    audio_files = []
    
    # Procurar arquivos WAV
    audio_files.extend(sorted(wavs_dir.glob('*.wav')))
    
    # Procurar arquivos MP3
    audio_files.extend(sorted(wavs_dir.glob('*.mp3')))
    
    # Procurar outros formatos comuns
    audio_files.extend(sorted(wavs_dir.glob('*.flac')))
    audio_files.extend(sorted(wavs_dir.glob('*.ogg')))
    
    return audio_files

def transcribe_audio(audio_path: Path, model_name: str = "base", language: str = "pt") -> str:
    """Transcreve um arquivo de áudio usando Whisper"""
    try:
        import whisper
        
        print_info(f"Transcrevendo: {audio_path.name}...")
        
        # Carregar modelo Whisper
        model = whisper.load_model(model_name)
        
        # Transcrever
        result = model.transcribe(
            str(audio_path),
            language=language if language != "auto" else None,
            task="transcribe",
            verbose=False
        )
        
        # Extrair texto
        text = result["text"].strip()
        
        return text
    
    except ImportError:
        print_error("Whisper não está instalado!")
        print_info("Instale com: pip install openai-whisper")
        sys.exit(1)
    except Exception as e:
        print_error(f"Erro ao transcrever {audio_path.name}: {str(e)}")
        return None

def get_audio_duration(audio_path: Path) -> float:
    """Retorna duração do áudio em segundos"""
    try:
        import librosa
        duration = librosa.get_duration(path=str(audio_path))
        return duration
    except ImportError:
        print_warning("librosa não está instalado. Duração não será verificada.")
        return None
    except Exception as e:
        print_warning(f"Erro ao obter duração de {audio_path.name}: {str(e)}")
        return None

def validate_transcription(text: str, duration: float = None) -> Tuple[bool, str]:
    """Valida a transcrição"""
    if not text or len(text.strip()) < 5:
        return False, "Transcrição muito curta (mínimo 5 caracteres)"
    
    if len(text) > 500:
        return False, f"Transcrição muito longa ({len(text)} caracteres, máximo 500)"
    
    if duration:
        if duration < 1:
            return False, f"Áudio muito curto ({duration:.2f}s, mínimo 1s)"
        if duration > 30:
            return False, f"Áudio muito longo ({duration:.2f}s, máximo 30s)"
    
    return True, None

def generate_metadata_csv(audio_files: List[Path], wavs_dir: Path, metadata_path: Path, 
                         model_name: str = "base", language: str = "pt", 
                         min_duration: float = 1.0, max_duration: float = 30.0,
                         min_text_length: int = 5, max_text_length: int = 500,
                         skip_existing: bool = False):
    """Gera metadata.csv transcrevendo arquivos de áudio"""
    
    # Ler metadata existente (sempre ler para manter entradas existentes)
    existing_entries = {}
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='|')
                for row in reader:
                    if len(row) == 2:
                        filename, text = row
                        existing_entries[filename.strip()] = text.strip()
            print_info(f"Carregados {len(existing_entries)} registros existentes do CSV")
        except Exception as e:
            print_warning(f"Erro ao ler metadata existente: {str(e)}")
    
    # Processar arquivos mantendo ordem
    entries = []
    processed = 0
    skipped = 0
    errors = 0
    
    print_header(f"🎤 Processando {len(audio_files)} arquivo(s) de áudio")
    
    for i, audio_path in enumerate(audio_files, 1):
        filename = audio_path.stem  # Nome sem extensão
        
        # Se já existe no CSV, manter a transcrição existente
        if filename in existing_entries:
            print_info(f"[{i}/{len(audio_files)}] Mantendo existente: {audio_path.name}")
            entries.append((filename, existing_entries[filename]))
            skipped += 1
            continue
        
        # Verificar duração (se possível)
        duration = get_audio_duration(audio_path)
        if duration:
            if duration < min_duration:
                print_warning(f"[{i}/{len(audio_files)}] Pulando (muito curto): {audio_path.name} ({duration:.2f}s)")
                errors += 1
                continue
            if duration > max_duration:
                print_warning(f"[{i}/{len(audio_files)}] Pulando (muito longo): {audio_path.name} ({duration:.2f}s)")
                errors += 1
                continue
        
        # Transcrever apenas se não existe no CSV
        print_info(f"[{i}/{len(audio_files)}] Transcrevendo novo: {audio_path.name}")
        text = transcribe_audio(audio_path, model_name, language)
        
        if text is None:
            print_error(f"Falha ao transcrever: {audio_path.name}")
            errors += 1
            continue
        
        # Validar transcrição
        is_valid, error_msg = validate_transcription(text, duration)
        if not is_valid:
            print_warning(f"[{i}/{len(audio_files)}] Transcrição inválida: {audio_path.name} - {error_msg}")
            errors += 1
            continue
        
        # Adicionar entrada
        entries.append((filename, text))
        processed += 1
        print_success(f"Transcrito: {filename} -> '{text[:50]}...'")
    
    # Ordenar por nome do arquivo (ordem numérica: audio0, audio1, audio2, ... audio10, etc.)
    def sort_key(filename):
        """Ordena por número no final do nome (audio0, audio1, audio10, etc.)"""
        import re
        match = re.search(r'(\d+)$', filename)
        if match:
            return (filename[:match.start()], int(match.group(1)))
        return (filename, 0)
    
    entries.sort(key=lambda x: sort_key(x[0]))
    
    # Salvar metadata.csv
    print_header("💾 Salvando metadata.csv")
    
    try:
        with open(metadata_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter='|')
            for filename, text in entries:
                writer.writerow([filename, text])
        
        print_success(f"metadata.csv salvo: {metadata_path}")
        print_info(f"Total de entradas: {len(entries)}")
        print_info(f"Processados: {processed}")
        print_info(f"Pulados (já existentes): {skipped}")
        print_info(f"Erros: {errors}")
        
    except Exception as e:
        print_error(f"Erro ao salvar metadata.csv: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description='Transcreve arquivos de áudio e gera metadata.csv automaticamente',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python transcrever_audios.py skipnho
  python transcrever_audios.py skipnho --model base --language pt
  python transcrever_audios.py skipnho --model large-v2 --language pt --skip-existing
        """
    )
    
    parser.add_argument('streamer', help='Nome do streamer/dataset (ex: skipnho, jonvlogs)')
    parser.add_argument('--model', type=str, default='base',
                       choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
                       help='Modelo Whisper a usar (padrão: base)')
    parser.add_argument('--language', type=str, default='pt',
                       help='Idioma do áudio (pt, en, es, etc. ou "auto" para detectar, padrão: pt)')
    parser.add_argument('--min-duration', type=float, default=1.0,
                       help='Duração mínima do áudio em segundos (padrão: 1.0)')
    parser.add_argument('--max-duration', type=float, default=30.0,
                       help='Duração máxima do áudio em segundos (padrão: 30.0)')
    parser.add_argument('--min-text-length', type=int, default=5,
                       help='Comprimento mínimo do texto (padrão: 5)')
    parser.add_argument('--max-text-length', type=int, default=500,
                       help='Comprimento máximo do texto (padrão: 500)')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Manter transcrições existentes no metadata.csv (padrão: sempre ativo)')
    parser.add_argument('--backup', action='store_true',
                       help='Criar backup do metadata.csv existente')
    
    args = parser.parse_args()
    
    # Caminhos
    base_dir = Path(__file__).parent.parent
    dataset_dir = base_dir / "TTSDataset" / args.streamer
    metadata_path = dataset_dir / "metadata.csv"
    wavs_dir = dataset_dir / "wavs"
    
    print_header(f"🎤 Transcrevendo Áudios: {args.streamer}")
    
    # Validar diretórios
    if not dataset_dir.exists():
        print_error(f"Dataset não encontrado: {dataset_dir}")
        sys.exit(1)
    
    if not wavs_dir.exists():
        print_error(f"Diretório wavs não encontrado: {wavs_dir}")
        sys.exit(1)
    
    # Backup do metadata existente
    if args.backup and metadata_path.exists():
        backup_path = dataset_dir / f"metadata_backup_{Path(metadata_path).stat().st_mtime}.csv"
        import shutil
        shutil.copy2(metadata_path, backup_path)
        print_success(f"Backup criado: {backup_path}")
    
    # Obter arquivos de áudio
    audio_files = get_audio_files(wavs_dir)
    
    if not audio_files:
        print_error(f"Nenhum arquivo de áudio encontrado em: {wavs_dir}")
        print_info("Formatos suportados: .wav, .mp3, .flac, .ogg")
        sys.exit(1)
    
    print_info(f"Encontrados {len(audio_files)} arquivo(s) de áudio")
    print_info(f"Modelo Whisper: {args.model}")
    print_info(f"Idioma: {args.language}")
    print_info(f"Duração: {args.min_duration}-{args.max_duration} segundos")
    print_info(f"Texto: {args.min_text_length}-{args.max_text_length} caracteres")
    
    # Gerar metadata.csv
    generate_metadata_csv(
        audio_files=audio_files,
        wavs_dir=wavs_dir,
        metadata_path=metadata_path,
        model_name=args.model,
        language=args.language,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        min_text_length=args.min_text_length,
        max_text_length=args.max_text_length,
        skip_existing=args.skip_existing
    )
    
    print_header("✅ Transcrição Concluída!")
    print_success(f"metadata.csv gerado: {metadata_path}")
    print_info("Próximos passos:")
    print_info("  1. Revisar o metadata.csv gerado")
    print_info("  2. Corrigir transcrições se necessário")
    print_info("  3. Validar dataset: python scripts/validar_dataset.py " + args.streamer)

if __name__ == "__main__":
    main()

