#!/usr/bin/env python3
"""
Testes para a API de Clonagem de Voz
"""

import requests
import json
import sys

BASE_URL = "http://localhost:8000"

def test_health():
    """Testar health check"""
    print("\n🔍 Testando /health...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")

def test_generate():
    """Testar geração de áudio"""
    print("\n🔍 Testando /generate...")
    
    data = {
        "text": "Este é um teste da API de clonagem de voz",
        "voice_ref": "../../audio/minha_voz.mp3",
        "language": "pt",
        "speed": 0.95
    }
    
    response = requests.post(
        f"{BASE_URL}/generate",
        json=data
    )
    
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    
    if result.get("success"):
        print(f"\n✅ Áudio gerado: {result['filename']}")
        print(f"   Tamanho: {result['size_kb']} KB")
        print(f"   Download: {BASE_URL}/audio/{result['filename']}")

def test_list():
    """Testar listagem de arquivos"""
    print("\n🔍 Testando /list...")
    response = requests.get(f"{BASE_URL}/list")
    print(f"Status: {response.status_code}")
    
    result = response.json()
    print(f"Total de arquivos: {result['count']}")
    
    if result['files']:
        print("\n📁 Arquivos:")
        for file in result['files'][:5]:  # Mostrar apenas 5
            print(f"  - {file['filename']} ({file['size_kb']} KB)")

def test_transcribe(audio_file=None):
    """Testar transcrição de áudio"""
    print("\n🔍 Testando /transcribe...")
    
    if not audio_file:
        print("⚠️  Nenhum arquivo de áudio fornecido")
        print("   Uso: python test_api.py audio.mp3")
        return
    
    import os
    if not os.path.exists(audio_file):
        print(f"❌ Arquivo não encontrado: {audio_file}")
        return
    
    try:
        with open(audio_file, "rb") as f:
            response = requests.post(
                f"{BASE_URL}/transcribe",
                files={"file": f},
                data={"language": "pt"}
            )
        
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        
        if result.get("success"):
            print(f"\n✅ Transcrição: {result['text']}")
            print(f"   Idioma: {result.get('language', 'N/A')}")
            print(f"   Duração: {result.get('duration', 'N/A')} segundos")
    except Exception as e:
        print(f"❌ Erro ao transcrever: {e}")

if __name__ == "__main__":
    print("="*60)
    print("🧪 TESTES DA API")
    print("="*60)
    
    # Verificar se API está rodando
    try:
        requests.get(f"{BASE_URL}/health")
    except:
        print("❌ API não está rodando!")
        print("Execute: python api/app.py")
        exit(1)
    
    # Se foi fornecido um arquivo de áudio como argumento
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        test_transcribe(audio_file)
    else:
        test_health()
        test_generate()
        test_list()
    
    print("\n✅ Testes concluídos!")

