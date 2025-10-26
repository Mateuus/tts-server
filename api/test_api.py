#!/usr/bin/env python3
"""
Testes para a API de Clonagem de Voz
"""

import requests
import json

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
    
    test_health()
    test_generate()
    test_list()
    
    print("\n✅ Testes concluídos!")

