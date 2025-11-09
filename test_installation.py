"""
Test de instalación - LoRA Analyzer
Verifica que todas las dependencias estén instaladas correctamente
"""

import sys


def test_imports():
    """Verifica que todos los módulos necesarios estén disponibles"""
    print("🔍 Verificando instalación...\n")
    
    results = []
    
    # Core dependencies
    tests = [
        ("numpy", "NumPy", "pip install numpy"),
        ("safetensors", "SafeTensors", "pip install safetensors"),
        ("torch", "PyTorch", "pip install torch"),
        ("gradio", "Gradio (Web App)", "pip install gradio"),
        ("fastapi", "FastAPI (API)", "pip install fastapi"),
        ("uvicorn", "Uvicorn (API)", "pip install uvicorn[standard]"),
    ]
    
    for module_name, display_name, install_cmd in tests:
        try:
            __import__(module_name)
            print(f"✅ {display_name:<25} - Instalado")
            results.append(True)
        except ImportError:
            print(f"❌ {display_name:<25} - NO instalado")
            print(f"   Instalar con: {install_cmd}")
            results.append(False)
    
    print("\n" + "=" * 70)
    
    if all(results):
        print("✅ ¡Todas las dependencias están instaladas!")
        print("\n🚀 Puedes usar:")
        print("   • CLI:     python lora_cli.py <archivo>")
        print("   • Web App: python lora_webapp.py")
        print("   • API:     python lora_api.py")
        return True
    else:
        print("⚠️  Faltan algunas dependencias")
        print("\n📦 Para instalar todo:")
        print("   pip install -r requirements.txt")
        return False


def test_analyzer_module():
    """Verifica que el módulo principal funcione"""
    print("\n" + "=" * 70)
    print("🧪 Probando módulo de análisis...\n")
    
    try:
        from lora_analyzer import LoRAAnalyzer, format_analysis_report
        print("✅ Módulo 'lora_analyzer' cargado correctamente")
        
        # Verificar que las clases existen
        assert hasattr(LoRAAnalyzer, 'analyze')
        assert callable(format_analysis_report)
        print("✅ Todas las funciones principales disponibles")
        
        return True
    except Exception as e:
        print(f"❌ Error al cargar el módulo: {str(e)}")
        return False


def display_system_info():
    """Muestra información del sistema"""
    print("\n" + "=" * 70)
    print("💻 Información del sistema:\n")
    
    print(f"Python: {sys.version}")
    print(f"Plataforma: {sys.platform}")
    
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA disponible: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
    except:
        pass
    
    try:
        import numpy
        print(f"NumPy: {numpy.__version__}")
    except:
        pass
    
    try:
        import gradio
        print(f"Gradio: {gradio.__version__}")
    except:
        pass
    
    try:
        import fastapi
        print(f"FastAPI: {fastapi.__version__}")
    except:
        pass


def show_next_steps():
    """Muestra los próximos pasos"""
    print("\n" + "=" * 70)
    print("📚 Próximos pasos:\n")
    
    print("1. Consigue un archivo LoRA (.safetensors, .pt, .ckpt)")
    print("2. Prueba la CLI:")
    print("   python lora_cli.py tu_archivo.safetensors")
    print("\n3. O inicia la Web App:")
    print("   python lora_webapp.py")
    print("\n4. O inicia la API:")
    print("   python lora_api.py")
    print("\n5. Lee el README.md para más ejemplos y documentación")
    print("\n💡 Tip: Puedes descargar LoRAs de ejemplo de:")
    print("   • https://civitai.com")
    print("   • https://huggingface.co")
    print("=" * 70)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("    🔍 LoRA Analyzer - Test de Instalación")
    print("=" * 70 + "\n")
    
    # Ejecutar tests
    deps_ok = test_imports()
    module_ok = test_analyzer_module()
    display_system_info()
    
    if deps_ok and module_ok:
        print("\n" + "=" * 70)
        print("✅ ¡TODO LISTO! La instalación es correcta")
        show_next_steps()
    else:
        print("\n" + "=" * 70)
        print("⚠️  Por favor instala las dependencias faltantes")
        print("\nEjecuta: pip install -r requirements.txt")
        print("=" * 70)
