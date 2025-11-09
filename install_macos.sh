#!/bin/bash

# LoRA Analyzer - Instalación para macOS
# Script automático que detecta y configura todo

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║         🔍 LoRA Analyzer - Instalación para macOS                    ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Detectar si estamos en macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ Este script es solo para macOS"
    echo "   Usa quick_start.sh en su lugar"
    exit 1
fi

echo "✅ Sistema operativo: macOS detectado"
echo ""

# Función para verificar comando
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 1. Verificar Python3
echo "📋 Paso 1: Verificando Python..."
if command_exists python3; then
    PYTHON_VERSION=$(python3 --version)
    echo "✅ Python encontrado: $PYTHON_VERSION"
    PYTHON_CMD="python3"
else
    echo "❌ Python 3 no está instalado"
    echo ""
    echo "Por favor instala Python de una de estas formas:"
    echo ""
    echo "Opción A - Con Homebrew (recomendado):"
    echo "  1. Instala Homebrew:"
    echo "     /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    echo "  2. Instala Python:"
    echo "     brew install python"
    echo ""
    echo "Opción B - Descarga directa:"
    echo "  1. Ve a: https://www.python.org/downloads/"
    echo "  2. Descarga e instala Python 3.11 o superior"
    echo ""
    exit 1
fi

# 2. Verificar pip3
echo ""
echo "📋 Paso 2: Verificando pip..."
if command_exists pip3; then
    PIP_VERSION=$(pip3 --version)
    echo "✅ pip encontrado: $PIP_VERSION"
    PIP_CMD="pip3"
elif command_exists pip; then
    PIP_VERSION=$(pip --version)
    echo "✅ pip encontrado: $PIP_VERSION"
    PIP_CMD="pip"
else
    echo "❌ pip no está instalado"
    echo ""
    echo "Instalando pip..."
    $PYTHON_CMD -m ensurepip --upgrade
    
    if command_exists pip3; then
        echo "✅ pip instalado correctamente"
        PIP_CMD="pip3"
    else
        echo "❌ No se pudo instalar pip automáticamente"
        echo "   Instala pip manualmente: $PYTHON_CMD -m ensurepip"
        exit 1
    fi
fi

# 3. Crear entorno virtual
echo ""
echo "📋 Paso 3: Configurando entorno virtual..."
if [ -d "venv" ]; then
    echo "✅ Entorno virtual ya existe"
else
    echo "Creando entorno virtual..."
    $PYTHON_CMD -m venv venv
    if [ $? -eq 0 ]; then
        echo "✅ Entorno virtual creado"
    else
        echo "⚠️  No se pudo crear entorno virtual"
        echo "   Continuando sin entorno virtual..."
    fi
fi

# 4. Activar entorno virtual
if [ -d "venv" ]; then
    echo ""
    echo "🔌 Activando entorno virtual..."
    source venv/bin/activate
    echo "✅ Entorno virtual activado"
    
    # Actualizar pip en el entorno virtual
    pip install --upgrade pip --quiet
fi

# 5. Instalar dependencias
echo ""
echo "📋 Paso 4: Instalando dependencias..."
echo "Esto puede tomar algunos minutos..."
echo ""

$PIP_CMD install -r requirements.txt

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Dependencias instaladas correctamente"
else
    echo ""
    echo "❌ Hubo un error instalando las dependencias"
    echo "   Intenta manualmente: $PIP_CMD install -r requirements.txt"
    exit 1
fi

# 6. Verificar instalación
echo ""
echo "📋 Paso 5: Verificando instalación..."
echo ""
$PYTHON_CMD test_installation.py

# 7. Instrucciones finales
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                    ✅ ¡INSTALACIÓN COMPLETADA!                        ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "🚀 Para usar la herramienta:"
echo ""
echo "1. Activa el entorno virtual (si lo creaste):"
echo "   source venv/bin/activate"
echo ""
echo "2. Ejecuta la aplicación que necesites:"
echo ""
echo "   📱 Web App (interfaz gráfica):"
echo "   $PYTHON_CMD lora_webapp.py"
echo "   Luego abre: http://localhost:7860"
echo ""
echo "   💻 CLI (línea de comandos):"
echo "   $PYTHON_CMD lora_cli.py mi_lora.safetensors"
echo ""
echo "   🔌 API REST:"
echo "   $PYTHON_CMD lora_api.py"
echo "   Luego abre: http://localhost:8000/docs"
echo ""
echo "📚 Lee las guías:"
echo "   • LEEME_PRIMERO.txt - Introducción"
echo "   • GUIA_WEBAPP.txt - Guía de la Web App"
echo "   • README.md - Documentación completa"
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
