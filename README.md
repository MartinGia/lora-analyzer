# 🔍 LoRA Analyzer

Herramienta completa para analizar archivos LoRA (Low-Rank Adaptation) y descubrir cómo fueron entrenados.

## 🌟 Características

- ✅ **Análisis completo** de arquitectura LoRA (rank, alpha, capas)
- ✅ **Extracción de metadatos** de entrenamiento
- ✅ **Estadísticas de pesos** y distribuciones
- ✅ **Comparación** de múltiples LoRAs
- ✅ **Recomendaciones** para optimización
- ✅ **3 interfaces diferentes**: CLI, Web App y API REST

## 📁 Formatos Soportados

- `.safetensors` (recomendado)
- `.pt` / `.pth` (PyTorch)
- `.ckpt` (Checkpoints)

## 🚀 Instalación Rápida

```bash
# Clonar o descargar los archivos
cd lora-analyzer

# Instalar dependencias
pip install -r requirements.txt

# ¡Listo para usar!
```

### Instalación opcional de PyTorch

Si no tienes PyTorch instalado, usa:

```bash
# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu

# CUDA (GPU)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## 💻 Uso

### 1️⃣ CLI (Línea de Comandos)

La forma más rápida para análisis desde terminal:

```bash
# Analizar un archivo
python lora_cli.py mi_lora.safetensors

# Analizar múltiples archivos
python lora_cli.py lora1.safetensors lora2.pt lora3.ckpt

# Guardar resultado en JSON
python lora_cli.py mi_lora.safetensors --output resultado.json

# Comparar múltiples LoRAs
python lora_cli.py lora1.safetensors lora2.safetensors --compare

# Modo verbose
python lora_cli.py mi_lora.safetensors --verbose

# Ver ayuda
python lora_cli.py --help
```

**Ejemplo de salida:**
```
🔍 Analizando: my_style_lora.safetensors
======================================================================
REPORTE DE ANÁLISIS LORA
======================================================================

📁 INFORMACIÓN DEL ARCHIVO:
  Nombre: my_style_lora.safetensors
  Tamaño: 144.2 MB
  Formato: .safetensors

🏗️  ARQUITECTURA:
  Total de capas: 192
  Rank más común: 32
  Rango de ranks: 32 - 32

⚙️  METADATOS DE ENTRENAMIENTO:
  Modelo base: sd_xl_base_1.0.safetensors
  Network dim (rank): 32
  Alpha: 32
  Learning rate: 0.0001
  Épocas: 10
  Imágenes de entrenamiento: 45
  Batch size: 1
  Resolución: 1024

💡 RECOMENDACIONES:
  1. Rank óptimo (32): Buen balance entre detalle y eficiencia.
  2. Dataset mediano (45 imágenes): Bueno para conceptos específicos.
```

### 2️⃣ Web App (Interfaz Gráfica)

Interfaz visual con Gradio - ideal para uso interactivo:

```bash
# Iniciar la aplicación web
python lora_webapp.py
```

Luego abre tu navegador en: **http://localhost:7860**

**Funcionalidades:**
- 📤 Arrastrar y soltar archivos
- 📊 Vista de análisis con resumen visual
- 📈 Datos JSON completos
- ⚖️ Comparación lado a lado
- 💾 Exportar resultados

![Web App Screenshot](https://via.placeholder.com/800x400?text=Web+App+Interface)

### 3️⃣ API REST

Servidor FastAPI para integración programática:

```bash
# Iniciar el servidor API
python lora_api.py
```

El servidor estará disponible en: **http://localhost:8000**

**Documentación interactiva:** http://localhost:8000/docs

#### Endpoints disponibles:

##### `POST /analyze` - Analizar un archivo

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@mi_lora.safetensors"
```

```python
import requests

with open('mi_lora.safetensors', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/analyze',
        files={'file': f}
    )
    result = response.json()
    print(result)
```

##### `POST /analyze/batch` - Analizar múltiples archivos

```python
import requests

files = [
    ('files', open('lora1.safetensors', 'rb')),
    ('files', open('lora2.safetensors', 'rb'))
]
response = requests.post(
    'http://localhost:8000/analyze/batch',
    files=files
)
print(response.json())
```

##### `POST /compare` - Comparar archivos

```bash
curl -X POST "http://localhost:8000/compare" \
  -F "files=@lora1.safetensors" \
  -F "files=@lora2.safetensors"
```

##### `GET /health` - Estado del servicio

```bash
curl http://localhost:8000/health
```

##### `GET /examples` - Ver más ejemplos

```bash
curl http://localhost:8000/examples
```

## 📊 Información que Puedes Extraer

### ✅ Lo que SÍ puedes obtener:

- **Arquitectura completa**
  - Dimensión de rank (8, 16, 32, 64, 128, etc.)
  - Factor alpha de escalado
  - Capas modificadas (attention blocks, MLP, etc.)
  - Total de parámetros

- **Metadatos de entrenamiento** (si están incluidos)
  - Modelo base utilizado (SD 1.5, SDXL, etc.)
  - Learning rate
  - Número de epochs
  - Batch size
  - Resolución de entrenamiento
  - Número de imágenes de entrenamiento
  - Herramienta usada (Kohya, EveryDream, etc.)

- **Estadísticas de pesos**
  - Distribución de valores
  - Media y desviación estándar
  - Intensidad por capa

### ❌ Lo que NO puedes recuperar:

- **Los datos de entrenamiento originales** - Técnicamente imposible
- **Imágenes específicas del dataset** - No están almacenadas en el LoRA
- **Prompts exactos usados** - Solo si están en metadatos

## 🎯 Casos de Uso

### 1. Ingeniería Inversa
```bash
# Analiza un LoRA público para aprender de él
python lora_cli.py awesome_public_lora.safetensors

# Compara con tu propio LoRA
python lora_cli.py awesome_public_lora.safetensors my_lora.safetensors --compare
```

### 2. Optimización de tus LoRAs
```bash
# Analiza diferentes versiones para encontrar la mejor configuración
python lora_cli.py lora_v1_rank16.safetensors lora_v2_rank32.safetensors \
                   lora_v3_rank64.safetensors --compare
```

### 3. Debugging
```bash
# Verifica que tu LoRA se entrenó correctamente
python lora_cli.py my_new_lora.safetensors --verbose
```

### 4. Investigación
```python
# Analiza múltiples LoRAs programáticamente
import requests
import os

for filename in os.listdir('loras/'):
    if filename.endswith('.safetensors'):
        with open(f'loras/{filename}', 'rb') as f:
            response = requests.post(
                'http://localhost:8000/analyze',
                files={'file': f}
            )
            result = response.json()
            # Procesar resultados...
```

## 🔧 Uso Programático

### Como módulo Python

```python
from lora_analyzer import LoRAAnalyzer, format_analysis_report

# Analizar un archivo
analyzer = LoRAAnalyzer('mi_lora.safetensors')
analysis = analyzer.analyze()

# Ver reporte formateado
print(format_analysis_report(analysis))

# Acceder a datos específicos
rank = analysis['architecture']['rank_info']['most_common_rank']
print(f"Rank detectado: {rank}")

# Extraer metadatos
metadata = analysis['metadata']
learning_rate = metadata.get('ss_learning_rate', 'N/A')
print(f"Learning rate: {learning_rate}")
```

## 📦 Estructura del Proyecto

```
lora-analyzer/
├── lora_analyzer.py      # Módulo core de análisis
├── lora_cli.py          # Aplicación CLI
├── lora_webapp.py       # Aplicación Web (Gradio)
├── lora_api.py          # API REST (FastAPI)
├── requirements.txt     # Dependencias
└── README.md           # Esta documentación
```

## 🛠️ Requisitos del Sistema

- Python 3.8 o superior
- 4GB RAM mínimo (8GB recomendado)
- Espacio en disco: ~2GB para dependencias

## ⚙️ Configuración Avanzada

### Cambiar puertos

**Web App:**
```python
# En lora_webapp.py, línea 249
app.launch(server_port=8080)  # Cambiar de 7860 a 8080
```

**API:**
```python
# En lora_api.py, línea 360
uvicorn.run(app, host="0.0.0.0", port=9000)  # Cambiar de 8000 a 9000
```

### Análisis de archivos grandes

Para LoRAs muy grandes (>1GB):

```python
# Aumentar límite de análisis de capas
analyzer = LoRAAnalyzer('huge_lora.safetensors')
# Modifica _analyze_weights para analizar más capas
```

## 🤝 Contribuciones

Las contribuciones son bienvenidas! Para reportar bugs o sugerir features, abre un issue.

## 📝 Notas Importantes

1. **Privacidad**: Todo el análisis se hace localmente. No se envía información a servidores externos.

2. **Rendimiento**: El análisis de archivos grandes puede tomar tiempo. La CLI es más rápida que la Web App.

3. **Compatibilidad**: Diseñado principalmente para LoRAs de Stable Diffusion, pero puede funcionar con otros tipos.

4. **Metadatos**: La cantidad de información disponible depende de cómo fue guardado el LoRA. LoRAs entrenados con Kohya suelen tener más metadatos.

## 🔮 Roadmap

- [ ] Soporte para análisis de LoRAs de LLMs
- [ ] Visualización de distribución de pesos
- [ ] Detección automática de estilo/concepto
- [ ] Base de datos para indexar LoRAs analizados
- [ ] Exportar reportes en PDF
- [ ] Integración con Hugging Face Hub

## 📄 Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.

## ❓ FAQ

**P: ¿Puedo recuperar las imágenes con las que se entrenó el LoRA?**
R: No, es técnicamente imposible. Los pesos del modelo son el resultado de la optimización, pero no contienen los datos originales.

**P: ¿Funciona con cualquier tipo de LoRA?**
R: Está optimizado para LoRAs de Stable Diffusion, pero puede analizar cualquier LoRA en formato safetensors o PyTorch.

**P: ¿Por qué algunos metadatos no aparecen?**
R: Depende de cómo fue guardado el archivo. Usa herramientas como Kohya que incluyen metadatos extensos.

**P: ¿Puedo usar esto para crear LoRAs mejores?**
R: Sí! Analiza LoRAs exitosos para aprender qué configuraciones funcionan mejor para diferentes casos de uso.

**P: ¿Es seguro analizar LoRAs de fuentes desconocidas?**
R: El análisis es seguro, pero ten cuidado al cargar pesos en un modelo. Los archivos .pt y .ckpt pueden contener código malicioso.

## 🙏 Agradecimientos

Desarrollado para la comunidad de ML y entusiastas de LoRA.

---

**¿Preguntas o problemas?** Abre un issue en el repositorio.

**¿Te resultó útil?** ¡Dale una estrella ⭐!
