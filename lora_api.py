"""
LoRA Analyzer API REST
API con FastAPI para analizar archivos LoRA programáticamente
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import tempfile
import shutil
from pathlib import Path
import numpy as np
from lora_analyzer import LoRAAnalyzer


def convert_numpy_types(obj):
    """Convierte tipos numpy a tipos nativos de Python para serialización JSON"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, set):
        return list(obj)
    else:
        return obj


# Crear la aplicación FastAPI
app = FastAPI(
    title="LoRA Analyzer API",
    description="API REST para analizar archivos LoRA (.safetensors, .pt, .ckpt)",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "name": "LoRA Analyzer API",
        "version": "1.0.0",
        "endpoints": {
            "GET /": "Esta información",
            "POST /analyze": "Analiza un archivo LoRA",
            "POST /analyze/batch": "Analiza múltiples archivos LoRA",
            "POST /compare": "Compara múltiples archivos LoRA",
            "GET /health": "Estado de salud del servicio",
            "GET /docs": "Documentación interactiva (Swagger)"
        },
        "supported_formats": [".safetensors", ".pt", ".pth", ".ckpt"],
        "documentation": "/docs"
    }


@app.get("/health")
async def health_check():
    """Verifica el estado del servicio"""
    return {
        "status": "healthy",
        "service": "lora-analyzer-api",
        "version": "1.0.0"
    }


@app.post("/analyze")
async def analyze_lora(file: UploadFile = File(...)):
    """
    Analiza un único archivo LoRA
    
    Args:
        file: Archivo LoRA (.safetensors, .pt, .pth, .ckpt)
    
    Returns:
        JSON con análisis completo del LoRA
    """
    # Validar extensión
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in [".safetensors", ".pt", ".pth", ".ckpt"]:
        raise HTTPException(
            status_code=400,
            detail=f"Formato no soportado: {file_ext}. Use .safetensors, .pt, .pth o .ckpt"
        )
    
    # Guardar archivo temporal
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        # Analizar el archivo
        analyzer = LoRAAnalyzer(temp_path)
        analysis = analyzer.analyze()
        
        # Convertir tipos numpy a tipos nativos de Python
        analysis = convert_numpy_types(analysis)
        
        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "analysis": analysis
        })
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al analizar el archivo: {str(e)}"
        )
    
    finally:
        # Limpiar archivo temporal
        if temp_file:
            try:
                Path(temp_path).unlink()
            except:
                pass


@app.post("/analyze/batch")
async def analyze_batch(files: List[UploadFile] = File(...)):
    """
    Analiza múltiples archivos LoRA
    
    Args:
        files: Lista de archivos LoRA
    
    Returns:
        JSON con análisis de todos los archivos
    """
    if not files:
        raise HTTPException(status_code=400, detail="No se proporcionaron archivos")
    
    results = []
    errors = []
    
    for file in files:
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in [".safetensors", ".pt", ".pth", ".ckpt"]:
            errors.append({
                "filename": file.filename,
                "error": f"Formato no soportado: {file_ext}"
            })
            continue
        
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                shutil.copyfileobj(file.file, temp_file)
                temp_path = temp_file.name
            
            analyzer = LoRAAnalyzer(temp_path)
            analysis = analyzer.analyze()
            
            # Convertir tipos numpy
            analysis = convert_numpy_types(analysis)
            
            results.append({
                "filename": file.filename,
                "analysis": analysis
            })
        
        except Exception as e:
            errors.append({
                "filename": file.filename,
                "error": str(e)
            })
        
        finally:
            if temp_file:
                try:
                    Path(temp_path).unlink()
                except:
                    pass
    
    return JSONResponse(content={
        "success": True,
        "total_files": len(files),
        "analyzed": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors
    })


@app.post("/compare")
async def compare_loras(files: List[UploadFile] = File(...)):
    """
    Compara múltiples archivos LoRA
    
    Args:
        files: Lista de al menos 2 archivos LoRA
    
    Returns:
        JSON con comparación detallada
    """
    if len(files) < 2:
        raise HTTPException(
            status_code=400,
            detail="Se requieren al menos 2 archivos para comparar"
        )
    
    analyses = []
    temp_files = []
    
    try:
        # Analizar todos los archivos
        for file in files:
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in [".safetensors", ".pt", ".pth", ".ckpt"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Formato no soportado: {file_ext}"
                )
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                shutil.copyfileobj(file.file, temp_file)
                temp_path = temp_file.name
                temp_files.append(temp_path)
            
            analyzer = LoRAAnalyzer(temp_path)
            analysis = analyzer.analyze()
            
            # Convertir tipos numpy
            analysis = convert_numpy_types(analysis)
            
            analyses.append({
                "filename": file.filename,
                "analysis": analysis
            })
        
        # Crear comparación
        comparison = create_comparison(analyses)
        
        return JSONResponse(content={
            "success": True,
            "total_files": len(files),
            "comparison": comparison
        })
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al comparar archivos: {str(e)}"
        )
    
    finally:
        # Limpiar archivos temporales
        for temp_path in temp_files:
            try:
                Path(temp_path).unlink()
            except:
                pass


def create_comparison(analyses):
    """Crea un objeto de comparación estructurado"""
    comparison = {
        "files": [],
        "comparison_table": {}
    }
    
    # Información de cada archivo
    for item in analyses:
        analysis = item["analysis"]
        comparison["files"].append({
            "filename": item["filename"],
            "size_mb": analysis["file_info"].get("tamaño_mb", 0),
            "format": analysis["file_info"].get("extension", "unknown")
        })
    
    # Tabla de comparación
    features = [
        ("size_mb", "Tamaño (MB)", lambda a: a["file_info"].get("tamaño_mb", "N/A")),
        ("total_layers", "Total capas", lambda a: a.get("architecture", {}).get("total_layers", "N/A")),
        ("rank", "Rank", lambda a: a.get("architecture", {}).get("rank_info", {}).get("most_common_rank", "N/A")),
        ("base_model", "Modelo base", lambda a: a.get("metadata", {}).get("ss_base_model", "N/A")),
        ("train_images", "Imágenes entreno", lambda a: a.get("metadata", {}).get("ss_num_train_images", "N/A")),
        ("learning_rate", "Learning rate", lambda a: a.get("metadata", {}).get("ss_learning_rate", "N/A")),
        ("epochs", "Épocas", lambda a: a.get("metadata", {}).get("ss_num_epochs", "N/A")),
    ]
    
    for key, label, getter in features:
        comparison["comparison_table"][key] = {
            "label": label,
            "values": [getter(item["analysis"]) for item in analyses]
        }
    
    return comparison


# Agregar documentación personalizada
@app.get("/examples")
async def api_examples():
    """Ejemplos de uso de la API"""
    return {
        "curl_examples": {
            "analyze_single": """
curl -X POST "http://localhost:8000/analyze" \\
  -H "accept: application/json" \\
  -H "Content-Type: multipart/form-data" \\
  -F "file=@mi_lora.safetensors"
            """,
            "analyze_batch": """
curl -X POST "http://localhost:8000/analyze/batch" \\
  -H "accept: application/json" \\
  -H "Content-Type: multipart/form-data" \\
  -F "files=@lora1.safetensors" \\
  -F "files=@lora2.pt"
            """,
            "compare": """
curl -X POST "http://localhost:8000/compare" \\
  -H "accept: application/json" \\
  -H "Content-Type: multipart/form-data" \\
  -F "files=@lora1.safetensors" \\
  -F "files=@lora2.safetensors"
            """
        },
        "python_example": """
import requests

# Analizar un archivo
with open('mi_lora.safetensors', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/analyze',
        files={'file': f}
    )
    print(response.json())

# Comparar múltiples archivos
files = [
    ('files', open('lora1.safetensors', 'rb')),
    ('files', open('lora2.safetensors', 'rb'))
]
response = requests.post(
    'http://localhost:8000/compare',
    files=files
)
print(response.json())
        """,
        "javascript_example": """
// Usando fetch API
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/analyze', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => console.log(data));
        """
    }


if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Iniciando LoRA Analyzer API...")
    print("📡 API disponible en: http://localhost:8000")
    print("📚 Documentación: http://localhost:8000/docs")
    print("🔍 Ejemplos: http://localhost:8000/examples")
    print("🛑 Presiona Ctrl+C para detener el servidor")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
