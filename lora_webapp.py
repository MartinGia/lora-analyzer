"""
LoRA Analyzer Web App
Interfaz web con Gradio para analizar archivos LoRA
Versión mejorada con interfaz en español
"""

import gradio as gr
import json
from pathlib import Path
import tempfile
import numpy as np
from lora_analyzer import LoRAAnalyzer, format_analysis_report


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


def analyze_lora_file(file):
    """Analiza un archivo LoRA subido por el usuario"""
    if file is None:
        return "❌ Por favor sube un archivo LoRA primero", "{}", "⚠️ Sin archivo para analizar"
    
    try:
        # Analizar el archivo
        analyzer = LoRAAnalyzer(file.name)
        analysis = analyzer.analyze()
        
        # Convertir tipos numpy a tipos nativos de Python
        analysis = convert_numpy_types(analysis)
        
        # Formato de reporte legible
        report = format_analysis_report(analysis)
        
        # JSON formateado
        json_output = json.dumps(analysis, indent=2, ensure_ascii=False)
        
        # Información clave para la UI
        summary = generate_summary(analysis)
        
        return report, json_output, summary
    
    except Exception as e:
        error_msg = f"❌ Error al analizar el archivo: {str(e)}\n\n💡 Verifica que sea un archivo LoRA válido (.safetensors, .pt, .ckpt)"
        return error_msg, "{}", error_msg


def generate_summary(analysis):
    """Genera un resumen visual de la información clave"""
    summary_parts = []
    
    # Información del archivo
    if "file_info" in analysis:
        info = analysis["file_info"]
        summary_parts.append(f"📁 **{info.get('nombre', 'N/A')}**")
        summary_parts.append(f"💾 Tamaño: {info.get('tamaño_mb', 0)} MB")
    
    # Arquitectura
    if "architecture" in analysis:
        arch = analysis["architecture"]
        summary_parts.append(f"🏗️ Capas: {arch.get('total_layers', 0)}")
        
        if "rank_info" in arch and arch["rank_info"]:
            rank = arch["rank_info"].get("most_common_rank", "N/A")
            summary_parts.append(f"📊 Rank: {rank}")
    
    # Metadatos clave
    if "metadata" in analysis:
        meta = analysis["metadata"]
        
        if "ss_base_model" in meta:
            model = str(meta["ss_base_model"])
            if len(model) > 50:
                model = model[:47] + "..."
            summary_parts.append(f"🤖 Modelo base: {model}")
        
        if "ss_num_train_images" in meta:
            summary_parts.append(f"🖼️ Imágenes entrenamiento: {meta['ss_num_train_images']}")
        
        if "ss_learning_rate" in meta:
            summary_parts.append(f"📈 Learning rate: {meta['ss_learning_rate']}")
    
    # Recomendaciones
    if "recommendations" in analysis and analysis["recommendations"]:
        summary_parts.append("\n**💡 Recomendaciones principales:**")
        for i, rec in enumerate(analysis["recommendations"][:3], 1):
            summary_parts.append(f"{i}. {rec}")
    
    return "\n\n".join(summary_parts) if summary_parts else "No hay información disponible"


def compare_loras(files):
    """Compara múltiples archivos LoRA"""
    if not files or len(files) < 2:
        return "❌ Por favor sube al menos 2 archivos LoRA para comparar"
    
    try:
        results = []
        for file in files:
            analyzer = LoRAAnalyzer(file.name)
            analysis = analyzer.analyze()
            # Convertir tipos numpy
            analysis = convert_numpy_types(analysis)
            results.append(analysis)
        
        # Crear tabla comparativa
        comparison = create_comparison_table(results)
        return comparison
    
    except Exception as e:
        return f"❌ Error al comparar archivos: {str(e)}"


def create_comparison_table(results):
    """Crea una tabla comparativa de múltiples LoRAs"""
    if not results:
        return "No hay resultados para comparar"
    
    table = "# 📊 Comparación de LoRAs\n\n"
    
    # Nombres de archivos
    table += "| Característica | " + " | ".join(r["file_info"]["nombre"] for r in results) + " |\n"
    table += "|" + "---|" * (len(results) + 1) + "\n"
    
    # Filas comparativas
    rows = [
        ("Tamaño (MB)", lambda r: f"{r['file_info'].get('tamaño_mb', 'N/A')}"),
        ("Formato", lambda r: r['file_info'].get('extension', 'N/A')),
        ("Total capas", lambda r: r.get('architecture', {}).get('total_layers', 'N/A')),
        ("Rank", lambda r: r.get('architecture', {}).get('rank_info', {}).get('most_common_rank', 'N/A')),
        ("Modelo base", lambda r: str(r.get('metadata', {}).get('ss_base_model', 'N/A'))[:30]),
        ("Imágenes entreno", lambda r: r.get('metadata', {}).get('ss_num_train_images', 'N/A')),
        ("Learning rate", lambda r: r.get('metadata', {}).get('ss_learning_rate', 'N/A')),
        ("Epochs", lambda r: r.get('metadata', {}).get('ss_num_epochs', 'N/A')),
    ]
    
    for label, getter in rows:
        row = f"| {label} | "
        row += " | ".join(str(getter(r)) for r in results)
        row += " |\n"
        table += row
    
    return table


# Crear la interfaz Gradio
with gr.Blocks(title="LoRA Analyzer", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 🔍 LoRA Analyzer - Analizador de Archivos LoRA
    ### Descubre la arquitectura, metadatos y configuración de entrenamiento de tus LoRAs
    
    **📁 Formatos soportados:** `.safetensors` (recomendado), `.pt`, `.pth`, `.ckpt`
    
    **💡 Tip rápido:** Arrastra tu archivo LoRA en la zona de carga y haz clic en "🔍 Analizar"
    """)
    
    with gr.Tabs():
        # Tab 1: Análisis individual
        with gr.Tab("📄 Analizar LoRA"):
            gr.Markdown("### Sube un archivo LoRA para análisis completo")
            
            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(
                        label="Arrastra tu archivo LoRA aquí",
                        file_types=[".safetensors", ".pt", ".pth", ".ckpt"]
                    )
                    analyze_btn = gr.Button("🔍 Analizar", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    summary_output = gr.Markdown(label="Resumen")
            
            with gr.Row():
                with gr.Column():
                    report_output = gr.Textbox(
                        label="Reporte Completo",
                        lines=20,
                        max_lines=30
                    )
                
                with gr.Column():
                    json_output = gr.Code(
                        label="Datos JSON",
                        language="json",
                        lines=20
                    )
            
            analyze_btn.click(
                fn=analyze_lora_file,
                inputs=[file_input],
                outputs=[report_output, json_output, summary_output]
            )
        
        # Tab 2: Comparación
        with gr.Tab("📊 Comparar LoRAs"):
            gr.Markdown("### Sube 2 o más archivos LoRA para comparar")
            
            files_input = gr.File(
                label="Arrastra múltiples archivos LoRA aquí",
                file_count="multiple",
                file_types=[".safetensors", ".pt", ".pth", ".ckpt"]
            )
            compare_btn = gr.Button("⚖️ Comparar", variant="primary", size="lg")
            
            comparison_output = gr.Markdown(label="Comparación")
            
            compare_btn.click(
                fn=compare_loras,
                inputs=[files_input],
                outputs=[comparison_output]
            )
        
        # Tab 3: Información y ayuda
        with gr.Tab("📖 Cómo Usar"):
            gr.Markdown("""
            ## 🚀 Guía Rápida de Uso
            
            ### Paso 1: Consigue un archivo LoRA
            - Descarga LoRAs de [CivitAI](https://civitai.com) o [Hugging Face](https://huggingface.co)
            - O usa tus propios LoRAs entrenados
            - Formatos aceptados: `.safetensors`, `.pt`, `.pth`, `.ckpt`
            
            ### Paso 2: Analiza un LoRA
            1. Ve a la pestaña **"📄 Analizar LoRA"**
            2. Arrastra tu archivo a la zona de carga (o haz clic para seleccionar)
            3. Haz clic en el botón **"🔍 Analizar"**
            4. ¡Listo! Verás el análisis completo
            
            ### Paso 3: Lee los resultados
            - **Panel izquierdo (Resumen)**: Información clave visual
            - **Panel central (Reporte)**: Análisis completo en texto
            - **Panel derecho (JSON)**: Datos técnicos en formato JSON
            
            ### Paso 4: Compara LoRAs (opcional)
            1. Ve a la pestaña **"📊 Comparar LoRAs"**
            2. Sube 2 o más archivos LoRA
            3. Haz clic en **"⚖️ Comparar"**
            4. Verás una tabla comparativa
            
            ---
            
            ## 📊 ¿Qué información obtienes?
            
            ### ✅ Arquitectura del LoRA
            - **Rank**: Dimensión de las matrices (8, 16, 32, 64, 128...)
            - **Alpha**: Factor de escalado
            - **Capas**: Número total de capas modificadas
            - **Parámetros**: Total de parámetros entrenables
            
            ### ✅ Metadatos de Entrenamiento
            - **Modelo base**: SD 1.5, SDXL, etc.
            - **Learning rate**: Tasa de aprendizaje
            - **Epochs**: Número de épocas
            - **Batch size**: Tamaño del lote
            - **Resolución**: Resolución de entrenamiento
            - **Dataset**: Número de imágenes usadas
            
            ### ✅ Recomendaciones
            - Sugerencias para optimizar tus LoRAs
            - Comparación con mejores prácticas
            - Ideas para mejorar resultados
            
            ---
            
            ## 💡 Casos de Uso
            
            ### 🔍 Ingeniería Inversa
            Analiza LoRAs públicos exitosos para aprender:
            - ¿Qué rank utilizaron?
            - ¿Cuántas imágenes necesitaron?
            - ¿Qué learning rate funcionó?
            
            ### 🎯 Optimización
            Compara diferentes versiones de tu LoRA:
            - Encuentra la configuración óptima
            - Identifica qué cambios mejoraron resultados
            - Ahorra tiempo en experimentación
            
            ### 🐛 Debugging
            Detecta problemas en tus LoRAs:
            - Verifica que se entrenó correctamente
            - Confirma los parámetros usados
            - Identifica configuraciones incorrectas
            
            ### 📚 Investigación
            Estudia diferentes enfoques:
            - Compara técnicas de entrenamiento
            - Analiza múltiples LoRAs
            - Documenta mejores prácticas
            
            ---
            
            ## ⚠️ Limitaciones Importantes
            
            ### ❌ NO puedes recuperar:
            - **Imágenes originales del dataset** (técnicamente imposible)
            - **Prompts exactos usados** (a menos que estén en metadatos)
            - **Detalles del preprocesamiento** (cómo se limpiaron las imágenes)
            
            ### ⚠️ Dependencias:
            - La cantidad de información depende de cómo fue guardado el LoRA
            - LoRAs de Kohya suelen tener más metadatos
            - Algunos LoRAs antiguos pueden tener información limitada
            
            ---
            
            ## 🛠️ Otras Herramientas Disponibles
            
            Además de esta Web App, también puedes usar:
            
            ### 📟 CLI (Línea de Comandos)
            ```bash
            python lora_cli.py mi_lora.safetensors
            python lora_cli.py lora1.safetensors lora2.safetensors --compare
            ```
            Ideal para: análisis rápido, automatización, scripts
            
            ### 🔌 API REST
            ```bash
            python lora_api.py
            # Visita http://localhost:8000/docs
            ```
            Ideal para: integración con otras apps, procesamiento masivo
            
            ---
            
            ## 📞 Necesitas Ayuda?
            
            1. **Verifica la instalación**: `python test_installation.py`
            2. **Lee el README.md**: Documentación completa
            3. **Revisa examples.py**: Ejemplos de código
            4. **Archivos de prueba**: Descarga LoRAs de ejemplo de CivitAI
            
            ---
            
            ## 🎓 Tips y Mejores Prácticas
            
            ### Para Analizar:
            - Usa archivos `.safetensors` cuando sea posible (más seguros)
            - Verifica que el archivo no esté corrupto
            - Ten paciencia con archivos grandes (>500MB)
            
            ### Para Comparar:
            - Compara LoRAs del mismo tipo (mismo modelo base)
            - Enfócate en diferencias de rank y dataset
            - Usa la comparación para A/B testing
            
            ### Para Aprender:
            - Analiza varios LoRAs populares de tu estilo favorito
            - Documenta qué configuraciones funcionan mejor
            - Experimenta con los parámetros que descubras
            
            ---
            
            **📚 Desarrollado para la comunidad de ML y creadores con IA**
            
            *Versión Web App - LoRA Analyzer v1.0*
            """)

    
    gr.Markdown("""
    ---
    💡 **Tip**: Para obtener mejores resultados, asegúrate de que tus LoRAs incluyan metadatos de entrenamiento.
    """)


if __name__ == "__main__":
    print("🚀 Iniciando LoRA Analyzer Web App...")
    print("📱 Abre tu navegador en: http://localhost:7860")
    print("🛑 Presiona Ctrl+C para detener el servidor")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
