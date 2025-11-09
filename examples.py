"""
Ejemplos de uso del LoRA Analyzer
Demuestra diferentes formas de usar la herramienta
"""

from lora_analyzer import LoRAAnalyzer, format_analysis_report
import json


def example_basic_analysis():
    """Ejemplo básico: analizar un archivo"""
    print("=" * 70)
    print("EJEMPLO 1: Análisis Básico")
    print("=" * 70)
    
    # Reemplaza con tu archivo real
    file_path = "mi_lora.safetensors"
    
    try:
        # Crear analizador
        analyzer = LoRAAnalyzer(file_path)
        
        # Realizar análisis
        analysis = analyzer.analyze()
        
        # Mostrar reporte formateado
        print(format_analysis_report(analysis))
        
    except FileNotFoundError:
        print(f"⚠️  Archivo no encontrado: {file_path}")
        print("   Reemplaza 'mi_lora.safetensors' con tu archivo real")


def example_extract_specific_data():
    """Ejemplo: extraer datos específicos"""
    print("\n" + "=" * 70)
    print("EJEMPLO 2: Extraer Datos Específicos")
    print("=" * 70)
    
    file_path = "mi_lora.safetensors"
    
    try:
        analyzer = LoRAAnalyzer(file_path)
        analysis = analyzer.analyze()
        
        # Extraer información específica
        print("\n📊 Información clave:")
        
        # Rank
        rank_info = analysis.get('architecture', {}).get('rank_info', {})
        if rank_info:
            rank = rank_info.get('most_common_rank', 'N/A')
            print(f"  • Rank: {rank}")
        
        # Modelo base
        metadata = analysis.get('metadata', {})
        if metadata:
            base_model = metadata.get('ss_base_model', 'N/A')
            print(f"  • Modelo base: {base_model}")
            
            # Learning rate
            lr = metadata.get('ss_learning_rate', 'N/A')
            print(f"  • Learning rate: {lr}")
            
            # Imágenes de entrenamiento
            num_images = metadata.get('ss_num_train_images', 'N/A')
            print(f"  • Imágenes: {num_images}")
            
            # Epochs
            epochs = metadata.get('ss_num_epochs', 'N/A')
            print(f"  • Epochs: {epochs}")
        
        # Tamaño del archivo
        file_size = analysis.get('file_info', {}).get('tamaño_mb', 'N/A')
        print(f"  • Tamaño: {file_size} MB")
        
    except FileNotFoundError:
        print(f"⚠️  Archivo no encontrado: {file_path}")


def example_compare_loras():
    """Ejemplo: comparar múltiples LoRAs"""
    print("\n" + "=" * 70)
    print("EJEMPLO 3: Comparar Múltiples LoRAs")
    print("=" * 70)
    
    # Reemplaza con tus archivos reales
    files = [
        "lora_v1.safetensors",
        "lora_v2.safetensors",
        "lora_v3.safetensors"
    ]
    
    results = []
    
    for file_path in files:
        try:
            analyzer = LoRAAnalyzer(file_path)
            analysis = analyzer.analyze()
            results.append({
                'file': file_path,
                'analysis': analysis
            })
        except FileNotFoundError:
            print(f"⚠️  Archivo no encontrado: {file_path}")
    
    if results:
        print("\n📊 Comparación:")
        print(f"{'Archivo':<30} {'Rank':<10} {'Tamaño (MB)':<15} {'Imágenes'}")
        print("-" * 70)
        
        for result in results:
            filename = result['file']
            analysis = result['analysis']
            
            rank = analysis.get('architecture', {}).get('rank_info', {}).get('most_common_rank', 'N/A')
            size = analysis.get('file_info', {}).get('tamaño_mb', 'N/A')
            images = analysis.get('metadata', {}).get('ss_num_train_images', 'N/A')
            
            print(f"{filename:<30} {str(rank):<10} {str(size):<15} {str(images)}")


def example_save_to_json():
    """Ejemplo: guardar análisis en JSON"""
    print("\n" + "=" * 70)
    print("EJEMPLO 4: Guardar en JSON")
    print("=" * 70)
    
    file_path = "mi_lora.safetensors"
    output_path = "analysis_result.json"
    
    try:
        analyzer = LoRAAnalyzer(file_path)
        analysis = analyzer.analyze()
        
        # Guardar en JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Análisis guardado en: {output_path}")
        
    except FileNotFoundError:
        print(f"⚠️  Archivo no encontrado: {file_path}")


def example_batch_processing():
    """Ejemplo: procesamiento por lotes"""
    print("\n" + "=" * 70)
    print("EJEMPLO 5: Procesamiento por Lotes")
    print("=" * 70)
    
    import os
    
    # Directorio con archivos LoRA
    lora_directory = "loras/"
    
    if not os.path.exists(lora_directory):
        print(f"⚠️  Directorio no encontrado: {lora_directory}")
        print("   Crea un directorio 'loras/' y coloca tus archivos allí")
        return
    
    # Analizar todos los archivos .safetensors
    results = []
    
    for filename in os.listdir(lora_directory):
        if filename.endswith('.safetensors'):
            file_path = os.path.join(lora_directory, filename)
            
            try:
                print(f"\n🔍 Analizando: {filename}")
                analyzer = LoRAAnalyzer(file_path)
                analysis = analyzer.analyze()
                results.append(analysis)
                
                # Mostrar resumen rápido
                rank = analysis.get('architecture', {}).get('rank_info', {}).get('most_common_rank', 'N/A')
                print(f"   ✓ Rank: {rank}")
                
            except Exception as e:
                print(f"   ✗ Error: {str(e)}")
    
    print(f"\n✅ Análisis completado: {len(results)} archivos procesados")


def example_filter_by_rank():
    """Ejemplo: filtrar LoRAs por rank"""
    print("\n" + "=" * 70)
    print("EJEMPLO 6: Filtrar por Rank")
    print("=" * 70)
    
    import os
    
    target_rank = 32
    lora_directory = "loras/"
    
    if not os.path.exists(lora_directory):
        print(f"⚠️  Directorio no encontrado: {lora_directory}")
        return
    
    matching_loras = []
    
    for filename in os.listdir(lora_directory):
        if filename.endswith(('.safetensors', '.pt', '.ckpt')):
            file_path = os.path.join(lora_directory, filename)
            
            try:
                analyzer = LoRAAnalyzer(file_path)
                analysis = analyzer.analyze()
                
                rank = analysis.get('architecture', {}).get('rank_info', {}).get('most_common_rank')
                
                if rank == target_rank:
                    matching_loras.append(filename)
                    print(f"✓ {filename} - Rank {rank}")
                    
            except Exception as e:
                continue
    
    print(f"\n📊 Encontrados {len(matching_loras)} LoRAs con rank {target_rank}")


if __name__ == "__main__":
    print("\n🔍 EJEMPLOS DE USO - LoRA Analyzer\n")
    
    # Ejecuta los ejemplos
    # Nota: Muchos ejemplos fallarán si no tienes archivos reales
    # Esto es solo para demostración
    
    print("💡 Estos ejemplos requieren archivos LoRA reales")
    print("   Reemplaza los nombres de archivo con tus propios archivos\n")
    
    # Descomenta el ejemplo que quieras probar:
    
    # example_basic_analysis()
    # example_extract_specific_data()
    # example_compare_loras()
    # example_save_to_json()
    # example_batch_processing()
    # example_filter_by_rank()
    
    print("\n" + "=" * 70)
    print("Para usar estos ejemplos:")
    print("1. Reemplaza los nombres de archivo con tus archivos reales")
    print("2. Descomenta el ejemplo que quieras probar")
    print("3. Ejecuta: python examples.py")
    print("=" * 70)
