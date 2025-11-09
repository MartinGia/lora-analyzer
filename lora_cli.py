#!/usr/bin/env python3
"""
LoRA Analyzer CLI
Herramienta de línea de comandos para analizar archivos LoRA
"""

import argparse
import sys
import json
from pathlib import Path
from lora_analyzer import LoRAAnalyzer, format_analysis_report


def main():
    parser = argparse.ArgumentParser(
        description="Analiza archivos LoRA y extrae información técnica",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Analizar un archivo
  python lora_cli.py mi_lora.safetensors
  
  # Guardar resultado en JSON
  python lora_cli.py mi_lora.safetensors --output resultado.json
  
  # Analizar múltiples archivos
  python lora_cli.py lora1.safetensors lora2.pt lora3.ckpt
  
  # Modo verbose con más detalles
  python lora_cli.py mi_lora.safetensors --verbose
        """
    )
    
    parser.add_argument(
        "files",
        nargs="+",
        help="Archivo(s) LoRA a analizar (.safetensors, .pt, .ckpt)"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Guardar resultado en archivo JSON",
        type=str
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Mostrar información detallada"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Salida en formato JSON"
    )
    
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Comparar múltiples LoRAs (requiere 2+ archivos)"
    )
    
    args = parser.parse_args()
    
    # Validar archivos
    files = [Path(f) for f in args.files]
    for f in files:
        if not f.exists():
            print(f"❌ Error: El archivo no existe: {f}", file=sys.stderr)
            sys.exit(1)
    
    results = []
    
    # Analizar cada archivo
    for file_path in files:
        print(f"\n🔍 Analizando: {file_path.name}")
        print("-" * 70)
        
        try:
            analyzer = LoRAAnalyzer(str(file_path))
            analysis = analyzer.analyze()
            results.append(analysis)
            
            if args.json:
                print(json.dumps(analysis, indent=2, ensure_ascii=False))
            else:
                print(format_analysis_report(analysis))
        
        except Exception as e:
            print(f"❌ Error al analizar {file_path.name}: {str(e)}", file=sys.stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    # Modo comparación
    if args.compare and len(results) > 1:
        print("\n" + "=" * 70)
        print("📊 COMPARACIÓN DE LORAS")
        print("=" * 70)
        compare_loras(results)
    
    # Guardar en JSON si se especificó
    if args.output and results:
        output_path = Path(args.output)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Resultados guardados en: {output_path}")


def compare_loras(results):
    """Compara múltiples LoRAs"""
    print("\n📋 Comparación de características:\n")
    
    # Tabla comparativa
    headers = ["Característica"] + [r["file_info"]["nombre"] for r in results]
    
    comparisons = []
    
    # Tamaño
    row = ["Tamaño (MB)"]
    for r in results:
        row.append(f"{r['file_info'].get('tamaño_mb', 'N/A')}")
    comparisons.append(row)
    
    # Rank
    row = ["Rank"]
    for r in results:
        rank_info = r.get("architecture", {}).get("rank_info", {})
        rank = rank_info.get("most_common_rank", "N/A")
        row.append(str(rank))
    comparisons.append(row)
    
    # Capas
    row = ["Total capas"]
    for r in results:
        layers = r.get("architecture", {}).get("total_layers", "N/A")
        row.append(str(layers))
    comparisons.append(row)
    
    # Imágenes de entrenamiento
    row = ["Imágenes entreno"]
    for r in results:
        num_images = r.get("metadata", {}).get("ss_num_train_images", "N/A")
        row.append(str(num_images))
    comparisons.append(row)
    
    # Learning rate
    row = ["Learning rate"]
    for r in results:
        lr = r.get("metadata", {}).get("ss_learning_rate", "N/A")
        row.append(str(lr))
    comparisons.append(row)
    
    # Imprimir tabla
    col_widths = [max(len(str(row[i])) for row in [headers] + comparisons) + 2 
                  for i in range(len(headers))]
    
    # Header
    print("  ".join(h.ljust(w) for h, w in zip(headers, col_widths)))
    print("-" * sum(col_widths))
    
    # Filas
    for row in comparisons:
        print("  ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths)))


if __name__ == "__main__":
    main()
