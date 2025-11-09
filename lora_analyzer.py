"""
LoRA Analyzer - Core Module
Analiza archivos LoRA (.safetensors, .pt, .ckpt) y extrae información técnica
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

try:
    from safetensors import safe_open
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class LoRAAnalyzer:
    """Analizador de archivos LoRA"""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.file_name = self.file_path.name
        self.file_size = self.file_path.stat().st_size
        self.extension = self.file_path.suffix.lower()
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"El archivo no existe: {file_path}")
    
    def analyze(self) -> Dict[str, Any]:
        """Análisis completo del archivo LoRA"""
        
        result = {
            "file_info": self._get_file_info(),
            "architecture": {},
            "metadata": {},
            "weights_analysis": {},
            "recommendations": []
        }
        
        # Cargar y analizar según el formato
        if self.extension == ".safetensors":
            result.update(self._analyze_safetensors())
        elif self.extension in [".pt", ".pth", ".ckpt"]:
            result.update(self._analyze_pytorch())
        else:
            raise ValueError(f"Formato no soportado: {self.extension}")
        
        # Generar recomendaciones
        result["recommendations"] = self._generate_recommendations(result)
        
        return result
    
    def _get_file_info(self) -> Dict[str, Any]:
        """Información básica del archivo"""
        return {
            "nombre": self.file_name,
            "ruta": str(self.file_path.absolute()),
            "tamaño_mb": round(self.file_size / (1024 * 1024), 2),
            "extension": self.extension
        }
    
    def _analyze_safetensors(self) -> Dict[str, Any]:
        """Analiza archivos .safetensors"""
        if not SAFETENSORS_AVAILABLE:
            return {"error": "Instala 'safetensors': pip install safetensors"}
        
        result = {
            "architecture": {},
            "metadata": {},
            "weights_analysis": {}
        }
        
        with safe_open(self.file_path, framework="pt") as f:
            # Obtener metadatos
            metadata = f.metadata()
            if metadata:
                result["metadata"] = self._parse_metadata(metadata)
            
            # Analizar capas y pesos
            keys = list(f.keys())
            result["architecture"] = self._analyze_architecture(keys, f)
            result["weights_analysis"] = self._analyze_weights(keys, f)
        
        return result
    
    def _analyze_pytorch(self) -> Dict[str, Any]:
        """Analiza archivos .pt, .pth, .ckpt"""
        if not TORCH_AVAILABLE:
            return {"error": "Instala 'torch': pip install torch"}
        
        result = {
            "architecture": {},
            "metadata": {},
            "weights_analysis": {}
        }
        
        try:
            state_dict = torch.load(self.file_path, map_location="cpu")
            
            # Extraer metadatos si existen
            if isinstance(state_dict, dict):
                if "metadata" in state_dict:
                    result["metadata"] = state_dict["metadata"]
                
                # Obtener los pesos (pueden estar en diferentes claves)
                weights = state_dict.get("state_dict", state_dict)
                
                keys = list(weights.keys())
                result["architecture"] = self._analyze_architecture_dict(keys, weights)
                result["weights_analysis"] = self._analyze_weights_dict(keys, weights)
        
        except Exception as e:
            result["error"] = f"Error al cargar archivo: {str(e)}"
        
        return result
    
    def _parse_metadata(self, metadata: Dict) -> Dict[str, Any]:
        """Parsea y organiza los metadatos"""
        parsed = {}
        
        # Buscar información común
        common_keys = [
            "ss_network_module", "ss_network_dim", "ss_network_alpha",
            "ss_learning_rate", "ss_text_encoder_lr", "ss_unet_lr",
            "ss_num_epochs", "ss_num_train_images", "ss_num_batches_per_epoch",
            "ss_batch_size", "ss_base_model", "ss_sd_model_name",
            "ss_resolution", "ss_clip_skip", "ss_max_train_steps",
            "ss_dataset_dirs", "ss_training_comment"
        ]
        
        for key in common_keys:
            if key in metadata:
                value = metadata[key]
                # Intentar parsear JSON si es string
                if isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except:
                        pass
                parsed[key] = value
        
        # Agregar otros metadatos
        for key, value in metadata.items():
            if key not in parsed:
                parsed[key] = value
        
        return parsed
    
    def _analyze_architecture(self, keys: List[str], tensor_file) -> Dict[str, Any]:
        """Analiza la arquitectura del LoRA desde safetensors"""
        arch = {
            "total_layers": len(keys),
            "layer_details": [],
            "rank_info": {},
            "alpha_info": {},
            "module_types": set()
        }
        
        # Analizar cada capa
        for key in keys:
            tensor = tensor_file.get_tensor(key)
            shape = tensor.shape
            
            layer_info = {
                "name": key,
                "shape": list(shape),
                "num_params": np.prod(shape)
            }
            
            # Detectar rank (típicamente la dimensión más pequeña en matrices LoRA)
            if len(shape) >= 2:
                potential_rank = min(shape)
                layer_info["potential_rank"] = int(potential_rank)
            
            # Identificar tipo de módulo
            if "lora_up" in key or "lora_down" in key:
                module_type = "lora"
            elif "alpha" in key:
                module_type = "alpha"
            else:
                module_type = "other"
            
            layer_info["type"] = module_type
            arch["module_types"].add(module_type)
            arch["layer_details"].append(layer_info)
        
        arch["module_types"] = list(arch["module_types"])
        
        # Calcular estadísticas de rank
        ranks = [l["potential_rank"] for l in arch["layer_details"] 
                if "potential_rank" in l]
        if ranks:
            arch["rank_info"] = {
                "detected_ranks": list(set(ranks)),
                "most_common_rank": max(set(ranks), key=ranks.count),
                "min_rank": min(ranks),
                "max_rank": max(ranks)
            }
        
        return arch
    
    def _analyze_architecture_dict(self, keys: List[str], weights: Dict) -> Dict[str, Any]:
        """Analiza la arquitectura del LoRA desde diccionario PyTorch"""
        arch = {
            "total_layers": len(keys),
            "layer_details": [],
            "rank_info": {},
            "module_types": set()
        }
        
        for key in keys:
            tensor = weights[key]
            if hasattr(tensor, 'shape'):
                shape = tensor.shape
            else:
                continue
            
            layer_info = {
                "name": key,
                "shape": list(shape),
                "num_params": np.prod(shape)
            }
            
            if len(shape) >= 2:
                potential_rank = min(shape)
                layer_info["potential_rank"] = int(potential_rank)
            
            # Identificar tipo de módulo
            if "lora" in key.lower():
                module_type = "lora"
            elif "alpha" in key.lower():
                module_type = "alpha"
            else:
                module_type = "other"
            
            layer_info["type"] = module_type
            arch["module_types"].add(module_type)
            arch["layer_details"].append(layer_info)
        
        arch["module_types"] = list(arch["module_types"])
        
        ranks = [l["potential_rank"] for l in arch["layer_details"] 
                if "potential_rank" in l]
        if ranks:
            arch["rank_info"] = {
                "detected_ranks": list(set(ranks)),
                "most_common_rank": max(set(ranks), key=ranks.count),
                "min_rank": min(ranks),
                "max_rank": max(ranks)
            }
        
        return arch
    
    def _analyze_weights(self, keys: List[str], tensor_file) -> Dict[str, Any]:
        """Analiza los pesos del modelo"""
        analysis = {
            "total_parameters": 0,
            "weight_statistics": {},
            "layer_analysis": []
        }
        
        for key in keys[:10]:  # Limitar para rendimiento
            try:
                tensor = tensor_file.get_tensor(key)
                
                # Convertir a numpy para análisis
                if hasattr(tensor, 'numpy'):
                    weights = tensor.numpy()
                else:
                    weights = np.array(tensor)
                
                stats = {
                    "layer": key,
                    "mean": float(np.mean(weights)),
                    "std": float(np.std(weights)),
                    "min": float(np.min(weights)),
                    "max": float(np.max(weights)),
                    "non_zero_ratio": float(np.count_nonzero(weights) / weights.size)
                }
                
                analysis["layer_analysis"].append(stats)
                analysis["total_parameters"] += weights.size
            except:
                continue
        
        return analysis
    
    def _analyze_weights_dict(self, keys: List[str], weights: Dict) -> Dict[str, Any]:
        """Analiza los pesos desde diccionario"""
        analysis = {
            "total_parameters": 0,
            "layer_analysis": []
        }
        
        for key in keys[:10]:
            try:
                tensor = weights[key]
                if hasattr(tensor, 'numpy'):
                    w = tensor.numpy()
                else:
                    w = np.array(tensor)
                
                stats = {
                    "layer": key,
                    "mean": float(np.mean(w)),
                    "std": float(np.std(w)),
                    "min": float(np.min(w)),
                    "max": float(np.max(w)),
                    "non_zero_ratio": float(np.count_nonzero(w) / w.size)
                }
                
                analysis["layer_analysis"].append(stats)
                analysis["total_parameters"] += w.size
            except:
                continue
        
        return analysis
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Genera recomendaciones basadas en el análisis"""
        recommendations = []
        
        # Recomendaciones basadas en rank
        if "architecture" in analysis and "rank_info" in analysis["architecture"]:
            rank_info = analysis["architecture"]["rank_info"]
            if rank_info:
                common_rank = rank_info.get("most_common_rank", 0)
                
                if common_rank < 16:
                    recommendations.append(
                        f"Rank bajo ({common_rank}): Bueno para eficiencia. "
                        "Para más detalles, prueba rank 32-64."
                    )
                elif common_rank > 128:
                    recommendations.append(
                        f"Rank alto ({common_rank}): Mucho detalle pero más pesado. "
                        "Considera reducir a 64-128 para mejor balance."
                    )
                else:
                    recommendations.append(
                        f"Rank óptimo ({common_rank}): Buen balance entre detalle y eficiencia."
                    )
        
        # Recomendaciones basadas en metadatos
        if "metadata" in analysis and analysis["metadata"]:
            metadata = analysis["metadata"]
            
            if "ss_num_train_images" in metadata:
                num_images = metadata["ss_num_train_images"]
                try:
                    num_images = int(num_images)
                    if num_images < 20:
                        recommendations.append(
                            f"Dataset pequeño ({num_images} imágenes): "
                            "Considera aumentar a 30-50 imágenes para mejor generalización."
                        )
                    elif num_images > 100:
                        recommendations.append(
                            f"Dataset grande ({num_images} imágenes): "
                            "Excelente para capturar variaciones."
                        )
                except:
                    pass
            
            if "ss_learning_rate" in metadata:
                lr = metadata["ss_learning_rate"]
                recommendations.append(f"Learning rate usado: {lr}")
        
        if not recommendations:
            recommendations.append(
                "Sube el archivo LoRA para obtener recomendaciones específicas."
            )
        
        return recommendations


def format_analysis_report(analysis: Dict) -> str:
    """Formatea el análisis como texto legible"""
    report = []
    report.append("=" * 70)
    report.append("REPORTE DE ANÁLISIS LORA")
    report.append("=" * 70)
    report.append("")
    
    # Información del archivo
    if "file_info" in analysis:
        report.append("📁 INFORMACIÓN DEL ARCHIVO:")
        info = analysis["file_info"]
        report.append(f"  Nombre: {info.get('nombre', 'N/A')}")
        report.append(f"  Tamaño: {info.get('tamaño_mb', 0)} MB")
        report.append(f"  Formato: {info.get('extension', 'N/A')}")
        report.append("")
    
    # Arquitectura
    if "architecture" in analysis and analysis["architecture"]:
        report.append("🏗️  ARQUITECTURA:")
        arch = analysis["architecture"]
        report.append(f"  Total de capas: {arch.get('total_layers', 0)}")
        
        if "rank_info" in arch and arch["rank_info"]:
            rank = arch["rank_info"]
            report.append(f"  Rank más común: {rank.get('most_common_rank', 'N/A')}")
            report.append(f"  Rango de ranks: {rank.get('min_rank', 'N/A')} - {rank.get('max_rank', 'N/A')}")
        
        report.append("")
    
    # Metadatos
    if "metadata" in analysis and analysis["metadata"]:
        report.append("⚙️  METADATOS DE ENTRENAMIENTO:")
        meta = analysis["metadata"]
        
        key_info = {
            "ss_base_model": "Modelo base",
            "ss_network_dim": "Network dim (rank)",
            "ss_network_alpha": "Alpha",
            "ss_learning_rate": "Learning rate",
            "ss_num_epochs": "Épocas",
            "ss_num_train_images": "Imágenes de entrenamiento",
            "ss_batch_size": "Batch size",
            "ss_resolution": "Resolución"
        }
        
        for key, label in key_info.items():
            if key in meta:
                report.append(f"  {label}: {meta[key]}")
        
        report.append("")
    
    # Análisis de pesos
    if "weights_analysis" in analysis and analysis["weights_analysis"]:
        weights = analysis["weights_analysis"]
        if "total_parameters" in weights:
            report.append("📊 ANÁLISIS DE PESOS:")
            report.append(f"  Total parámetros: {weights['total_parameters']:,}")
            report.append("")
    
    # Recomendaciones
    if "recommendations" in analysis and analysis["recommendations"]:
        report.append("💡 RECOMENDACIONES:")
        for i, rec in enumerate(analysis["recommendations"], 1):
            report.append(f"  {i}. {rec}")
        report.append("")
    
    report.append("=" * 70)
    
    return "\n".join(report)
