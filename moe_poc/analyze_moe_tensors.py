"""
Script de análisis estructural de tensores MoE en GGUF.
Extrae metadatos de arquitectura, topología de expertos, tamaños de tensores y cálculo de offsets.
"""
import gguf
import json
from pathlib import Path

def to_scalar(val):
    if isinstance(val, (list, tuple)) and len(val) == 1:
        return to_scalar(val[0])
    if hasattr(val, "item"):
        return val.item()
    if isinstance(val, (bytes, bytearray)):
        return str(val, encoding="utf-8", errors="replace")
    return val

def analyze_moe_model(model_path: str):
    p = Path(model_path)
    print(f"==================================================")
    print(f" Analizando Modelo MoE: {p.name}")
    print(f" Tamano total: {p.stat().st_size / (1024**3):.2f} GB")
    print(f"==================================================")

    reader = gguf.GGUFReader(model_path)
    
    arch = None
    for field in reader.fields.values():
        if field.name == "general.architecture":
            arch = str(bytes(field.parts[-1]), encoding="utf-8", errors="replace")
            break
            
    print(f"Arquitectura detectada: {arch}")
    
    metadata = {}
    for field in reader.fields.values():
        val = field.parts[-1]
        try:
            if isinstance(val, (bytes, bytearray)):
                metadata[field.name] = str(val, encoding="utf-8", errors="replace")
            elif isinstance(val, memoryview):
                metadata[field.name] = [to_scalar(x) for x in val.tolist()] if len(val) < 20 else f"<array len={len(val)}>"
            else:
                metadata[field.name] = to_scalar(val)
        except Exception:
            metadata[field.name] = str(val)

    block_count = int(to_scalar(metadata.get(f"{arch}.block_count", 0)))
    expert_count = int(to_scalar(metadata.get(f"{arch}.expert_count", 0)))
    expert_used_count = int(to_scalar(metadata.get(f"{arch}.expert_used_count", 0)))
    embedding_length = int(to_scalar(metadata.get(f"{arch}.embedding_length", 0)))
    expert_feed_forward_length = int(to_scalar(metadata.get(f"{arch}.expert_feed_forward_length", 0)))
    context_length = int(to_scalar(metadata.get(f"{arch}.context_length", 0)))
    
    print(f"  Capas (block_count):             {block_count}")
    print(f"  Total Expertos por Capa:        {expert_count}")
    print(f"  Expertos Activos por Token:     {expert_used_count}")
    print(f"  Embedding Length (Hidden Dim):  {embedding_length}")
    print(f"  Expert Intermediate Dim:        {expert_feed_forward_length}")
    print(f"  Context Length:                 {context_length}")
    print("-" * 50)

    dense_tensors = []
    moe_expert_tensors = {}
    router_tensors = {}
    
    total_dense_bytes = 0
    total_moe_bytes = 0

    for tensor in reader.tensors:
        t_name = tensor.name
        t_shape = [int(s) for s in tensor.shape]
        t_type = tensor.tensor_type
        t_nbytes = int(tensor.n_bytes)

        if "exps" in t_name:
            total_moe_bytes += t_nbytes
            parts = t_name.split(".")
            layer_idx = int(parts[1])
            if layer_idx not in moe_expert_tensors:
                moe_expert_tensors[layer_idx] = {}
            moe_expert_tensors[layer_idx][parts[2]] = {
                "name": t_name,
                "shape": t_shape,
                "type": str(t_type),
                "bytes": t_nbytes,
                "mb": round(t_nbytes / (1024**2), 2)
            }
        elif "ffn_gate_inp" in t_name or "router" in t_name:
            parts = t_name.split(".")
            layer_idx = int(parts[1])
            router_tensors[layer_idx] = {
                "name": t_name,
                "shape": t_shape,
                "type": str(t_type),
                "bytes": t_nbytes,
                "mb": round(t_nbytes / (1024**2), 2)
            }
            total_dense_bytes += t_nbytes
        else:
            dense_tensors.append({
                "name": t_name,
                "shape": t_shape,
                "type": str(t_type),
                "bytes": t_nbytes,
                "mb": round(t_nbytes / (1024**2), 2)
            })
            total_dense_bytes += t_nbytes

    print(f"Desglose de Memoria:")
    print(f"  Memoria Tensores Densos (Atencion, Embeddings, Norms): {total_dense_bytes / (1024**2):.2f} MB ({total_dense_bytes / (1024**3):.2f} GB)")
    print(f"  Memoria Tensores MoE (Expertos Consolidados):           {total_moe_bytes / (1024**2):.2f} MB ({total_moe_bytes / (1024**3):.2f} GB)")
    print(f"  Total Modelo en Tensores:                               {(total_dense_bytes + total_moe_bytes) / (1024**3):.2f} GB")
    print("-" * 50)

    if 0 in moe_expert_tensors:
        layer0_exps = moe_expert_tensors[0]
        layer0_bytes = sum(t["bytes"] for t in layer0_exps.values())
        bytes_per_single_expert = float(layer0_bytes) / float(expert_count)
        single_expert_mb = bytes_per_single_expert / (1024**2)
        
        print(f"Granularidad Matematica de Expertos:")
        print(f"  Bloque MoE completo por capa (60 expertos): {layer0_bytes / (1024**2):.2f} MB")
        for k, v in layer0_exps.items():
            print(f"    - {k}: {v['name']} | shape={v['shape']} | {v['mb']} MB | tipo={v['type']}")
        print(f"  Tamano de UN SOLO experto completo (gate + down + up): {single_expert_mb:.2f} MB ({bytes_per_single_expert / 1024:.1f} KB)")
        print(f"  Tamano de 4 expertos activos por token:                {(bytes_per_single_expert * 4) / (1024**2):.2f} MB por capa")
        print(f"  Total 4 expertos activos en 24 capas:                  {(bytes_per_single_expert * 4 * block_count) / (1024**2):.2f} MB (~{((bytes_per_single_expert * 4 * block_count) / (1024**3)):.2f} GB)")
        print("-" * 50)

    vram_budget_mb = 3900.0
    vram_dense_mb = float(total_dense_bytes) / (1024**2)
    vram_kv_cache_mb = 350.0
    vram_for_hot_experts_mb = vram_budget_mb - vram_dense_mb - vram_kv_cache_mb
    
    max_hot_experts_total = int(vram_for_hot_experts_mb / single_expert_mb)
    hot_experts_per_layer = int(max_hot_experts_total / block_count)
    
    print(f"Presupuesto de VRAM para HotSet en 4 GB (GTX 1650 Ti):")
    print(f"  VRAM Presupuestada:                   {vram_budget_mb:.1f} MB")
    print(f"  VRAM requerida para TODAS las atenciones densas: {vram_dense_mb:.1f} MB")
    print(f"  VRAM requerida para KV Cache (2K ctx): {vram_kv_cache_mb:.1f} MB")
    print(f"  VRAM Disponible EXCLUSIVA para Expertos HOT:    {vram_for_hot_experts_mb:.1f} MB (~{vram_for_hot_experts_mb/1024:.2f} GB)")
    print(f"  -> Capacidad Total de Expertos HOT en VRAM:    {max_hot_experts_total} expertos individuales")
    print(f"  -> Capacidad por Capa (24 capas):              {hot_experts_per_layer} expertos HOT de 60 por capa ({hot_experts_per_layer/expert_count*100:.1f}%)")
    print("=" * 50)

    summary = {
        "model_path": str(model_path),
        "architecture": arch,
        "block_count": block_count,
        "expert_count": expert_count,
        "expert_used_count": expert_used_count,
        "embedding_length": embedding_length,
        "expert_feed_forward_length": expert_feed_forward_length,
        "total_dense_bytes": total_dense_bytes,
        "total_dense_mb": round(vram_dense_mb, 2),
        "total_moe_bytes": total_moe_bytes,
        "total_moe_mb": round(total_moe_bytes / (1024**2), 2),
        "single_expert_bytes": int(bytes_per_single_expert),
        "single_expert_mb": round(single_expert_mb, 3),
        "hot_experts_capacity_4gb_vram": max_hot_experts_total,
        "hot_experts_per_layer_capacity": hot_experts_per_layer
    }
    
    with open("C:/as-code/moe_poc/moe_tensor_analysis.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Analisis guardado en C:/as-code/moe_poc/moe_tensor_analysis.json")

if __name__ == "__main__":
    analyze_moe_model(r"C:\as-code\moe_poc\models\qwen1.5-moe-a2.7b-q4_k_m.gguf")
