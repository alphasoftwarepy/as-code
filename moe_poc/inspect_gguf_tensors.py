"""
Script de inspección de tensores GGUF para analizar estructura de expertos MoE.
Lee la estructura interna y metadatos clave directamente de la cabecera GGUF.
"""
import struct
import sys
from pathlib import Path

# Constantes de tipos GGUF
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

def read_string(f):
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8", errors="replace")

def read_value(f, val_type):
    if val_type == GGUF_TYPE_UINT8:
        return struct.unpack("<B", f.read(1))[0]
    elif val_type == GGUF_TYPE_INT8:
        return struct.unpack("<b", f.read(1))[0]
    elif val_type == GGUF_TYPE_UINT16:
        return struct.unpack("<H", f.read(2))[0]
    elif val_type == GGUF_TYPE_INT16:
        return struct.unpack("<h", f.read(2))[0]
    elif val_type == GGUF_TYPE_UINT32:
        return struct.unpack("<I", f.read(4))[0]
    elif val_type == GGUF_TYPE_INT32:
        return struct.unpack("<i", f.read(4))[0]
    elif val_type == GGUF_TYPE_FLOAT32:
        return struct.unpack("<f", f.read(4))[0]
    elif val_type == GGUF_TYPE_BOOL:
        return struct.unpack("<B", f.read(1))[0] != 0
    elif val_type == GGUF_TYPE_STRING:
        return read_string(f)
    elif val_type == GGUF_TYPE_UINT64:
        return struct.unpack("<Q", f.read(8))[0]
    elif val_type == GGUF_TYPE_INT64:
        return struct.unpack("<q", f.read(8))[0]
    elif val_type == GGUF_TYPE_FLOAT64:
        return struct.unpack("<d", f.read(8))[0]
    elif val_type == GGUF_TYPE_ARRAY:
        elem_type = struct.unpack("<I", f.read(4))[0]
        array_len = struct.unpack("<Q", f.read(8))[0]
        return [read_value(f, elem_type) for _ in range(min(array_len, 20))]
    else:
        return f"<unknown_type_{val_type}>"

def inspect_gguf(filepath: str):
    p = Path(filepath)
    if not p.exists():
        print(f"Error: {filepath} not found.")
        return

    print(f"==================================================")
    print(f" GGUF MoE Inspector: {p.name}")
    print(f" File Size: {p.stat().st_size / (1024**3):.2f} GB")
    print(f"==================================================")

    with open(filepath, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            print(f"Error: Not a valid GGUF file (magic={magic})")
            return

        version = struct.unpack("<I", f.read(4))[0]
        tensor_count = struct.unpack("<Q", f.read(8))[0]
        kv_count = struct.unpack("<Q", f.read(8))[0]

        print(f"GGUF Version:      {version}")
        print(f"Total Tensors:     {tensor_count}")
        print(f"Metadata KeyCount: {kv_count}")
        print("-" * 50)

        metadata = {}
        for _ in range(kv_count):
            key = read_string(f)
            val_type = struct.unpack("<I", f.read(4))[0]
            val = read_value(f, val_type)
            metadata[key] = val

        # Extraer campos clave de MoE
        arch = metadata.get("general.architecture", "unknown")
        print(f"Architecture:      {arch}")
        print(f"Model Name:        {metadata.get('general.name', 'N/A')}")
        print(f"Context Length:    {metadata.get(f'{arch}.context_length', 'N/A')}")
        print(f"Block/Layer Count: {metadata.get(f'{arch}.block_count', 'N/A')}")
        print(f"Embedding Length:  {metadata.get(f'{arch}.embedding_length', 'N/A')}")
        print(f"Expert Count:      {metadata.get(f'{arch}.expert_count', 'N/A')}")
        print(f"Expert Used/Token: {metadata.get(f'{arch}.expert_used_count', 'N/A')}")
        print(f"Expert Shared FFN: {metadata.get(f'{arch}.expert_shared_count', 'N/A')}")
        print("-" * 50)

        # Leer tensores
        print("Muestreo de tensores de expertos:")
        expert_tensors = []
        dense_tensors = []

        for i in range(tensor_count):
            t_name = read_string(f)
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            t_type = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]

            if "exp" in t_name.lower():
                expert_tensors.append((t_name, dims, t_type))
            else:
                dense_tensors.append((t_name, dims, t_type))

        print(f"Tensores densos identificados:   {len(dense_tensors)}")
        print(f"Tensores de expertos (MoE):     {len(expert_tensors)}")
        print("")
        print("Ejemplos de tensores de expertos (primeros 6):")
        for t_name, dims, t_type in expert_tensors[:6]:
            print(f"  {t_name:40s} | dims={dims} | type_id={t_type}")

        print("")
        print("Estructura de tensores observada:")
        if expert_tensors:
            first_exp = expert_tensors[0]
            print(f"  Tensor MoE típico: {first_exp[0]}")
            print(f"  Dimensiones: {first_exp[1]} (dim 0 = num_experts consolidado)")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else r"C:\as-code\moe_poc\models\qwen1.5-moe-a2.7b-q4_k_m.gguf"
    inspect_gguf(path)
