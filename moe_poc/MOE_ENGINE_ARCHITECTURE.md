# MOE_ENGINE_ARCHITECTURE.md
# AS-Core — Arquitectura del MoE Residency Engine
### Estado: Bloques B1, B2 & B3 Completados | Fecha: 2026-08-27

---

## 1. Principio Arquitectónico Central

El **MoE Residency Engine** permite ejecutar modelos Mixture-of-Experts que exceden holgadamente la memoria VRAM disponible (ej: modelos de 8 a 32 GB en GPUs de 4 GB VRAM) mediante una jerarquía física de memoria de 3 niveles:

```
┌────────────────────────────────────────────────────────────────────────┐
│                          JERARQUÍA DE MEMORIA                          │
├───────────────┬───────────────────────────────────┬────────────────────┤
│ Nivel         │ Contenido                         │ Ancho de Banda     │
├───────────────┼───────────────────────────────────┼────────────────────┤
│ 1. HOT (VRAM) │ Atenciones Densas (100%)          │ GDDR6: 192 GB/s    │
│               │ KV Cache                          │ [MEASURED]         │
│               │ HotSet de Expertos Frecuentes     │                    │
├───────────────┼───────────────────────────────────┼────────────────────┤
│ 2. WARM (RAM) │ Expertos Secundarios Mapped/Pinned│ Host RAM: 25 GB/s  │
│               │ Buffer de DMA Host (Pinned)      │ [MEASURED: 10.4GBs]│
├───────────────┼───────────────────────────────────┼────────────────────┤
│ 3. COLD (Disk)│ Modelo Base GGUF en Storage       │ NVMe: ~2.4 GB/s    │
└───────────────┴───────────────────────────────────┴────────────────────┘
```

---

## 2. Aislamiento e Integración con AS-Core

El motor MoE se integra sin modificar el runtime cognitivo estable:

```
                  ┌──────────────────────┐
                  │       AS-Core        │
                  │ (Coordinator, Skills,│
                  │  RAG, Working Memory)│
                  └──────────┬───────────┘
                             │
                  ┌──────────▼───────────┐
                  │  InferenceProvider   │
                  │  (Abstracción Base)  │
                  └──────────┬───────────┘
                             │
                  ┌──────────▼───────────┐
                  │ MoEInferenceProvider │
                  │  (Nuevo Provider)    │
                  └──────────┬───────────┘
                             │
     ┌───────────────────────┼───────────────────────┬──────────────────────┐
     │                       │                       │                      │
┌────▼─────────────┐ ┌───────▼──────────────┐ ┌──────▼─────────────┐ ┌──────▼─────────────┐
│   ModelProfile   │ │    ExpertRegistry    │ │   VRAMExpertPool   │ │    RAMWarmPool     │
│(Dynamic Analyzer)│ │(Slice-level Indexer) │ │ (O(1) Slot Reuse)  │ │ (Pinned DMA Engine)│
└──────────────────┘ └──────────────────────┘ └────────────────────┘ └────────────────────┘
```

---

## 3. Módulos Implementados

### 3.1. `core.moe.expert_tensor` (Bloque B1)
- **`ExpertTensorSlice`:** Representa un slice individual de tensor (`gate`, `up`, `down`, `router`) con dimensiones, offsets exactos en el archivo binario GGUF, tipo de cuantización y cálculo de bytes.
- **`ExpertTensor`:** Agrupa los 3 slices de un experto específico en una capa dada. Mantiene el estado de residencia (`COLD`, `WARM`, `HOT`) y punteros de memoria (`vram_device_ptr`, `ram_host_ptr`).

### 3.2. `core.moe.model_profile` (Bloque B1)
- **`ModelProfile`:** Parser agnóstico a la arquitectura del modelo (`from_gguf`). Extrae dinámicamente:
  - `block_count` (Capas)
  - `expert_count` (Total de expertos por capa)
  - `expert_used_count` (Expertos activos por token)
  - `single_expert_bytes` (Cálculo exacto del tamaño de un experto)
  - `calculate_hotset_capacity()` (Cálculo dinámico de capacidad en VRAM)

### 3.3. `core.moe.expert_registry` (Bloque B1)
- **`ExpertRegistry`:** Indexa deterministamente la matriz bidimensional `layers[layer_id][expert_id]`.
- Soporta cuantización mixta por capa (ej: Q6_K en capas iniciales, Q4_K en capas intermedias en formato `Q4_K_M`).
- Proporciona resolución instantánea de `routing_set` de expertos activados.

### 3.4. `core.moe.cuda_driver` (Bloque B2)
- Wrapper de bajo nivel sobre `nvcuda.dll` en Windows:
  - `mem_alloc()` / `mem_free()`
  - `mem_alloc_host()` / `mem_free_host()`
  - `memcpy_htod()` / `memcpy_htod_async()`
  - `synchronize()` / `get_mem_info()`

### 3.5. `core.moe.vram_pool` (Bloque B2)
- **`VRAMExpertPool`:** Pool de memoria pre-asignada en VRAM para tensores de expertos.
- **`VRAMSlot`:** Buffer contiguo en GPU para un experto individual.
- **Operaciones O(1):**
  - `allocate_slot()`: Latencia medida: **`0.0075 ms`**
  - `upload_expert()`: Latencia medida (6.02 MB): **`4.11 ms`**
  - `evict_slot()`: Latencia medida: **`0.0067 ms`**
  - `reuse_slot()`: Promoción directa en slot existente (sin reasignación): **`3.18 ms`**

### 3.6. `core.moe.ram_warm_pool` (Bloque B3)
- **`RAMWarmPool`:** Almacenamiento de nivel 2 en Host RAM.
- **`WarmSlot`:** Buffer en RAM del Host para expertos secundarios.
- **Comparativa Física Pageable vs Pinned:**
  - `Pageable RAM -> VRAM`: **1.055 ms** (`5.57 GB/s`)
  - `Pinned RAM (DMA) -> VRAM`: **0.565 ms** (`10.41 GB/s`)
  - **Reducción de Latencia de Promoción:** **`46.5%`**
