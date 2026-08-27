# PERFORMANCE_BENCHMARKS.md
# AS-Core MoE Engine — Registro Oficial de Benchmarks
### Fecha: 2026-08-27 | Clasificación Estricta de Métricas

---

## 1. Baselines Oficiales Medidos [MEASURED]

| Test ID | Modelo | Configuración | VRAM (MB) | Tok/s Promedio | TTFT (ms) | GPU Util (%) | Estado |
|---|---|---|---|---|---|---|---|
| **BASELINE-0** | OLMoE-1B-7B Q4_K_M (3.92 GB) | 100% en GPU (`-ngl 99`) | 3892 MB | **65.37 tok/s** | 55.7 ms | 94% | **PASS** |
| **BASELINE-1A** | Qwen1.5-MoE-A2.7B Q4_K_M (8.84 GB) | 10 capas GPU / 14 capas RAM (`-ngl 10`) | 3893 MB | **14.38 tok/s** | 182.1 ms | 26% | **PASS** |
| **BASELINE-1B** | Qwen1.5-MoE-A2.7B Q4_K_M (8.84 GB) | 24 Atenciones GPU / Exp CPU (`--cpu-moe`) | 1480 MB | **14.80 tok/s** | 185.0 ms | 33% | **PASS** |
| **BASELINE-1C** | Qwen1.5-MoE-A2.7B Q4_K_M (8.84 GB) | 100% en Host RAM (`-ngl 0`) | 272 MB | **7.90 tok/s** | 233.2 ms | 0% | **FAIL** (<10 tok/s) |

---

## 2. Micro-benchmarks de VRAM Pool, RAM Warm Pool y Ejecución de Expertos [MEASURED]

| Operación | Componente / Tipo | Tamaño / Contexto | Latencia Medida [MEASURED] | Rendimiento / Ancho de Banda |
|---|---|---|---|---|
| **`allocate_slot()`** | `VRAMExpertPool` | Reserva de slot en RAM | **0.0075 ms** (7.5 µs) | $O(1)$ inmediato |
| **`evict_slot()`** | `VRAMExpertPool` | Desalojo de slot VRAM | **0.0067 ms** (6.7 µs) | $O(1)$ inmediato |
| **`upload_expert()`** | `VRAMExpertPool` + CUDA | 6.02 MB (1 experto Qwen) | **4.1174 ms** | Transferencia síncrona PCIe |
| **`reuse_slot()`** | `VRAMExpertPool` (Overwriting) | 6.02 MB (Reemplazo en slot) | **3.1841 ms** | $O(1)$ sin reasignar VRAM |
| **Promoción RAM -> VRAM (Pageable)** | `RAMWarmPool` $\to$ `VRAMExpertPool` | 6.02 MB (1 experto) | **1.055 ms** | **5.57 GB/s** (Paginable estándar) |
| **Promoción RAM -> VRAM (Pinned DMA)**| `RAMWarmPool` $\to$ `VRAMExpertPool` | 6.02 MB (1 experto) | **0.565 ms** | **10.41 GB/s** (DMA directo PCIe 3.0) |
| **B4.1 Single Expert SwiGLU GEMM** | `SingleExpertExecutor` + cuBLAS | 1 Token (2048 dims) | **3.8572 ms** | **Cosine Sim: 1.0000000** |
| **B4.1 Max Absolute Error** | GPU vs CPU Reference | Float32 exactness | **5.58e-07** | Tolerancia estricta cumplida |
| **B4.2 Four Active Experts GEMMs** | `MultiExpertExecutor` + cuBLAS | 4 Expertos en VRAM | **6.024 ms** | **4/4 Exactos (CosSim 1.0)** |
| **B4.2 Single Expert Compute in Slot**| `MultiExpertExecutor` (Warm Slot) | 1 Token (2048 dims) | **0.691–0.865 ms** | Latencia sub-milisegundo en GPU |
| **B4.3.1 Isolated Router Projection**| `RealRouter` + cuBLAS | 1 Token (60 logits) | **0.1770 ms** (177 µs) | **100% Top-K Match (Rel Err: 1.98e-7)** |
| **B4.3.2 Router -> Registry Lookup** | `RealRouter.route_and_resolve` | 4 ExpertTensors | **0.2100 ms** (210 µs) | **100% Identidad y Límites de Offsets** |
| **B4.3.3 Cold Promotion 4 Experts** | `ResidencyManager.dispatch_routing`| 4 Expertos (24.06 MB) | **4.785 ms** (1.19 ms/exp) | **4/4 MISSes resueltos a VRAM** |
| **B4.3.3 Warm Residency Hit Path** | `ResidencyManager.dispatch_routing`| 4 Expertos residentes | **0.0183 ms** (18.3 µs) | **100% HIT Rate (0 Bytes PCIe)** |
| **B4.3.4 Full MoE Layer (WARM)** | `MoELayerExecutor` (Router+4Exp+Saxpy)| 1 Capa MoE completa | **`2.338 ms`** | **Cosine Sim: 0.9999999 (MaxErr 2.38e-7)** |
| **B4.3.4 In-GPU Weighted Sum** | `cublasSaxpy` (4 accum) | 2048 dims en GPU | **0.178 ms** (178 µs) | **Zero Host Transfers** |
| **B4.3.5 RealRouter GPU Throughput** | `RoutingTracer` + cuBLAS | 18,432 decisiones | **5,943.8 dec/s** | **3.10 s para 768 tokens x 24 capas** |
| **B4.3.5 Tracing Overhead** | `RoutingTracer.record()` | Buffer JSONL en RAM | **10.964 µs / evento** | **Overhead despreciable (0.011 ms)** |

---

## 3. Resultados de la Simulación HotSet Offline (B4.3.6) [SIMULATED]

| HotSet Size / Capa | % Expertos | VRAM HotSet (MB) | VRAM Total Sistema | ¿Cabe en 4 GB? | Static Hit Rate [SIMULATED] | LRU Hit Rate [SIMULATED] |
|---|---|---|---|---|---|---|
| **4 Slots** | 6.7% | 577.5 MB | 2,788.8 MB | **YES (PASS)** | 12.29% | **46.42%** |
| **8 Slots** | 13.3% | 1,155.0 MB | 3,366.2 MB | **YES (PASS)** | 22.70% | **`67.28%`** |
| **12 Slots** | 20.0% | 1,732.5 MB | 3,943.8 MB | **YES (PASS)** | 32.06% | **`72.60%`** |
| **16 Slots** | 26.7% | 2,310.0 MB | 4,521.2 MB | **NO (OOM)** | 40.65% | 75.70% |
| **20 Slots** | 33.3% | 2,887.5 MB | 5,098.8 MB | **NO (OOM)** | 48.63% | 78.16% |
| **24 Slots** | 40.0% | 3,465.0 MB | 5,676.2 MB | **NO (OOM)** | 56.11% | 80.17% |
| **30 Slots (50%)** | 50.0% | 4,331.2 MB | 6,542.5 MB | **NO (OOM)** | 66.37% | 83.20% |
| **36 Slots (60%)** | 60.0% | 5,197.5 MB | 7,408.8 MB | **NO (OOM)** | 75.57% | 86.01% |

---

## 4. Benchmark End-to-End Sostenido en Hardware Real (B4.4.4) [MEASURED]

Evaluado sobre `Qwen1.5-MoE-A2.7B-Chat` en NVIDIA GeForce GTX 1650 Ti (4096 MB VRAM / 16 GB RAM / NVMe):

| Configuración | VRAM Asignada (MB) | Tok/s Promedio | TTFT (ms) | Hit Rate Real (%) | Fallos PCIe Totales | VRAM Headroom | Estado Gate ($\ge 10$) |
|---|---|---|---|---|---|---|---|
| **A) llama.cpp CPU (`-ngl 0`)** | 272.0 MB | **15.09 tok/s** | 547.5 ms | 0.0% | 0 fallos | $+3,824\text{ MB}$ | **PASS** |
| **B) llama.cpp Híbrido (`-ngl 10`)** | 3893.0 MB | **18.64 tok/s** | 480.0 ms | 41.7% | 0 fallos | $+203\text{ MB}$ | **PASS** |
| **C) Residency Engine (4 slots)** | 577.5 MB | **2.69 tok/s** | 525.6 ms | **45.40%** | 20,129 fallos | $+3,518\text{ MB}$ | Sub-gate (CPU slice overhead) |
| **D) Residency Engine (8 slots)** | 1155.0 MB | **4.25 tok/s** | 426.7 ms | **66.65%** | 12,295 fallos | $+2,941\text{ MB}$ | Sub-gate (CPU slice overhead) |
| **E) Residency Engine (12 slots)** | 1732.5 MB | **4.98 tok/s** | **392.6 ms** | **72.11%** | 10,282 fallos | $+2,363\text{ MB}$ | **Mejor TTFT del benchmark** |

---

## 5. Resultados de Adaptación de Cold Start (B4.4.6) [MEASURED]

Comparativa de los primeros 10 tokens en `Qwen1.5-MoE-A2.7B-Chat` con 12 slots/capa en VRAM:

| Métrica | Cold Start (VRAM Vacía) | Profile Preloaded (Top-12 en VRAM) | Impacto / Mejora [MEASURED] |
|---|---|---|---|
| **Latencia 1er Token (TTFT)** | **619.0 ms** | **312.6 ms** | **`1.98x más rápido`** |
| **Velocidad 1er Token** | **1.62 tok/s** | **3.20 tok/s** | **`+97.5% aceleración`** |
| **Hit Rate (10 tokens)** | **62.8%** | **69.0%** | **`+6.2 puntos porcentuales`** |
| **Fallos de Promoción PCIe** | 357 fallos | 298 fallos | **`-16.5% menos fallos`** |
| **Tráfico PCIe Ahorrado** | 1,975.4 MB | 1,656.5 MB | **`318.8 MB transferidos menos`** |
| **Tiempo de Pre-carga VRAM** | 0 ms | 1,676.8 ms (288 expertos) | One-time startup cost |

