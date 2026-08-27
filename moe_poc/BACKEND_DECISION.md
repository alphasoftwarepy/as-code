# BACKEND_DECISION.md
# AS-Core MoE POC — Decisión Formal de Backend
# FASE 0 / CONDICIÓN 1 & CONDICIÓN 2 — COMPLETADO

## Estado: COMPLETADO Y SATISFECHO ✅

---

## 1. Hardware Verificado Empíricamente

| Componente | Especificación Medida |
|---|---|
| GPU | **NVIDIA GeForce GTX 1650 Ti** |
| VRAM Total | **4096 MB** |
| VRAM Libre (idle) | **3935 MB** |
| VRAM Ocupada por Modelo | **3892 MB** (100% offload a GPU con `-ngl 99`) |
| Driver NVIDIA | **595.71** |
| CUDA Capability | **7.5** (Turing / sm_75) |
| RAM Total | **15 GB** (disponible ~5 GB) |
| Sistema Operativo | **Windows 11 (x64) + WSL2 Ubuntu disponible** |

---

## 2. Resultados Empíricos del Benchmark (Condición 1)

**Modelo de prueba:** `OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf` (6.9B total / 1.0B activos, 64 expertos / 8 activos por token).
**Backend evaluado:** `llama-server.exe` build 10649 (commit `2bb9bddaf`), CUDA 12.4 Windows x64 nativo.

| Métrica | Resultado Medido | Target del Gate | Estado |
|---|---|---|---|
| **Velocidad de generación (Run 1)** | **65.40 tok/s** | $\ge 10$ tok/s | ✅ **PASS** (>6.5x) |
| **Velocidad de generación (Run 2)** | **64.92 tok/s** | $\ge 10$ tok/s | ✅ **PASS** (>6.4x) |
| **Velocidad de generación (Run 3)** | **65.79 tok/s** | $\ge 10$ tok/s | ✅ **PASS** (>6.5x) |
| **Promedio sostenido** | **65.37 tok/s** | $\ge 10$ tok/s | ✅ **PASS CRÍTICO** |
| **Time-To-First-Token (TTFT)** | **55.7 ms** | $< 500$ ms | ✅ **Excelente** |
| **Tiempo de carga inicial** | **35.2 s** | N/A | Normal (carga completa) |
| **GPU Utilization activa** | **94%** | Compute en GPU | ✅ **GPU-First verificado** |
| **VRAM Remanente tras carga** | **43 MB** | $> 0$ MB | ✅ Cabe íntegramente |

---

## 3. Evaluación de Opciones de Backend (Condición 2)

| Criterio | `llama-server` (CUDA Windows Nativo) | `moe-l2` (Vía WSL2) |
|---|---|---|
| **Ejecución en Windows** | Nativo sin virtualización ni overhead IPC | Requiere runtime WSL2 activo y puente de sockets |
| **Velocidad empírica demostrada** | **65.37 tok/s** en hardware real | Estimada ~50–65 tok/s (overhead de capa virtualizada) |
| **Compatibilidad con AS-Core** | Inicia como subproceso Windows directo | Requiere comando wrapper `wsl` |
| **Estabilidad de proceso** | Alta (binario oficial `ggml-org` b10649) | Media (fork comunitario enfocado en Linux) |
| **Uso para POC (Horizonte 1)** | **Óptimo y validado** | Opcional como motor secundario para Horizonte 2 |

---

## 4. Decisión Formal

```
BACKEND SELECCIONADO PARA HORIZONTE 1: llamacpp (CUDA Windows Nativo)

Binario: C:/as-code/moe_poc/bins/llama-server.exe
Modo: backend_mode: "llamacpp"
```

### Justificación:
1. **Rendimiento superó ampliamente el gate:** 65.37 tok/s representa más de 6.5 veces el mínimo de 10 tok/s establecido.
2. **Cero fricción de arquitectura:** Al ejecutarse nativamente en Windows vía CUDA 12.4, no introduce complejidades de WSL2 ni dependencias de virtualización.
3. **Desacoplamiento total:** La interfaz `InferenceBackend` definida en el plan mantiene la abstracción abierta para conectar `moe-l2` en Horizonte 2 sin modificar el `MoEInferenceProvider`.

---

## 5. Dictamen de Gates de Fase 0

- **Condición 1 (Benchmark empírico $\ge 10$ tok/s):** ✅ **SUPERADA CON ÉXITO (65.37 tok/s)**
- **Condición 2 (Decisión de Backend documentada):** ✅ **SUPERADA CON ÉXITO (`llamacpp` CUDA nativo)**

*Fecha de validación: 2026-08-27*
