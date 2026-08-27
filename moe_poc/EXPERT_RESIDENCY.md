# EXPERT_RESIDENCY.md
# AS-Core MoE Engine — Políticas y Mecánica de Residencia de Expertos
### Estado: Bloque B1 | Fecha: 2026-08-27

---

## 1. Principios de Ubicación de Tensores

```
             ┌──────────────────────────────────────────────────┐
             │                 MODELO MoE GGUF                  │
             └────────┬────────────────────────────────┬────────┘
                      │                                │
        ┌─────────────▼────────────┐     ┌─────────────▼────────────┐
        │     TENSORES DENSOS      │     │       TENSORES MoE       │
        │(Atención, Embed, Norms)  │     │(ffn_gate, up, down exps) │
        └─────────────┬────────────┘     └─────────────┬────────────┘
                      │                                │
                      │ [100% en VRAM]                 │ [Slicing por Experto]
                      │                                │
                      ▼                                ▼
        ┌──────────────────────────┐     ┌──────────────────────────┐
        │       VRAM (HOT)         │     │     ExpertRegistry       │
        │ - 24 Atenciones: 1.13 GB │     │   (1440 ExpertTensors)   │
        │ - KV Cache 2K:   0.35 GB │     └─────────────┬────────────┘
        │ - HOTSET EXPERTS:2.42 GB │                   │
        └──────────────────────────┘                   ├───► Top Frecuentes (17/capa) ──► VRAM [HOT]
                                                       ├───► Secundarios Mapped/Pinned ─► RAM  [WARM]
                                                       └───► Restantes en Disco ────────► NVMe [COLD]
```

---

## 2. Granularidad Física de Expertos

Para `Qwen1.5-MoE-A2.7B` (cuantización mixta Q4_K_M):
- **Capas iniciales / finales (Q6_K):** `6.02 MB` por experto completo.
- **Capas intermedias (Q4_K):** `5.00 MB` por experto completo.
- **Set de enrutamiento activo (4 expertos/capa):** ~`20.0–24.0 MB` por capa.
- **Mapeo:** La extracción ocurre por offsets continuos directamente desde el archivo GGUF mapeado en memoria, permitiendo cargar y transferir exclusivamente los bytes de los expertos requeridos.
