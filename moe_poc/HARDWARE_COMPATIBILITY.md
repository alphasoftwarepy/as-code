# HARDWARE_COMPATIBILITY.md
# AS-Core MoE Engine — Perfil de Hardware y Presupuesto de Memoria
### Fecha: 2026-08-27

---

## 1. Hardware Objetivo Verificado

| Dispositivo | Especificación Real | Capacidad Medida |
|---|---|---|
| **GPU** | NVIDIA GeForce GTX 1650 Ti | 4096 MB VRAM GDDR6 (3935 MB libre en idle) |
| **Bus PCIe** | PCIe 3.0 x16 / x8 | **10.88 GB/s** (Pinned DMA) / **7.09 GB/s** (Pageable) |
| **CPU** | Intel Core i5-10300H @ 2.50GHz | 4 núcleos físicos / 8 lógicos (AVX2, FMA) |
| **RAM** | 16 GB DDR4 (15.84 GB utilizables) | ~25 GB/s ancho de banda de sistema |
| **Storage** | NVMe SSD PCIe 3.0 | ~2400 MB/s lectura secuencial |
| **OS** | Windows 11 x64 | Driver NVIDIA 595.71, CUDA 12.4 |

---

## 2. Presupuesto Dinámico de Memoria (Hardware Budget Formula)

$$\text{VRAM}_{\text{HotSet}} = \text{VRAM}_{\text{Total}} - \text{VRAM}_{\text{Overhead}} - \text{VRAM}_{\text{Dense}} - \text{VRAM}_{\text{KV\_Cache}}$$

Para la GTX 1650 Ti (4096 MB):
- $\text{VRAM}_{\text{Overhead}} = 150.0\text{ MB}$
- $\text{VRAM}_{\text{Dense}} = 1130.5\text{ MB}$
- $\text{VRAM}_{\text{KV\_Cache}} = 350.0\text{ MB}$
- $\text{VRAM}_{\text{HotSet}} = \mathbf{2465.5\text{ MB}}$

$$\text{Capacidad HotSet} = \left\lfloor \frac{2465.5\text{ MB}}{6.02\text{ MB/experto}} \right\rfloor = \mathbf{409\text{ expertos}} \implies \mathbf{17\text{ expertos / capa}}$$
