"""
AS-Core MoE Engine — Frequency & Working Set Analyzer (B4.3.6)
==============================================================
Módulo analítico offline para estudiar el comportamiento real de enrutamiento:
- Frecuencia y dispersión de activaciones por capa (layer_id, expert_id).
- Top-N Activation Coverage vs Cache Hit Rate.
- Validación empírica de la hipótesis de residencia 50%-60%.
- Crecimiento del Working Set temporal (10, 25, 50, 100, 256 tokens).
- Localidad temporal (Overlap token-a-token, ventana 4 y ventana 8).
- Co-activación de pares de expertos para prefetching.
- Simulación Offline de HotSet: Static Top-N vs LRU Dinámico.
- Cálculo de VRAM real y compatibilidad con el presupuesto de 4 GB (GTX 1650 Ti).
"""

from __future__ import annotations

import json
import logging
from collections import Counter, OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from core.moe.expert_registry import ExpertRegistry

logger = logging.getLogger("as-code.core.moe.analyzer")


@dataclass
class LayerFrequencyStats:
    """Estadísticas de frecuencia para una capa específica."""
    layer_id: int
    total_tokens: int
    total_activations: int
    unique_experts_used: int
    min_frequency: int
    max_frequency: int
    mean_frequency: float
    median_frequency: float
    std_dev_frequency: float
    expert_counts: Dict[int, int]
    expert_percentages: Dict[int, float]
    sorted_experts: List[int]


class FrequencyAnalyzer:
    """Analizador estadístico exhaustivo del dataset de trazas de enrutamiento."""

    def __init__(
        self,
        trace_path: Union[str, Path] = r"C:\as-code\moe_poc\data\routing_trace.jsonl",
        model_path: Union[str, Path] = r"C:\as-code\moe_poc\models\qwen1.5-moe-a2.7b-q4_k_m.gguf",
    ):
        self.trace_path = Path(trace_path)
        self.model_path = Path(model_path)
        self.registry = ExpertRegistry(str(self.model_path))

        self.profile = self.registry.profile
        self.num_layers = self.profile.block_count  # 24
        self.num_experts_per_layer = self.profile.expert_count  # 60
        self.k_active = self.profile.expert_used_count  # 4

        # Estructuras de traza parseadas
        self.events: List[Dict[str, Any]] = []
        self.events_by_seq_layer: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
        self.events_by_layer: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        self.sequences: Set[int] = set()

        self._load_trace()

    def _load_trace(self) -> None:
        """Carga y valida el archivo JSONL de trazas."""
        if not self.trace_path.exists():
            raise FileNotFoundError(f"Archivo de traza no encontrado: {self.trace_path}")

        with open(self.trace_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line_str = line.strip()
                if not line_str:
                    continue
                ev = json.loads(line_str)
                self.events.append(ev)
                seq_id = ev["seq"]
                layer_id = ev["layer"]
                self.sequences.add(seq_id)
                self.events_by_seq_layer[(seq_id, layer_id)].append(ev)
                self.events_by_layer[layer_id].append(ev)

        logger.info(
            f"[FrequencyAnalyzer] Cargados {len(self.events):,} eventos de traza "
            f"({len(self.sequences)} secuencias, {self.num_layers} capas)."
        )

    # =========================================================================
    # 1. FRECUENCIA POR CAPA
    # =========================================================================
    def compute_frequency_per_layer(self) -> Dict[int, LayerFrequencyStats]:
        """Calcula la frecuencia y distribución de activación para cada experto en cada capa."""
        results: Dict[int, LayerFrequencyStats] = {}

        for layer_id in range(self.num_layers):
            layer_events = self.events_by_layer[layer_id]
            total_tokens = len(layer_events)
            total_activations = total_tokens * self.k_active

            counts: Dict[int, int] = {e: 0 for e in range(self.num_experts_per_layer)}
            for ev in layer_events:
                for exp_id in ev["experts"]:
                    counts[exp_id] = counts.get(exp_id, 0) + 1

            sorted_by_freq = sorted(counts.keys(), key=lambda e: counts[e], reverse=True)
            freq_values = [counts[e] for e in range(self.num_experts_per_layer)]

            percentages = {
                e: round((counts[e] / total_activations) * 100.0, 4) if total_activations > 0 else 0.0
                for e in range(self.num_experts_per_layer)
            }

            unique_used = sum(1 for c in freq_values if c > 0)

            stats = LayerFrequencyStats(
                layer_id=layer_id,
                total_tokens=total_tokens,
                total_activations=total_activations,
                unique_experts_used=unique_used,
                min_frequency=int(np.min(freq_values)),
                max_frequency=int(np.max(freq_values)),
                mean_frequency=float(np.mean(freq_values)),
                median_frequency=float(np.median(freq_values)),
                std_dev_frequency=float(np.std(freq_values)),
                expert_counts=counts,
                expert_percentages=percentages,
                sorted_experts=sorted_by_freq,
            )
            results[layer_id] = stats

        return results

    # =========================================================================
    # 2. TOP-N ACTIVATION COVERAGE & HIPÓTESIS 50%/60%
    # =========================================================================
    def compute_top_n_coverage(
        self,
        capacities: List[int] = [4, 8, 12, 16, 20, 24, 30, 36],
    ) -> Dict[str, Any]:
        """Calcula el porcentaje de activaciones totales cubiertas por los Top-N expertos más frecuentes."""
        freq_stats = self.compute_frequency_per_layer()
        layer_coverages: Dict[int, Dict[int, float]] = {}
        global_act_covered: Dict[int, int] = {c: 0 for c in capacities}
        total_global_activations = sum(stats.total_activations for stats in freq_stats.values())

        for layer_id, stats in freq_stats.items():
            layer_coverages[layer_id] = {}
            for cap in capacities:
                top_n_experts = stats.sorted_experts[:cap]
                act_covered = sum(stats.expert_counts[e] for e in top_n_experts)
                coverage_pct = (act_covered / stats.total_activations) * 100.0 if stats.total_activations > 0 else 0.0
                layer_coverages[layer_id][cap] = round(coverage_pct, 2)
                global_act_covered[cap] += act_covered

        global_coverages = {
            cap: round((global_act_covered[cap] / total_global_activations) * 100.0, 2)
            if total_global_activations > 0
            else 0.0
            for cap in capacities
        }

        return {
            "capacities": capacities,
            "layer_coverages": layer_coverages,
            "global_coverages": global_coverages,
            "coverage_50_pct_residents": global_coverages.get(30, 0.0),
            "coverage_60_pct_residents": global_coverages.get(36, 0.0),
        }

    # =========================================================================
    # 3. UNIQUE EXPERTS & WORKING SET GROWTH
    # =========================================================================
    def compute_working_set_growth(
        self,
        token_checkpoints: List[int] = [10, 25, 50, 100, 256],
    ) -> Dict[str, Any]:
        """Analiza la evolución del número de expertos únicos activados a lo largo del tiempo."""
        results_by_seq: Dict[int, Dict[int, Dict[int, int]]] = {}  # seq -> checkpoint -> layer -> unique_count
        aggregated_by_checkpoint: Dict[int, Dict[int, int]] = {cp: {} for cp in token_checkpoints}

        for seq_id in sorted(list(self.sequences)):
            results_by_seq[seq_id] = {}
            for cp in token_checkpoints:
                results_by_seq[seq_id][cp] = {}
                for layer_id in range(self.num_layers):
                    layer_events = self.events_by_seq_layer[(seq_id, layer_id)]
                    events_slice = layer_events[:cp]
                    unique_set: Set[int] = set()
                    for ev in events_slice:
                        unique_set.update(ev["experts"])
                    results_by_seq[seq_id][cp][layer_id] = len(unique_set)

        # Agregado global (acumulativo a través de todas las secuencias hasta el checkpoint)
        for cp in token_checkpoints:
            for layer_id in range(self.num_layers):
                unique_set_global: Set[int] = set()
                for seq_id in self.sequences:
                    events_slice = self.events_by_seq_layer[(seq_id, layer_id)][:cp]
                    for ev in events_slice:
                        unique_set_global.update(ev["experts"])
                aggregated_by_checkpoint[cp][layer_id] = len(unique_set_global)

        mean_working_set_by_checkpoint = {
            cp: round(float(np.mean(list(aggregated_by_checkpoint[cp].values()))), 2)
            for cp in token_checkpoints
        }

        return {
            "token_checkpoints": token_checkpoints,
            "by_sequence": results_by_seq,
            "aggregated": aggregated_by_checkpoint,
            "mean_working_set_size": mean_working_set_by_checkpoint,
        }

    # =========================================================================
    # 4. TEMPORAL LOCALITY
    # =========================================================================
    def compute_temporal_locality(self) -> Dict[str, Any]:
        """Calcula el solapamiento temporal de expertos entre tokens adyacentes y ventanas de tamaño 4 y 8."""
        layer_overlap_1: Dict[int, List[float]] = defaultdict(list)
        layer_overlap_4: Dict[int, List[float]] = defaultdict(list)
        layer_overlap_8: Dict[int, List[float]] = defaultdict(list)

        for seq_id in self.sequences:
            for layer_id in range(self.num_layers):
                events = self.events_by_seq_layer[(seq_id, layer_id)]
                if not events:
                    continue

                for t in range(len(events)):
                    curr_experts = set(events[t]["experts"])

                    # 1-step lag overlap (token t vs token t-1)
                    if t >= 1:
                        prev_1 = set(events[t - 1]["experts"])
                        overlap_1 = len(curr_experts & prev_1) / len(curr_experts)
                        layer_overlap_1[layer_id].append(overlap_1)

                    # 4-window overlap (token t vs union(t-4..t-1))
                    if t >= 4:
                        win_4: Set[int] = set()
                        for k in range(1, 5):
                            win_4.update(events[t - k]["experts"])
                        overlap_4 = len(curr_experts & win_4) / len(curr_experts)
                        layer_overlap_4[layer_id].append(overlap_4)

                    # 8-window overlap (token t vs union(t-8..t-1))
                    if t >= 8:
                        win_8: Set[int] = set()
                        for k in range(1, 9):
                            win_8.update(events[t - k]["experts"])
                        overlap_8 = len(curr_experts & win_8) / len(curr_experts)
                        layer_overlap_8[layer_id].append(overlap_8)

        mean_overlap_1_by_layer = {l: round(float(np.mean(v)), 4) if v else 0.0 for l, v in layer_overlap_1.items()}
        mean_overlap_4_by_layer = {l: round(float(np.mean(v)), 4) if v else 0.0 for l, v in layer_overlap_4.items()}
        mean_overlap_8_by_layer = {l: round(float(np.mean(v)), 4) if v else 0.0 for l, v in layer_overlap_8.items()}

        global_overlap_1 = round(float(np.mean([float(np.mean(v)) for v in layer_overlap_1.values() if v])), 4)
        global_overlap_4 = round(float(np.mean([float(np.mean(v)) for v in layer_overlap_4.values() if v])), 4)
        global_overlap_8 = round(float(np.mean([float(np.mean(v)) for v in layer_overlap_8.values() if v])), 4)

        return {
            "global_token_to_token_overlap": global_overlap_1,
            "global_window_4_overlap": global_overlap_4,
            "global_window_8_overlap": global_overlap_8,
            "mean_overlap_1_by_layer": mean_overlap_1_by_layer,
            "mean_overlap_4_by_layer": mean_overlap_4_by_layer,
            "mean_overlap_8_by_layer": mean_overlap_8_by_layer,
        }

    # =========================================================================
    # 5. CO-ACTIVACIÓN DE EXPERTOS
    # =========================================================================
    def compute_coactivations(self, top_k_pairs: int = 5) -> Dict[int, List[Dict[str, Any]]]:
        """Determina los pares de expertos que se co-activan con mayor frecuencia simultáneamente."""
        results: Dict[int, List[Dict[str, Any]]] = {}

        for layer_id in range(self.num_layers):
            layer_events = self.events_by_layer[layer_id]
            pair_counts: Counter[Tuple[int, int]] = Counter()
            total_decisions = len(layer_events)

            for ev in layer_events:
                exp_list = sorted(ev["experts"])
                for pair in combinations(exp_list, 2):
                    pair_counts[pair] += 1

            most_common = pair_counts.most_common(top_k_pairs)
            results[layer_id] = [
                {
                    "pair": list(pair),
                    "count": count,
                    "coactivation_pct": round((count / total_decisions) * 100.0, 2) if total_decisions > 0 else 0.0,
                }
                for pair, count in most_common
            ]

        return results

    # =========================================================================
    # 6. HOTSET OFFLINE SIMULATION (STATIC TOP-N vs LRU OFFLINE)
    # =========================================================================
    def simulate_hotset_offline(
        self,
        capacities: List[int] = [4, 8, 12, 16, 20, 24, 30, 36],
    ) -> Dict[str, Any]:
        """Ejecuta una simulación determinista offline comparando Static Top-N contra LRU dinámico."""
        freq_stats = self.compute_frequency_per_layer()

        static_results: Dict[int, Dict[str, Any]] = {}
        lru_results: Dict[int, Dict[str, Any]] = {}

        for cap in capacities:
            # 1. Simulación Static Top-N
            static_hits = 0
            static_misses = 0

            # 2. Simulación LRU por capa
            lru_hits = 0
            lru_misses = 0
            lru_promotions = 0

            # Inicializar caches LRU por (seq_id, layer_id)
            for seq_id in sorted(list(self.sequences)):
                for layer_id in range(self.num_layers):
                    events = self.events_by_seq_layer[(seq_id, layer_id)]
                    static_set = set(freq_stats[layer_id].sorted_experts[:cap])

                    # LRU cache: OrderedDict acting as an LRU queue (max capacity = cap)
                    lru_cache: OrderedDict[int, bool] = OrderedDict()

                    for ev in events:
                        for exp_id in ev["experts"]:
                            # Static check
                            if exp_id in static_set:
                                static_hits += 1
                            else:
                                static_misses += 1

                            # LRU check
                            if exp_id in lru_cache:
                                lru_hits += 1
                                lru_cache.move_to_end(exp_id)
                            else:
                                lru_misses += 1
                                lru_promotions += 1
                                if len(lru_cache) >= cap:
                                    lru_cache.popitem(last=False)  # Desalojo del más antiguo (FIFO/LRU head)
                                lru_cache[exp_id] = True

            total_static_requests = static_hits + static_misses
            total_lru_requests = lru_hits + lru_misses

            static_results[cap] = {
                "capacity": cap,
                "hits": static_hits,
                "misses": static_misses,
                "hit_rate": round((static_hits / total_static_requests) * 100.0, 2) if total_static_requests > 0 else 0.0,
            }

            lru_results[cap] = {
                "capacity": cap,
                "hits": lru_hits,
                "misses": lru_misses,
                "promotions": lru_promotions,
                "hit_rate": round((lru_hits / total_lru_requests) * 100.0, 2) if total_lru_requests > 0 else 0.0,
            }

        return {
            "capacities": capacities,
            "static_top_n": static_results,
            "lru_offline": lru_results,
        }

    # =========================================================================
    # 7. VRAM REAL Y PRESUPUESTO DE 4 GB
    # =========================================================================
    def compute_vram_requirements(
        self,
        capacities: List[int] = [4, 8, 12, 16, 20, 24, 30, 36],
    ) -> Dict[str, Any]:
        """Calcula el consumo exacto de VRAM para cada tamaño de HotSet basado en el tamaño real de ExpertTensor."""
        sample_exp = self.registry.get_expert(0, 0)
        bytes_per_expert = sample_exp.total_bytes  # 6,307,840 bytes (6.0156 MB)
        mb_per_expert = bytes_per_expert / (1024 * 1024)

        vram_by_capacity: Dict[int, Dict[str, Any]] = {}

        # Presupuesto físico oficial GTX 1650 Ti:
        # Total VRAM física: 4096 MB
        # Componentes base no-MoE en VRAM:
        #   - Capas Attention + Embeddings + Output Head (10 capas GPU de Baseline-1A): ~1,480 MB
        #   - 24 Matrices de Router en VRAM: 11.25 MB
        #   - KV Cache (ctx=2048): ~320 MB
        #   - Buffers de activación CUDA + cuBLAS workspace: ~150 MB
        #   - Driver / OS overhead: ~250 MB
        # Base VRAM fija = 2,211 MB
        # Margen disponible para HotSet MoE = 4,096 - 2,211 = 1,885 MB
        vram_base_fixed_mb = 1480.0 + 11.25 + 320.0 + 150.0 + 250.0  # 2,211.25 MB
        vram_total_card_mb = 4096.0
        vram_budget_for_hotset_mb = vram_total_card_mb - vram_base_fixed_mb  # 1,884.75 MB

        for cap in capacities:
            total_resident_experts = cap * self.num_layers
            total_bytes = total_resident_experts * bytes_per_expert
            total_mb = total_bytes / (1024 * 1024)
            pct_experts = round((cap / self.num_experts_per_layer) * 100.0, 1)

            total_system_vram_mb = vram_base_fixed_mb + total_mb
            fits_in_4gb = total_system_vram_mb <= vram_total_card_mb

            vram_by_capacity[cap] = {
                "hotset_size": cap,
                "pct_experts": pct_experts,
                "total_resident_experts": total_resident_experts,
                "hotset_vram_mb": round(total_mb, 2),
                "total_system_vram_mb": round(total_system_vram_mb, 2),
                "fits_in_4gb_budget": fits_in_4gb,
                "vram_headroom_mb": round(vram_total_card_mb - total_system_vram_mb, 2),
            }

        return {
            "bytes_per_expert": bytes_per_expert,
            "mb_per_expert": round(mb_per_expert, 4),
            "vram_base_fixed_mb": round(vram_base_fixed_mb, 2),
            "vram_budget_for_hotset_mb": round(vram_budget_for_hotset_mb, 2),
            "capacities": vram_by_capacity,
        }

    # =========================================================================
    # 8. COMPARACIÓN ENTRE SECUENCIAS
    # =========================================================================
    def compare_sequences(self, top_n: int = 12) -> Dict[str, Any]:
        """Compara la dispersión y el solapamiento de los expertos más frecuentes entre distintas secuencias temáticas."""
        seq_names = {
            0: "Seq 0 (Dialogo Conversacional)",
            1: "Seq 1 (Generacion de Codigo)",
            2: "Seq 2 (Documentacion Tecnica)",
        }

        seq_top_experts: Dict[int, Dict[int, Set[int]]] = {}  # seq -> layer -> set of top-N experts

        for seq_id in self.sequences:
            seq_top_experts[seq_id] = {}
            for layer_id in range(self.num_layers):
                events = self.events_by_seq_layer[(seq_id, layer_id)]
                counts: Counter[int] = Counter()
                for ev in events:
                    counts.update(ev["experts"])
                top_experts = set([e for e, _ in counts.most_common(top_n)])
                seq_top_experts[seq_id][layer_id] = top_experts

        # Calcular Jaccard similarity promedio entre pares de secuencias
        jaccard_0_1_list: List[float] = []
        jaccard_0_2_list: List[float] = []
        jaccard_1_2_list: List[float] = []

        for layer_id in range(self.num_layers):
            s0 = seq_top_experts.get(0, {}).get(layer_id, set())
            s1 = seq_top_experts.get(1, {}).get(layer_id, set())
            s2 = seq_top_experts.get(2, {}).get(layer_id, set())

            if s0 and s1:
                jaccard_0_1_list.append(len(s0 & s1) / len(s0 | s1))
            if s0 and s2:
                jaccard_0_2_list.append(len(s0 & s2) / len(s0 | s2))
            if s1 and s2:
                jaccard_1_2_list.append(len(s1 & s2) / len(s1 | s2))

        return {
            "top_n_analyzed": top_n,
            "mean_jaccard_conv_vs_code": round(float(np.mean(jaccard_0_1_list)), 4) if jaccard_0_1_list else 0.0,
            "mean_jaccard_conv_vs_tech": round(float(np.mean(jaccard_0_2_list)), 4) if jaccard_0_2_list else 0.0,
            "mean_jaccard_code_vs_tech": round(float(np.mean(jaccard_1_2_list)), 4) if jaccard_1_2_list else 0.0,
        }

    # =========================================================================
    # 9. EXPORTACIÓN DE REPORTES JSON
    # =========================================================================
    def generate_all_reports(self, output_dir: Union[str, Path] = r"C:\as-code\moe_poc\data") -> Dict[str, Path]:
        """Genera todos los reportes analíticos JSON requeridos en B4.3.6."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "model": self.profile.model_name,
            "architecture": "Qwen2MoE (SwiGLU, Float32 Projections, Q4_K/Q6_K)",
            "layers": self.num_layers,
            "experts_per_layer": self.num_experts_per_layer,
            "total_experts": self.profile.total_experts_in_model,
            "top_k": self.k_active,
            "trace_source": str(self.trace_path),
            "total_events": len(self.events),
            "sequence_count": len(self.sequences),
            "tokens_per_sequence": len(self.events) // (len(self.sequences) * self.num_layers) if self.sequences else 0,
        }

        # 1. expert_frequency.json
        freq_stats = self.compute_frequency_per_layer()
        coverage_data = self.compute_top_n_coverage()
        coact_data = self.compute_coactivations()
        seq_comp = self.compare_sequences()

        freq_export = {
            "metadata": meta,
            "global_top_n_coverage": coverage_data["global_coverages"],
            "sequence_comparison": seq_comp,
            "layers": {
                layer_id: {
                    "unique_experts_used": stats.unique_experts_used,
                    "min_frequency": stats.min_frequency,
                    "max_frequency": stats.max_frequency,
                    "mean_frequency": round(stats.mean_frequency, 2),
                    "median_frequency": round(stats.median_frequency, 2),
                    "std_dev": round(stats.std_dev_frequency, 2),
                    "top_n_coverage": coverage_data["layer_coverages"].get(layer_id, {}),
                    "sorted_experts_by_frequency": stats.sorted_experts,
                    "expert_percentages": {str(k): v for k, v in stats.expert_percentages.items()},
                    "top_coactivations": coact_data.get(layer_id, []),
                }
                for layer_id, stats in freq_stats.items()
            },
        }
        freq_path = out_dir / "expert_frequency.json"
        with open(freq_path, "w", encoding="utf-8") as f:
            json.dump(freq_export, f, indent=2)

        # 2. working_set_analysis.json
        ws_data = self.compute_working_set_growth()
        locality_data = self.compute_temporal_locality()
        ws_export = {
            "metadata": meta,
            "temporal_locality": locality_data,
            "working_set_growth": ws_data,
        }
        ws_path = out_dir / "working_set_analysis.json"
        with open(ws_path, "w", encoding="utf-8") as f:
            json.dump(ws_export, f, indent=2)

        # 3. hotset_simulation.json
        sim_data = self.simulate_hotset_offline()
        vram_data = self.compute_vram_requirements()

        # Construir tabla comparativa consolidada
        summary_table: List[Dict[str, Any]] = []
        for cap in sim_data["capacities"]:
            static_info = sim_data["static_top_n"][cap]
            lru_info = sim_data["lru_offline"][cap]
            vram_info = vram_data["capacities"][cap]
            act_cov = coverage_data["global_coverages"].get(cap, 0.0)

            summary_table.append({
                "hotset_size": cap,
                "pct_experts": vram_info["pct_experts"],
                "vram_mb": vram_info["hotset_vram_mb"],
                "total_system_vram_mb": vram_info["total_system_vram_mb"],
                "fits_in_4gb": vram_info["fits_in_4gb_budget"],
                "activation_coverage_pct": act_cov,
                "static_hit_rate_pct": static_info["hit_rate"],
                "lru_hit_rate_pct": lru_info["hit_rate"],
                "lru_promotions_count": lru_info["promotions"],
            })

        hotset_export = {
            "metadata": meta,
            "vram_parameters": {
                "bytes_per_expert": vram_data["bytes_per_expert"],
                "mb_per_expert": vram_data["mb_per_expert"],
                "vram_base_fixed_mb": vram_data["vram_base_fixed_mb"],
                "vram_budget_for_hotset_mb": vram_data["vram_budget_for_hotset_mb"],
            },
            "summary_comparison_table": summary_table,
            "static_top_n_details": sim_data["static_top_n"],
            "lru_offline_details": sim_data["lru_offline"],
        }
        hotset_path = out_dir / "hotset_simulation.json"
        with open(hotset_path, "w", encoding="utf-8") as f:
            json.dump(hotset_export, f, indent=2)

        return {
            "expert_frequency": freq_path,
            "working_set_analysis": ws_path,
            "hotset_simulation": hotset_path,
        }
