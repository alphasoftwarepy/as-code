"""
AS-Core MoE Engine — Unit Tests for Subfase B4.3.6 (Frequency & Working Set Analysis)
=====================================================================================
Valida:
1. Conteo exacto de frecuencias por capa y cálculo de Top-N coverage.
2. Identificación del working set temporal y localidad (ventana 1, 4, 8).
3. Pares de co-activación de expertos dentro del Top-4 routing set.
4. Simulación offline determinista: Static Top-N vs LRU Offline.
5. Cálculo de VRAM física real contra el presupuesto de 4 GB.
6. Reproducibilidad determinista bit-a-bit del pipeline analítico.
"""

import json
import tempfile
from pathlib import Path
import pytest

from core.moe.frequency_analyzer import FrequencyAnalyzer

TRACE_PATH = r"C:\as-code\moe_poc\data\routing_trace.jsonl"
MODEL_PATH = r"C:\as-code\moe_poc\models\qwen1.5-moe-a2.7b-q4_k_m.gguf"


@pytest.fixture(scope="module")
def analyzer():
    return FrequencyAnalyzer(trace_path=TRACE_PATH, model_path=MODEL_PATH)


class TestFrequencyAnalysis:
    """Pruebas unitarias y analíticas para B4.3.6."""

    def test_b436_frequency_counting_and_top_n_coverage(self, analyzer):
        """Valida que las frecuencias sumen exactamente el total de decisiones x K."""
        freq_stats = analyzer.compute_frequency_per_layer()
        assert len(freq_stats) == 24

        total_activations_sum = sum(stats.total_activations for stats in freq_stats.values())
        assert total_activations_sum == 18432 * 4

        coverage_data = analyzer.compute_top_n_coverage([4, 8, 12, 30, 36])
        assert 0.0 < coverage_data["global_coverages"][4] < coverage_data["global_coverages"][8]
        assert coverage_data["global_coverages"][8] < coverage_data["global_coverages"][12]
        assert coverage_data["global_coverages"][30] == coverage_data["coverage_50_pct_residents"]
        assert coverage_data["coverage_50_pct_residents"] < coverage_data["coverage_60_pct_residents"]

    def test_b436_working_set_and_temporal_locality(self, analyzer):
        """Valida el análisis de crecimiento del working set y localidad temporal."""
        ws_data = analyzer.compute_working_set_growth([10, 25, 50, 100, 256])
        mean_sizes = ws_data["mean_working_set_size"]
        assert mean_sizes[10] < mean_sizes[25] < mean_sizes[50] <= mean_sizes[256]

        loc_data = analyzer.compute_temporal_locality()
        assert 0.0 < loc_data["global_token_to_token_overlap"] < 1.0
        assert loc_data["global_token_to_token_overlap"] <= loc_data["global_window_4_overlap"]
        assert loc_data["global_window_4_overlap"] <= loc_data["global_window_8_overlap"]

    def test_b436_coactivation_pairs(self, analyzer):
        """Valida la detección de pares de co-activación de expertos."""
        coact = analyzer.compute_coactivations(top_k_pairs=5)
        assert len(coact) == 24
        for layer_id in range(24):
            pairs = coact[layer_id]
            assert len(pairs) <= 5
            for p in pairs:
                assert len(p["pair"]) == 2
                assert p["count"] > 0
                assert 0.0 <= p["coactivation_pct"] <= 100.0

    def test_b436_hotset_simulation_and_vram_bounds(self, analyzer):
        """Valida la simulación comparativa Static vs LRU y los límites de VRAM en 4GB."""
        sim = analyzer.simulate_hotset_offline([4, 8, 12, 16, 30, 36])
        vram = analyzer.compute_vram_requirements([4, 8, 12, 16, 30, 36])

        for cap in [4, 8, 12, 16]:
            static_hit = sim["static_top_n"][cap]["hit_rate"]
            lru_hit = sim["lru_offline"][cap]["hit_rate"]
            # LRU dinámico debe superar significativamente a Static Top-N debido a los cambios de tarea
            assert lru_hit > static_hit, f"LRU hit rate ({lru_hit}) debió superar a Static ({static_hit}) para cap {cap}"

        # Validar límites de presupuesto VRAM
        assert vram["capacities"][4]["fits_in_4gb_budget"] is True
        assert vram["capacities"][8]["fits_in_4gb_budget"] is True
        assert vram["capacities"][12]["fits_in_4gb_budget"] is True
        assert vram["capacities"][16]["fits_in_4gb_budget"] is False
        assert vram["capacities"][30]["fits_in_4gb_budget"] is False

    def test_b436_deterministic_reproducibility(self, analyzer):
        """Valida que dos pasadas independientes sobre el mismo trace generen reportes idénticos."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dir1 = Path(tmpdir) / "run1"
            dir2 = Path(tmpdir) / "run2"

            rep1 = analyzer.generate_all_reports(dir1)
            rep2 = analyzer.generate_all_reports(dir2)

            for key in ["expert_frequency", "working_set_analysis", "hotset_simulation"]:
                with open(rep1[key], "r", encoding="utf-8") as f1, open(rep2[key], "r", encoding="utf-8") as f2:
                    assert f1.read() == f2.read(), f"Discrepancia en {key} entre ejecuciones!"
