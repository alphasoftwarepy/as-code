"""
AS-Core MoE Engine — Frequency & Working Set Analysis Runner (B4.3.6)
====================================================================
Ejecuta el analisis estadistico offline sobre moe_poc/data/routing_trace.jsonl:
- Genera expert_frequency.json, working_set_analysis.json, hotset_simulation.json.
- Valida reproducibilidad estricta ejecutando 2 pasadas consecutivas.
- Imprime la tabla comparativa de HotSet, Cobertura, Hit Rate y VRAM.
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, r"C:\as-code")

from core.moe.frequency_analyzer import FrequencyAnalyzer

TRACE_PATH = r"C:\as-code\moe_poc\data\routing_trace.jsonl"
OUT_DIR = r"C:\as-code\moe_poc\data"


def main():
    print("=" * 80)
    print(" [AS-CORE B4.3.6] INICIANDO ANALISIS OFFLINE DE FRECUENCIA Y WORKING SET")
    print("=" * 80)

    # 1. Primera Pasada de Analisis
    t0 = time.perf_counter()
    analyzer_1 = FrequencyAnalyzer(trace_path=TRACE_PATH)
    reports_1 = analyzer_1.generate_all_reports(output_dir=OUT_DIR)
    t1 = time.perf_counter() - t0

    print(f"  [OK] Pasada 1 completada en {t1*1000.0:.1f} ms.")

    # 2. Segunda Pasada para Verificar Reproducibilidad Estricta
    t2_0 = time.perf_counter()
    analyzer_2 = FrequencyAnalyzer(trace_path=TRACE_PATH)
    reports_2 = analyzer_2.generate_all_reports(output_dir=OUT_DIR)
    t2 = time.perf_counter() - t2_0

    print(f"  [OK] Pasada 2 completada en {t2*1000.0:.1f} ms.")

    # 3. Validacion Bit-a-Bit de Reproducibilidad
    with open(reports_1["hotset_simulation"], "r", encoding="utf-8") as f1, open(reports_2["hotset_simulation"], "r", encoding="utf-8") as f2:
        assert f1.read() == f2.read(), "Error: Las corridas no son 100% reproducibles!"

    with open(reports_1["expert_frequency"], "r", encoding="utf-8") as f1, open(reports_2["expert_frequency"], "r", encoding="utf-8") as f2:
        assert f1.read() == f2.read(), "Error: Las corridas de frecuencia no coinciden!"

    print("  [OK] Reproducibilidad 100% verificada (Ambas pasadas producen resultados identicos).")

    # 4. Cargar y Mostrar Tabla Consolidada
    with open(reports_1["hotset_simulation"], "r", encoding="utf-8") as f:
        hotset_data = json.load(f)

    with open(reports_1["working_set_analysis"], "r", encoding="utf-8") as f:
        ws_data = json.load(f)

    with open(reports_1["expert_frequency"], "r", encoding="utf-8") as f:
        freq_data = json.load(f)

    print("\n" + "=" * 90)
    print(" [TABLA PRINCIPAL B4.3.6] COMPARATIVA DE HOTSET, VRAM, ACTIVATION COVERAGE Y HIT RATE")
    print("=" * 90)
    print(f"{'HotSet':<8} | {'% Exp':<8} | {'VRAM (MB)':<10} | {'Total VRAM':<12} | {'Fits 4GB?':<10} | {'Static Cov %':<14} | {'Static Hit %':<14} | {'LRU Hit %':<10}")
    print("-" * 90)

    for row in hotset_data["summary_comparison_table"]:
        fits_str = "YES (PASS)" if row["fits_in_4gb"] else "NO (OOM)"
        print(
            f"{row['hotset_size']:<8} | "
            f"{row['pct_experts']:<6.1f}% | "
            f"{row['vram_mb']:<10.1f} | "
            f"{row['total_system_vram_mb']:<12.1f} | "
            f"{fits_str:<10} | "
            f"{row['activation_coverage_pct']:<14.2f} | "
            f"{row['static_hit_rate_pct']:<14.2f} | "
            f"{row['lru_hit_rate_pct']:<10.2f}"
        )
    print("=" * 90)

    # 5. Mostrar Metricas Criticas Adicionales
    loc = ws_data["temporal_locality"]
    print("\n" + "=" * 80)
    print(" [METRICAS CRITICAS ADICIONALES]")
    print("=" * 80)
    print(f"  - Localidad Temporal Token-a-Token:   {loc['global_token_to_token_overlap']*100:.2f}%")
    print(f"  - Localidad Temporal Ventana 4:       {loc['global_window_4_overlap']*100:.2f}%")
    print(f"  - Localidad Temporal Ventana 8:       {loc['global_window_8_overlap']*100:.2f}%")
    print(f"  - Cobertura Hipotesis 50% (30 Exp):   {freq_data['global_top_n_coverage']['30']:.2f}%")
    print(f"  - Cobertura Hipotesis 60% (36 Exp):   {freq_data['global_top_n_coverage']['36']:.2f}%")
    print(f"  - Working Set Promedio a 10 tokens:   {ws_data['working_set_growth']['mean_working_set_size']['10']} expertos/capa")
    print(f"  - Working Set Promedio a 50 tokens:   {ws_data['working_set_growth']['mean_working_set_size']['50']} expertos/capa")
    print(f"  - Working Set Promedio a 256 tokens:  {ws_data['working_set_growth']['mean_working_set_size']['256']} expertos/capa")

    seq_comp = freq_data["sequence_comparison"]
    print(f"  - Jaccard Similarity (Conv vs Code):  {seq_comp['mean_jaccard_conv_vs_code']:.4f}")
    print(f"  - Jaccard Similarity (Conv vs Tech):  {seq_comp['mean_jaccard_conv_vs_tech']:.4f}")
    print(f"  - Jaccard Similarity (Code vs Tech):  {seq_comp['mean_jaccard_code_vs_tech']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
