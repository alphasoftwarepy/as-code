"""
Controlled Real End-to-End Test for AS-Core Experimental Skill Factory.

Creates a temporary skill 'csv_data_extractor', executes validation on non-sensitive data,
verifies sandboxing & zero contamination, and generates the proposal markdown.
"""

import csv
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from runtime.skills.temporary import SkillSpec, SkillTestCase
from runtime.skills.factory import SkillFactory
from runtime.skills.loader import SkillLoader



def main():
    print("=== STARTING CONTROLLED REAL END-TO-END TEST ===")
    
    # 1. Initialize Factory
    factory = SkillFactory()

    # 2. Define Spec for csv_data_extractor
    spec = SkillSpec(
        skill_id="csv_data_extractor",
        name="CSV Data Extractor",
        description="Extracts tabular structured data and summaries from CSV inputs without data loss",
        objective="Parse structured CSV data, extract row summaries, compute aggregate metrics (totals, counts), and output schema-validated JSON.",
        recipe=(
            "1. Instantiate TemporarySkill sandbox in temp_skills/csv_data_extractor/\n"
            "2. Generate manifest.json declaring documents.read scope\n"
            "3. Write system prompt in instructions.md defining CSV extraction rules\n"
            "4. Write synthetic non-sensitive CSV fixture to workspace/input.csv\n"
            "5. Execute deterministic extractor on input.csv\n"
            "6. Validate JSON output contains expected columns, record count, and calculated totals\n"
            "7. Verify zero contamination of official skills/ directory"
        ),
        instructions=(
            "MODO: extractor de datos CSV\n"
            "POSTURA: determinista, estructurada y precisa\n"
            "REGLAS DE EXTRACCIÓN:\n"
            "- Leer archivo CSV desde workspace autorizado\n"
            "- Identificar encabezados y tipos de datos\n"
            "- Generar resumen de filas y agregaciones numéricas\n"
            "- Emitir salida únicamente en JSON válido"
        ),
        requested_capabilities=["documents.read"],
        recommended_model="code",
        prompt_family="SOFTWARE_PROMPT",
        uses_capabilities=True,
        test_cases=[
            SkillTestCase(
                name="aggregate_summary_test",
                description="Compute total revenue and list product catalog from input.csv",
                input_data={"file": "input.csv"},
                validation_criteria=["contains_3_records", "revenue_equals_655.0", "valid_json"],
            )
        ],
    )

    # 3. Create Temporary Skill & Sandbox
    skill = factory.create_temporary_skill(spec)
    print(f"[1] Temporary skill created at: {skill.base_dir}")

    # 4. Write test CSV inside isolated workspace
    csv_content = "id,product,units,price\n1,Alpha Widget,10,25.50\n2,Beta Gadget,5,40.00\n3,Gamma Tool,2,100.00\n"
    csv_path = factory.write_workspace_file(skill, "input.csv", csv_content)
    print(f"[2] Test CSV written to: {csv_path}")

    # 5. Define Real Test Runner inside sandbox
    def execute_extraction(sk, tc):
        file_name = tc.input_data.get("file", "input.csv")
        input_file = factory.resolve_sandboxed_path(sk, f"workspace/{file_name}")
        records = []
        total_revenue = 0.0
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                units = int(row["units"])
                price = float(row["price"])
                row_total = units * price
                total_revenue += row_total
                records.append({
                    "id": row["id"],
                    "product": row["product"],
                    "units": units,
                    "price": price,
                    "total": row_total,
                })
        output_payload = {
            "record_count": len(records),
            "total_revenue": total_revenue,
            "products": [r["product"] for r in records],
            "records": records,
        }
        is_valid = (len(records) == 3 and total_revenue == 655.0)
        return {
            "success": is_valid,
            "output": json.dumps(output_payload, indent=2),
            "criteria_met": is_valid,
        }

    # 6. Execute Test in Sandbox
    result = factory.test_temporary_skill(skill, test_runner=execute_extraction)
    print(f"[3] Test executed: passed={result.passed}, recommendation={result.recommendation}, duration={result.metrics.get('duration_total_ms')}ms")

    # 7. Generate Proposal .md in dev-notes/skill-proposals/
    proposal_path = factory.generate_proposal(skill, result)
    print(f"[4] Proposal generated at: {proposal_path}")

    # 8. Verify Zero Contamination
    loader = SkillLoader()
    loader.load_skills()
    assert "csv_data_extractor" not in loader.skills
    assert not (Path("skills") / "csv_data_extractor").exists()
    print("[5] Zero contamination verified: skills/ untouched and loader isolated.")

    # 9. Verify Proposal Headers & Recipe
    with open(proposal_path, "r", encoding="utf-8") as f:
        text = f.read()
    assert "Status: EXPERIMENTAL" in text
    assert "Human Review Required: YES" in text
    assert "APPROVE" in text
    assert "1. Instantiate TemporarySkill sandbox" in text
    print("[6] Proposal headers and recipe verified successfully.")
    print("=== CONTROLLED REAL END-TO-END TEST PASSED ===")
    return proposal_path



if __name__ == "__main__":
    main()
