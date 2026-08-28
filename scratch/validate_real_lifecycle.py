"""
Real-world validation script for AS-Core Skill Factory Lifecycle (CREATE -> TEST -> EDIT -> RETEST -> HISTORY -> DELETE).
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from api.main import app
from runtime.skills.loader import SkillLoader


def run_validation():
    print("================================================================")
    print("AS-CORE SKILL FACTORY: REAL LIFECYCLE VALIDATION")
    print("================================================================")

    client = TestClient(app)
    skill_id = "csv_data_extractor"

    # Step 1: Create csv_data_extractor (v1)
    print("\n[STEP 1] Creating experimental skill 'csv_data_extractor' (v1)...")
    create_payload = {
        "name": "CSV Data Extractor",
        "skill_id": skill_id,
        "description": "Extracts tabular data from CSV files and outputs validated JSON",
        "objective": "Process raw CSV files and output clean schema-validated records",
        "recipe": "1. Instantiate sandbox\n2. Run deterministic validation\n3. Verify zero official skills contamination",
        "instructions": "MODO: extractor de datos v1\nPOSTURA: analitica y estructurada\nREGLAS: Extraer filas y calcular totales.",
        "requested_capabilities": ["documents.read"],
        "recommended_model": "code",
        "uses_capabilities": True,
    }
    res = client.post("/v1/skills/experimental", json=create_payload)
    assert res.status_code == 200, f"Failed create: {res.text}"
    v1_info = res.json()
    print(f"  [OK] Created: {v1_info['skill_id']} (v{v1_info['version']}) status={v1_info['status']}")

    # Step 2: Run Test on v1
    print("\n[STEP 2] Running initial test on v1...")
    res = client.post(f"/v1/skills/experimental/{skill_id}/test")
    assert res.status_code == 200, f"Failed test v1: {res.text}"
    test1_data = res.json()
    print(f"  [OK] v1 Test: passed={test1_data['passed']}, rec={test1_data['recommendation']}, version={test1_data['version']}")

    # Step 3: Edit & Retest -> v2
    print("\n[STEP 3] Editing prompt to v2 and running retest...")
    edit2_payload = {
        "instructions": "MODO: extractor de datos v2\nPOSTURA: analitica de alta precision\nREGLAS: Validar encabezados y calcular total_revenue con 2 decimales.",
        "changes_description": "Enhanced precision requirements and decimal rounding rules in prompt",
    }
    res = client.put(f"/v1/skills/experimental/{skill_id}", json=edit2_payload)
    assert res.status_code == 200, f"Failed edit v2: {res.text}"
    v2_info = res.json()
    print(f"  [OK] Updated to version: {v2_info['version']}")

    res = client.post(f"/v1/skills/experimental/{skill_id}/test")
    assert res.status_code == 200, f"Failed test v2: {res.text}"
    test2_data = res.json()
    print(f"  [OK] v2 Test: passed={test2_data['passed']}, rec={test2_data['recommendation']}, version={test2_data['version']}")

    # Step 4: Edit & Retest -> v3
    print("\n[STEP 4] Editing prompt to v3 and running retest...")
    edit3_payload = {
        "instructions": "MODO: extractor de datos v3 (PROD_CANDIDATE)\nPOSTURA: analitica de alta precision\nREGLAS: Validar estructura, calcular revenue, omitir filas corruptas y asegurar salida JSON determinista.",
        "changes_description": "Added corrupt row handling and deterministic JSON schema formatting for final candidate",
    }
    res = client.put(f"/v1/skills/experimental/{skill_id}", json=edit3_payload)
    assert res.status_code == 200, f"Failed edit v3: {res.text}"
    v3_info = res.json()
    print(f"  [OK] Updated to version: {v3_info['version']}")

    res = client.post(f"/v1/skills/experimental/{skill_id}/test")
    assert res.status_code == 200, f"Failed test v3: {res.text}"
    test3_data = res.json()
    print(f"  [OK] v3 Test: passed={test3_data['passed']}, rec={test3_data['recommendation']}, version={test3_data['version']}")

    # Step 5: Verify Unified Proposal Content and Full Version History
    print("\n[STEP 5] Verifying unified proposal and Version History in dev-notes/skill-proposals/...")
    res = client.get(f"/v1/skills/experimental/{skill_id}/proposal")
    assert res.status_code == 200, f"Failed proposal get: {res.text}"
    proposal_data = res.json()
    content = proposal_data["content"]

    assert "## 16. Version History" in content
    assert "### Version 1" in content
    assert "### Version 2" in content
    assert "### Version 3" in content
    assert "Enhanced precision requirements" in content
    assert "Added corrupt row handling" in content
    assert "Status: EXPERIMENTAL" in content
    assert "Human Review Required: YES" in content
    print("  [OK] Unified proposal verified: contains Version 1, Version 2, and Version 3 with audit logs")

    # Step 6: DELETE experimental sandbox
    print("\n[STEP 6] Deleting experimental sandbox under temp_skills/...")
    res = client.delete(f"/v1/skills/experimental/{skill_id}")
    assert res.status_code == 200, f"Failed delete: {res.text}"
    del_data = res.json()
    print(f"  [OK] DELETE response: success={del_data['success']}, proposal_preserved={del_data['proposal_preserved']}")

    # Verification: Sandbox deleted, Proposal preserved, Official skills untouched
    sandbox_dir = Path("temp_skills") / skill_id
    proposal_file = Path("dev-notes/skill-proposals") / f"{skill_id}.md"
    official_dir = Path("skills")

    assert not sandbox_dir.exists(), f"Sandbox directory still exists: {sandbox_dir}"
    assert proposal_file.exists(), f"Proposal was unexpectedly deleted: {proposal_file}"
    assert not (official_dir / skill_id).exists(), "Official skills directory was contaminated!"

    # Verify official SkillLoader
    loader = SkillLoader()
    loader.load_skills()
    assert skill_id not in loader.skills, "SkillLoader discovered experimental skill!"

    print("\n================================================================")
    print("[ALL CHECKS PASSED] ALL REAL-WORLD AUDIT REQUIREMENTS VERIFIED!")
    print("  - Sandbox deleted: temp_skills/csv_data_extractor/ [REMOVED]")
    print(f"  - Audit proposal preserved: {proposal_file} [PRESERVED]")
    print("  - Official skills untouched: skills/ [100% ISOLATED]")
    print("  - Official SkillLoader: [CLEAN - 0 CONTAMINATION]")
    print("================================================================")


if __name__ == "__main__":
    run_validation()
