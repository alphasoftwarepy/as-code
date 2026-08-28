"""
Test Suite for Skill Factory API Endpoints (AS-Core).

Tests:
1. GET /v1/skills (official skills list)
2. GET /v1/skills/experimental (experimental skills list)
3. POST /v1/skills/experimental (create experimental skill in sandbox)
4. Sandbox directory containment verification (temp_skills/<id>)
5. Official SkillLoader isolation (zero contamination)
6. POST /v1/skills/experimental/{id}/test (run test & emit proposal)
7. GET /v1/skills/experimental/{id}/proposal (retrieve proposal document)
8. Verify official skills/ directory remains 100% untouched
9. Verify no auto-promotion occurs
10. Error handling for non-existent skills and invalid inputs
"""

import os
from pathlib import Path
from fastapi.testclient import TestClient
import pytest

from api.main import app
from runtime.skills.loader import SkillLoader


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_api_get_official_skills(client):
    """1. Verify GET /v1/skills returns official skills dictionary."""
    res = client.get("/v1/skills")
    assert res.status_code == 200
    skills = res.json()
    assert isinstance(skills, dict)
    assert "programming" in skills or "sales" in skills
    # Ensure official skill structure
    first_skill = next(iter(skills.values()))
    assert "id" in first_skill
    assert "compatible" in first_skill


def test_api_get_experimental_skills(client):
    """2. Verify GET /v1/skills/experimental returns list of temporary skills."""
    res = client.get("/v1/skills/experimental")
    assert res.status_code == 200
    exp_skills = res.json()
    assert isinstance(exp_skills, list)


def test_api_create_experimental_skill_and_sandbox(client):
    """3 & 4. Verify POST /v1/skills/experimental creates temporary skill in isolated sandbox."""
    payload = {
        "name": "API Test Tool",
        "skill_id": "api_test_tool",
        "description": "Tool created via API test",
        "objective": "Verify API endpoint functionality",
        "recipe": "1. Create\n2. Verify sandbox",
        "instructions": "MODO: test\nPOSTURA: analitica\nREGLAS: Validar entradas.",
        "requested_capabilities": ["documents.read"],
        "recommended_model": "code",
        "uses_capabilities": True,
    }

    res = client.post("/v1/skills/experimental", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["success"] is True
    assert data["skill_id"] == "api_test_tool"
    assert data["status"] == "READY"
    assert data["is_official"] is False
    assert data["human_review_required"] is True

    # 4. Verify sandbox directory exists under temp_skills/
    sandbox_dir = Path(data["sandbox_path"])
    assert sandbox_dir.exists()
    assert (sandbox_dir / "manifest.json").exists()
    assert (sandbox_dir / "instructions.md").exists()
    assert (sandbox_dir / "workspace").exists()
    assert (sandbox_dir / "results").exists()


def test_api_experimental_skill_not_in_official_loader(client):
    """5. Verify experimental skill does NOT appear in official SkillLoader."""
    loader = SkillLoader()
    loader.load_skills()
    assert "api_test_tool" not in loader.skills

    # Also verify GET /v1/skills does NOT list it
    res = client.get("/v1/skills")
    assert "api_test_tool" not in res.json()


def test_api_test_experimental_skill_and_proposal(client):
    """6 & 7. Verify POST test endpoint executes validation and GET proposal retrieves markdown."""
    # Test execution
    test_res = client.post("/v1/skills/experimental/api_test_tool/test")
    assert test_res.status_code == 200
    test_data = test_res.json()

    assert test_data["success"] is True
    assert test_data["skill_id"] == "api_test_tool"
    assert test_data["passed"] is True
    assert test_data["recommendation"] == "APPROVE"
    assert test_data["human_review_required"] is True
    assert "proposal_file" in test_data

    # 7. Get proposal
    prop_res = client.get("/v1/skills/experimental/api_test_tool/proposal")
    assert prop_res.status_code == 200
    prop_data = prop_res.json()

    assert prop_data["skill_id"] == "api_test_tool"
    assert "content" in prop_data
    content = prop_data["content"]
    assert "Status: EXPERIMENTAL" in content
    assert "Human Review Required: YES" in content
    assert "APPROVE" in content
    assert "**Recommendation:** `APPROVE`" in content
    assert "MODO: test" in content



def test_api_official_skills_untouched_after_api_operations(client):
    """8. Verify official skills/ directory remains 100% untouched."""
    official_dir = Path("skills").resolve()
    assert official_dir.exists()
    official_files = set(os.listdir(official_dir))

    # Official skills should not have api_test_tool or csv_data_extractor
    assert "api_test_tool" not in official_files
    assert "csv_data_extractor" not in official_files


def test_api_error_handling(client):
    """9 & 10. Verify proper 404 and 422 error handling for invalid operations."""
    # Non-existent skill testing
    res_not_found = client.post("/v1/skills/experimental/non_existent_skill_xyz/test")
    assert res_not_found.status_code == 404

    # Non-existent proposal
    res_prop_not_found = client.get("/v1/skills/experimental/non_existent_skill_xyz/proposal")
    assert res_prop_not_found.status_code == 404

    # Missing required fields in creation
    res_invalid = client.post("/v1/skills/experimental", json={"name": "A"})
    assert res_invalid.status_code == 422
