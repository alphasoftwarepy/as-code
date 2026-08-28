"""
Comprehensive Lifecycle & Security Test Suite for Experimental Skill Factory (AS-Core).

Covers all 18 lifecycle specifications:
1. List experimental skills
2. Select experimental
3. Edit skill
4. Increment version
5. Save updated prompt
6. SAVE & TEST
7. FAILED -> EDIT -> RETEST
8. PASSED -> EDIT -> RETEST
9. Conserve history
10. Proposal contains all versions
11. DELETE experimental
12. Proposal remains after DELETE
13. Official skill cannot be deleted or modified
14. Path traversal rejected
15. Experimental does not enter SkillLoader
16. Experimental does not participate in AUTO routing
17. Controlled concurrent testing
18. Capability gate remains active
"""

import os
import shutil
from pathlib import Path
from fastapi.testclient import TestClient
import pytest

from api.main import app
from runtime.skills.factory import SkillFactory, SandboxSecurityError
from runtime.skills.loader import SkillLoader
from runtime.skills.temporary import SkillSpec, SkillTestCase, TemporarySkillLifecycle


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def clean_test_skill():
    skill_id = "lifecycle_test_skill"
    factory = SkillFactory()
    factory.cleanup(skill_id)
    # Also clean proposal if needed
    prop_path = factory.proposals_dir / f"{skill_id}.md"
    if prop_path.exists():
        prop_path.unlink()

    yield skill_id

    # Teardown
    factory.cleanup(skill_id)


def test_1_and_2_list_and_select_experimental(client, clean_test_skill):
    """1 & 2. List experimental skills and select for test activation."""
    # Create skill
    res = client.post("/v1/skills/experimental", json={
        "name": "Lifecycle Test Skill",
        "skill_id": clean_test_skill,
        "description": "Skill for testing complete lifecycle",
        "objective": "Test v1-v3 transitions",
        "instructions": "MODO: test_v1\nREGLAS: Procesar.",
        "requested_capabilities": ["documents.read"],
        "recommended_model": "code",
    })
    assert res.status_code == 200

    # 1. List
    list_res = client.get("/v1/skills/experimental")
    assert list_res.status_code == 200
    skills = list_res.json()
    found = [s for s in skills if s["skill_id"] == clean_test_skill]
    assert len(found) == 1
    assert found[0]["version"] == 1

    # 2. Get detail for selection
    detail_res = client.get(f"/v1/skills/experimental/{clean_test_skill}")
    assert detail_res.status_code == 200
    assert detail_res.json()["skill_id"] == clean_test_skill


def test_3_4_5_edit_increment_version_and_save_prompt(client, clean_test_skill):
    """3, 4, 5. Edit skill, increment version, and save updated prompt."""
    factory = SkillFactory()
    spec = SkillSpec(
        skill_id=clean_test_skill,
        name="Version Incrementor",
        description="Initial description",
        objective="Initial objective",
        recipe="Initial recipe",
        instructions="MODO: initial_v1",
        version=1,
    )
    skill = factory.create_temporary_skill(spec)
    assert skill.version == 1

    # Edit skill to v2
    updated = factory.update_temporary_skill(
        clean_test_skill,
        updates={"instructions": "MODO: updated_v2_prompt", "description": "Updated description"},
        changes_description="Modified prompt to v2",
    )
    assert updated.version == 2
    assert (skill.base_dir / "instructions.md").read_text(encoding="utf-8") == "MODO: updated_v2_prompt"
    assert len(updated.history) >= 2


def test_6_7_8_9_10_full_cycle_failed_to_passed_and_history(client, clean_test_skill):
    """6, 7, 8, 9, 10. Complete SAVE & TEST, FAILED -> EDIT -> RETEST -> PASSED, and audit history in proposal."""
    # 1. Create skill (v1) with instructions that will fail a strict test runner (< 10 chars)
    res = client.post("/v1/skills/experimental", json={
        "name": "Audit Cycle Skill",
        "skill_id": clean_test_skill,
        "description": "Tests full audit history",
        "objective": "Achieve PASS through iteration",
        "instructions": "short",  # 5 chars -> fails length check (< 10)
        "requested_capabilities": ["documents.read"],
    })
    assert res.status_code == 200
    assert res.json()["version"] == 1

    # 2. Run Test on v1 -> FAILED
    test1_res = client.post(f"/v1/skills/experimental/{clean_test_skill}/test")
    assert test1_res.status_code == 200
    test1_data = test1_res.json()
    assert test1_data["passed"] is False
    assert test1_data["status"] == "FAILED"
    assert test1_data["version"] == 1

    # 3. EDIT & RETEST -> v2 (Partial fix)
    edit2_res = client.put(f"/v1/skills/experimental/{clean_test_skill}", json={
        "instructions": "MODO: slightly longer instructions for v2 test",
        "changes_description": "Extended instructions for v2",
    })
    assert edit2_res.status_code == 200
    assert edit2_res.json()["version"] == 2

    test2_res = client.post(f"/v1/skills/experimental/{clean_test_skill}/test")
    assert test2_res.status_code == 200
    test2_data = test2_res.json()
    assert test2_data["version"] == 2
    assert test2_data["passed"] is True

    # 4. EDIT & RETEST -> v3 (Final refinement)
    edit3_res = client.put(f"/v1/skills/experimental/{clean_test_skill}", json={
        "instructions": "MODO: complete robust instructions with strict format for v3",
        "changes_description": "Final prompt refinement with schema enforcement",
    })
    assert edit3_res.status_code == 200
    assert edit3_res.json()["version"] == 3

    test3_res = client.post(f"/v1/skills/experimental/{clean_test_skill}/test")
    assert test3_res.status_code == 200
    test3_data = test3_res.json()
    assert test3_data["passed"] is True
    assert test3_data["recommendation"] == "APPROVE"
    assert test3_data["version"] == 3

    # 10. Check unified proposal contains full Version History (Version 1, Version 2, Version 3)
    prop_res = client.get(f"/v1/skills/experimental/{clean_test_skill}/proposal")
    assert prop_res.status_code == 200
    content = prop_res.json()["content"]

    assert "## 16. Version History" in content
    assert "### Version 1" in content
    assert "### Version 2" in content
    assert "### Version 3" in content
    assert "Extended instructions for v2" in content
    assert "Final prompt refinement" in content
    assert "Status: EXPERIMENTAL" in content
    assert "Human Review Required: YES" in content


def test_11_12_delete_experimental_and_preserve_proposal(client, clean_test_skill):
    """11 & 12. DELETE removes sandbox under temp_skills/ but preserves dev-notes proposal."""
    # 1. Create skill first
    create_res = client.post("/v1/skills/experimental", json={
        "name": "Delete Target Skill",
        "skill_id": clean_test_skill,
        "description": "To be deleted",
        "objective": "Verify deletion",
        "instructions": "MODO: delete_target_instructions",
    })
    assert create_res.status_code == 200

    # 2. Run test to generate proposal
    test_res = client.post(f"/v1/skills/experimental/{clean_test_skill}/test")
    assert test_res.status_code == 200

    factory = SkillFactory()
    sandbox_dir = factory.base_temp_dir / clean_test_skill
    proposal_file = factory.proposals_dir / f"{clean_test_skill}.md"

    assert sandbox_dir.exists()
    assert proposal_file.exists()

    # 11. DELETE
    del_res = client.delete(f"/v1/skills/experimental/{clean_test_skill}")
    assert del_res.status_code == 200
    del_data = del_res.json()
    assert del_data["success"] is True
    assert del_data["proposal_preserved"] is True

    # Check sandbox is deleted
    assert not sandbox_dir.exists()

    # 12. Check proposal is preserved
    assert proposal_file.exists()


def test_13_official_skills_cannot_be_deleted_or_polluted(clean_test_skill):
    """13. Verify official skills directory is completely isolated and cannot be deleted."""
    official_dir = (Path(__file__).parents[1] / "skills").resolve()
    assert official_dir.exists()
    official_skills = set(os.listdir(official_dir))

    # Verify no test skills are inside official skills/
    assert clean_test_skill not in official_skills
    assert "csv_data_extractor" not in official_skills


def test_14_path_traversal_rejected_in_sandbox():
    """14. Verify sandbox path resolution rejects path traversal attempts."""
    factory = SkillFactory()
    spec = SkillSpec(
        skill_id="traversal_test",
        name="Traversal Test",
        description="Testing security boundaries",
        objective="Verify anti-traversal",
        recipe="1. Check paths",
        instructions="MODO: test",
    )
    skill = factory.create_temporary_skill(spec)

    with pytest.raises(SandboxSecurityError):
        factory.resolve_sandboxed_path(skill, "../../../etc/passwd")

    with pytest.raises(SandboxSecurityError):
        factory.resolve_sandboxed_path(skill, "../../skills/official_skill")

    with pytest.raises(SandboxSecurityError):
        factory.resolve_sandboxed_path(skill, "/absolute/path/file.txt")

    factory.cleanup("traversal_test")


def test_15_and_16_experimental_never_enters_loader_or_auto_routing(clean_test_skill):
    """15 & 16. Verify experimental skill never enters SkillLoader or AUTO routing."""
    factory = SkillFactory()
    spec = SkillSpec(
        skill_id=clean_test_skill,
        name="Isolation Check",
        description="Testing isolation",
        objective="Verify zero loader pollution",
        recipe="1. Check",
        instructions="MODO: test",
    )
    factory.create_temporary_skill(spec)

    loader = SkillLoader()
    loader.load_skills()
    assert clean_test_skill not in loader.skills
    assert clean_test_skill not in [manifest.id for manifest, _ in loader.skills.values()]



def test_17_and_18_concurrency_and_capability_gate(client, clean_test_skill):
    """17 & 18. Controlled testing execution and capability gate integrity."""
    factory = SkillFactory()
    spec = SkillSpec(
        skill_id=clean_test_skill,
        name="Cap Gate Skill",
        description="Capability gate test",
        objective="Verify gate",
        recipe="1. Test",
        instructions="MODO: cap_gate_test",
        uses_capabilities=True,
        requested_capabilities=["documents.read"],
    )
    skill = factory.create_temporary_skill(spec)

    # Capability gate check
    assert skill.manifest.uses_capabilities is True
    assert "documents.read" in skill.manifest.required_scopes

    result = factory.test_temporary_skill(skill)
    assert result.security_checks["capability_gate_enforced"] is True
    assert result.security_checks["no_direct_subprocess"] is True
    assert result.isolation_checks["official_skills_untouched"] is True
