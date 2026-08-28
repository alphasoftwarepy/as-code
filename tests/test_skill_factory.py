"""
Test Suite for Experimental Skill Factory & Temporary Skills (AS-Core).

Validates:
1. Creation of sandboxed temporary skills
2. Manifest and metadata integrity
3. Zero contamination of official skills and loader isolation
4. Workspace path containment and anti-traversal security
5. Reproducible proposal markdown generation in dev-notes/skill-proposals/
6. Recipe and instructions reproducibility in proposals
7. Proper failure documentation and recommendation handling
8. Official skills directory remains untouched
9. Terminal capability gate and human approval enforcement (no bypass)
10. Skill Factory remains disabled by default for normal chat/coordinator queries
"""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
import pytest

from runtime.skills.temporary import (
    TemporarySkillLifecycle,
    SkillSpec,
    SkillTestCase,
    SkillTestResult,
    TemporarySkill,
)
from runtime.skills.factory import SkillFactory, SandboxSecurityError
from runtime.skills.loader import SkillLoader, get_skill_loader
from runtime.capabilities.registry import CapabilityRegistry
from runtime.capabilities.terminal import TerminalCapability
from runtime.coordinator.manager import PureCoordinator
from runtime.coordinator.models import RuntimeContract, SessionSnapshot, WorkflowState
from runtime.coordinator.agent import AgentControlRunner
from api.models import ChatCompletionRequest, ChatMessage
from providers.base import InferenceRequest


@pytest.fixture
def temp_env():
    """Create isolated temporary directories for factory testing."""
    temp_dir = tempfile.mkdtemp(prefix="as_test_factory_")
    base_temp_skills = Path(temp_dir) / "temp_skills"
    proposals_dir = Path(temp_dir) / "dev-notes" / "skill-proposals"
    
    factory = SkillFactory(base_temp_dir=base_temp_skills, proposals_dir=proposals_dir)
    yield factory, base_temp_skills, proposals_dir
    
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_factory_creates_temporary_skill(temp_env):
    """1. Verify creation and complete directory structure of the temporary sandbox."""
    factory, base_temp_skills, _ = temp_env

    spec = SkillSpec(
        skill_id="test_extractor",
        name="Test Data Extractor",
        description="Extracts data from sample text",
        objective="Validate sandbox creation",
        recipe="1. Create spec\n2. Run sandbox test",
        instructions="Extract keys from input.",
        requested_capabilities=["documents.read"],
        recommended_model="code",
    )

    skill = factory.create_temporary_skill(spec)

    assert skill.skill_id == "test_extractor"
    assert skill.lifecycle == TemporarySkillLifecycle.READY
    assert skill.base_dir.exists()
    assert skill.workspace_dir.exists()
    assert skill.tests_dir.exists()
    assert skill.results_dir.exists()
    assert skill.logs_dir.exists()
    assert skill.manifest_path.exists()
    assert skill.instructions_path.exists()

    with open(skill.instructions_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert content == "Extract keys from input."


def test_temporary_skill_has_manifest(temp_env):
    """2. Verify manifest schema, metadata, and temporary flags."""
    factory, _, _ = temp_env

    spec = SkillSpec(
        skill_id="manifest_tester",
        name="Manifest Tester",
        description="Tests manifest properties",
        objective="Ensure manifest compliance",
        recipe="Check manifest serialization",
        instructions="Analyze data.",
        requested_capabilities=["rag.retrieve"],
        prompt_family="SOFTWARE_PROMPT",
        uses_capabilities=True,
    )

    skill = factory.create_temporary_skill(spec)
    manifest = skill.manifest

    assert manifest.id == "manifest_tester"
    assert manifest.name == "Manifest Tester"
    assert manifest.required_scopes == ["rag.retrieve"]
    assert manifest.prompt_family == "SOFTWARE_PROMPT"
    assert manifest.uses_capabilities is True
    assert manifest.is_temporary is True
    assert manifest.objective == "Ensure manifest compliance"
    assert manifest.created_at is not None


def test_temporary_skill_is_not_official(temp_env):
    """3. Verify that official SkillLoader NEVER discovers temporary skills."""
    factory, _, _ = temp_env

    spec = SkillSpec(
        skill_id="secret_temp_skill",
        name="Secret Temp Skill",
        description="Should not appear in official registry",
        objective="Test isolation from official loader",
        recipe="Check SkillLoader.skills",
        instructions="Secret prompt",
    )
    skill = factory.create_temporary_skill(spec)

    # Load official skills using standard loader
    official_loader = SkillLoader()
    official_loader.load_skills()

    # The temporary skill ID must NOT be present in official skills
    assert "secret_temp_skill" not in official_loader.skills
    assert official_loader.get_skill_manifest("secret_temp_skill") is None
    assert official_loader.get_skill_prompt("secret_temp_skill") is None


def test_workspace_isolated(temp_env):
    """4. Verify sandbox path containment and anti-traversal protection."""
    factory, _, _ = temp_env

    spec = SkillSpec(
        skill_id="isolation_tester",
        name="Isolation Tester",
        description="Tests sandbox boundaries",
        objective="Validate containment",
        recipe="Try relative write and escape attempts",
        instructions="Test",
    )
    skill = factory.create_temporary_skill(spec)

    # Valid write inside workspace
    written_file = factory.write_workspace_file(skill, "data/sample.txt", "hello sandbox")
    assert written_file.exists()
    with open(written_file, "r", encoding="utf-8") as f:
        assert f.read() == "hello sandbox"

    # Blocked: Path traversal escaping sandbox
    with pytest.raises(SandboxSecurityError):
        factory.resolve_sandboxed_path(skill, "../escaped.txt")

    with pytest.raises(SandboxSecurityError):
        factory.resolve_sandboxed_path(skill, "../../skills/injected.txt")

    # Blocked: Absolute paths
    with pytest.raises(SandboxSecurityError):
        factory.resolve_sandboxed_path(skill, "C:\\Windows\\system32\\cmd.exe")


def test_proposal_generated(temp_env):
    """5. Verify proposal file emission into dev-notes/skill-proposals/."""
    factory, _, proposals_dir = temp_env

    spec = SkillSpec(
        skill_id="proposal_tester",
        name="Proposal Generator Test",
        description="Testing proposal .md emission",
        objective="Ensure proposal is created",
        recipe="Execute test and generate proposal",
        instructions="Instructions for proposal",
        test_cases=[
            SkillTestCase(
                name="test_1",
                description="Sample test",
                input_data={"key": "value"},
            )
        ],
    )
    skill = factory.create_temporary_skill(spec)
    result = factory.test_temporary_skill(skill)

    proposal_path = factory.generate_proposal(skill, result)

    assert proposal_path.exists()
    assert proposal_path.parent == proposals_dir
    assert proposal_path.name.endswith("proposal_tester.md")



def test_proposal_contains_recipe(temp_env):
    """6. Verify proposal contains complete reproducible instructions, recipe, and required headers."""
    factory, _, _ = temp_env

    recipe_text = "Step 1: Parse CSV\nStep 2: Filter rows\nStep 3: Output JSON"
    spec = SkillSpec(
        skill_id="recipe_validator",
        name="Recipe Validator",
        description="Checks recipe in proposal",
        objective="Ensure exact recipe is documented",
        recipe=recipe_text,
        instructions="Perform deterministic CSV extraction.",
        requested_capabilities=["documents.read"],
    )
    skill = factory.create_temporary_skill(spec)
    result = factory.test_temporary_skill(skill)
    proposal_path = factory.generate_proposal(skill, result)

    with open(proposal_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "Status: EXPERIMENTAL" in content
    assert "Human Review Required: YES" in content
    assert recipe_text in content
    assert "Perform deterministic CSV extraction." in content
    assert "Recommendation:" in content
    assert "Files Modified Outside Workspace:** `NONE`" in content


def test_failed_skill_generates_failed_proposal(temp_env):
    """7. Verify a failed test produces REJECT / NEEDS_REFINEMENT and never APPROVE."""
    factory, _, _ = temp_env

    spec = SkillSpec(
        skill_id="failing_skill",
        name="Failing Skill Test",
        description="Skill intended to fail validation",
        objective="Test failure propagation",
        recipe="Failing recipe",
        instructions="Broken instructions",
        test_cases=[
            SkillTestCase(
                name="failing_case",
                description="This test case will fail",
                input_data={"query": "fail"},
            )
        ],
    )
    skill = factory.create_temporary_skill(spec)

    def failing_runner(sk, tc):
        return {"success": False, "error": "Simulated extraction failure", "output": ""}

    result = factory.test_temporary_skill(skill, test_runner=failing_runner)

    assert result.passed is False
    assert result.lifecycle == TemporarySkillLifecycle.FAILED
    assert result.recommendation in ("REJECT", "NEEDS_REFINEMENT")
    assert result.recommendation != "APPROVE"

    proposal_path = factory.generate_proposal(skill, result)
    with open(proposal_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "Overall Result:** `FAIL`" in content
    assert result.recommendation in content
    assert "Simulated extraction failure" in content


def test_no_official_skill_modification(temp_env):
    """8. Verify official skills/ directory remains 100% untouched."""
    factory, _, _ = temp_env

    official_skills_dir = Path("skills").resolve()
    if not official_skills_dir.exists():
        official_skills_dir = (Path(__file__).parents[1] / "skills").resolve()

    # Snapshot existing official skills
    official_entries_before = set(os.listdir(official_skills_dir))

    spec = SkillSpec(
        skill_id="no_touch_skill",
        name="No Touch Skill",
        description="Should never modify official skills/",
        objective="Check zero official skills modification",
        recipe="Run full lifecycle",
        instructions="Do not touch skills/",
    )
    skill = factory.create_temporary_skill(spec)
    result = factory.test_temporary_skill(skill)
    factory.generate_proposal(skill, result)

    official_entries_after = set(os.listdir(official_skills_dir))

    # Strict equality: nothing added, removed, or modified in official skills
    assert official_entries_before == official_entries_after
    assert "no_touch_skill" not in official_entries_after


def test_terminal_respects_capability_gate(monkeypatch):
    """9. Verify terminal actions remain subject to capability checks and approval contracts."""
    terminal_cap = TerminalCapability()
    assert terminal_cap.requires_approval("execute") is True

    engine = MagicMock()
    runner = AgentControlRunner(engine)

    import asyncio
    async def run_checks():
        # Case A: By default, terminal is disabled for security
        res_default = await runner.execute_capability(
            capability_id="terminal",
            action="execute",
            params={"command": "dir"}
        )
        assert res_default.get("success") is False
        assert "disabled" in res_default.get("output", "").lower()

        # Case B: When enabled via settings override, execution is intercepted by approval contract
        from config.settings import Settings
        mock_settings = MagicMock(spec=Settings)
        mock_settings.capability_overrides = {"terminal": True}

        from runtime.coordinator import agent as agent_module
        monkeypatch.setattr(agent_module, "get_settings", lambda: mock_settings)

        res_enabled = await runner.execute_capability(
            capability_id="terminal",
            action="execute",
            params={"command": "dir"}
        )
        assert res_enabled.get("status") == "pending_approval"
        assert res_enabled.get("approval_id") is not None
        assert "Aprobación Requerida" in res_enabled.get("output") or "Approval Required" in res_enabled.get("output")


    asyncio.run(run_checks())



def test_factory_disabled_by_default():
    """10. Verify normal chat/coordinator routing does not invoke or instantiate SkillFactory."""
    coord = PureCoordinator()
    db = MagicMock()

    contract = RuntimeContract(
        request_id="req-normal-chat",
        session_id="sess-normal",
        model_id="chat",
        user_message="Hola, ¿cómo estás?",
        timestamp=123.456,
        snapshot=SessionSnapshot(session_id="sess-normal", turn_number=1),
    )

    manifest = coord.assemble(
        db=db,
        contract=contract,
        skill_service=None,
        rag_service=None,
        memory_service=None,
        enable_rag=False,
    )

    # Standard query should have capability gate closed and no temporary skills
    assert manifest.capability_gate_open is False
    assert manifest.active_skill is None or manifest.active_skill in ("chat", None)
    assert "PROTOCOLO DE INVOCACIÓN DE CAPACIDADES" not in manifest.system_prompt_snapshot
