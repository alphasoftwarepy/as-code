"""
Temporary Skill Models & Lifecycle Definition for AS-Core Experimental Skill Factory.

Zero Contamination: These classes represent ephemeral/sandbox skills that reside
strictly under temp_skills/<skill_id>/ and never in official skills/.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from runtime.skills.models import SkillManifest

logger = logging.getLogger("as-code.runtime.skills.temporary")


class TemporarySkillLifecycle(str, Enum):
    CREATED = "CREATED"
    READY = "READY"
    TESTING = "TESTING"
    PASSED = "PASSED"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"


class SkillTestCase(BaseModel):
    """Individual test case for validating temporary skill performance."""
    name: str
    description: str
    input_data: Dict[str, Any] = Field(default_factory=dict)
    expected_output_pattern: Optional[str] = None
    validation_criteria: List[str] = Field(default_factory=list)


class SkillVersionRecord(BaseModel):
    """Immutable audit record for each experimental iteration/version."""
    version: int
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    changes_description: str = "Initial experimental skill"
    instructions_snippet: str = ""
    test_passed: Optional[bool] = None
    recommendation: Optional[str] = "NEEDS_REFINEMENT"
    metrics: Dict[str, Any] = Field(default_factory=dict)
    problems: List[str] = Field(default_factory=list)


class SkillSpec(BaseModel):
    """Specification used by SkillFactory to instantiate or update a temporary skill."""
    skill_id: str = Field(..., description="Unique slug/identifier for the experimental skill")
    name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Short functional description")
    objective: str = Field(..., description="Specific problem or objective this skill solves")
    recipe: str = Field(..., description="Step-by-step reproducible recipe used to build and configure the skill")
    instructions: str = Field(..., description="System instructions / prompt for the model")
    requested_capabilities: List[str] = Field(default_factory=list, description="Capability IDs or scopes needed")
    recommended_model: Optional[str] = Field(default="code", description="Recommended model ID (e.g. code, reasoning)")
    prompt_family: Optional[str] = Field(default="SOFTWARE_PROMPT", description="Prompt family e.g. SOFTWARE_PROMPT, BUSINESS_PROMPT")
    uses_capabilities: bool = Field(default=True, description="Whether this skill requires the capability gate opened")
    test_cases: List[SkillTestCase] = Field(default_factory=list, description="Test cases to validate the skill")
    version: int = Field(default=1, description="Experimental version number")
    history: List[SkillVersionRecord] = Field(default_factory=list, description="Historical audit trail of versions")


class TemporarySkillManifest(SkillManifest):
    """Extended manifest for temporary skills containing sandbox metadata."""
    objective: Optional[str] = None
    recommended_model: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    is_temporary: bool = True
    version: int = 1
    history: List[SkillVersionRecord] = Field(default_factory=list)


class SkillTestResult(BaseModel):
    """Results from testing a temporary skill."""
    skill_id: str
    lifecycle: TemporarySkillLifecycle
    passed: bool
    version: int = 1
    test_cases_total: int
    test_cases_passed: int
    test_cases_failed: int
    execution_flow: List[Dict[str, Any]] = Field(default_factory=list)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)  # e.g. duration_ms, tokens, etc.
    problems_encountered: List[str] = Field(default_factory=list)
    security_checks: Dict[str, bool] = Field(default_factory=dict)
    isolation_checks: Dict[str, bool] = Field(default_factory=dict)
    recommendation: str = "NEEDS_REFINEMENT"  # APPROVE, REJECT, NEEDS_REFINEMENT
    notes: Optional[str] = None


class TemporarySkill:
    """Represents an active, isolated temporary skill in the sandbox workspace."""

    def __init__(
        self,
        spec: SkillSpec,
        base_dir: Path,
        lifecycle: TemporarySkillLifecycle = TemporarySkillLifecycle.CREATED,
        created_at: Optional[datetime] = None
    ):
        self.spec = spec
        self.skill_id = spec.skill_id
        self.base_dir = base_dir.resolve()
        self.lifecycle = lifecycle
        self.created_at = created_at or datetime.now(timezone.utc)
        self.version = spec.version
        self.history: List[SkillVersionRecord] = list(spec.history)

        # Isolated directory tree
        self.workspace_dir = self.base_dir / "workspace"
        self.tests_dir = self.base_dir / "tests"
        self.results_dir = self.base_dir / "results"
        self.logs_dir = self.base_dir / "logs"

        self.manifest_path = self.base_dir / "manifest.json"
        self.instructions_path = self.base_dir / "instructions.md"

    @property
    def manifest(self) -> TemporarySkillManifest:
        return TemporarySkillManifest(
            id=self.spec.skill_id,
            name=self.spec.name,
            description=self.spec.description,
            required_scopes=self.spec.requested_capabilities,
            enabled=True,
            prompt_family=self.spec.prompt_family,
            uses_capabilities=self.spec.uses_capabilities,
            objective=self.spec.objective,
            recommended_model=self.spec.recommended_model,
            created_at=self.created_at.isoformat(),
            is_temporary=True,
            version=self.version,
            history=self.history,
        )

    def record_version(
        self,
        changes_description: Optional[str] = None,
        test_result: Optional[SkillTestResult] = None
    ) -> SkillVersionRecord:
        """Add or update version history entry."""
        existing_idx = next((i for i, r in enumerate(self.history) if r.version == self.version), None)
        existing_record = self.history[existing_idx] if existing_idx is not None else None

        final_changes = changes_description
        if not final_changes or final_changes.startswith("Validation run for version"):
            if existing_record and existing_record.changes_description and not existing_record.changes_description.startswith("Validation run for version"):
                final_changes = existing_record.changes_description
            elif not final_changes:
                final_changes = f"Experimental version {self.version}"

        snippet = (self.spec.instructions[:150] + "...") if len(self.spec.instructions) > 150 else self.spec.instructions
        record = SkillVersionRecord(
            version=self.version,
            created_at=datetime.now(timezone.utc).isoformat(),
            changes_description=final_changes,
            instructions_snippet=snippet,
            test_passed=test_result.passed if test_result else (existing_record.test_passed if existing_record else None),
            recommendation=test_result.recommendation if test_result else (existing_record.recommendation if existing_record else "NEEDS_REFINEMENT"),
            metrics=test_result.metrics if test_result else (existing_record.metrics if existing_record else {}),
            problems=test_result.problems_encountered if test_result else (existing_record.problems if existing_record else []),
        )

        if existing_idx is not None:
            self.history[existing_idx] = record
        else:
            self.history.append(record)

        self.spec.history = self.history
        return record


    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.spec.name,
            "description": self.spec.description,
            "objective": self.spec.objective,
            "recipe": self.spec.recipe,
            "instructions": self.spec.instructions,
            "requested_capabilities": self.spec.requested_capabilities,
            "recommended_model": self.spec.recommended_model,
            "version": self.version,
            "history": [h.model_dump() for h in self.history],
            "workspace_path": str(self.workspace_dir),
            "created_at": self.created_at.isoformat(),
            "lifecycle": self.lifecycle.value,
        }
