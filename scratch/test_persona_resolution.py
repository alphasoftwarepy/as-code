import unittest
from typing import Optional
from pydantic import BaseModel
from runtime.skills.models import SkillManifest
from runtime.coordinator.prompts import resolve_root_prompt, PROMPT_FAMILIES
from runtime.coordinator.models import RuntimeContract, WorkflowState, SessionSnapshot
from runtime.coordinator.manager import PureCoordinator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from api.rag_models import Base
from api.memory_models import MemoryBase

# Simple mock for SkillLoader
class MockSkillLoader:
    def __init__(self, skills_dict):
        self.skills = skills_dict

    def get_skill_manifest(self, skill_id: str) -> Optional[SkillManifest]:
        skill = self.skills.get(skill_id)
        if skill:
            return skill[0]
        return None

    def get_skill_prompt(self, skill_id: str) -> Optional[str]:
        skill = self.skills.get(skill_id)
        if skill:
            return skill[1]
        return None

class TestPersonaResolution(unittest.TestCase):
    def test_prompt_family_resolution_direct(self):
        # 1. Test General Analytical fallback in Spanish and English
        es_general = resolve_root_prompt("ES", None)
        self.assertIn("asistente de inteligencia artificial", es_general)
        self.assertIn("fuente primaria", es_general)

        en_general = resolve_root_prompt("EN", None)
        self.assertIn("artificial intelligence assistant", en_general)
        self.assertIn("primary source", en_general)

        # 2. Test Business Prompt
        es_business = resolve_root_prompt("ES", "BUSINESS_PROMPT")
        self.assertIn("experto en negocios", es_business)
        self.assertNotIn("DIAGNÓSTICO", es_business)

        # 3. Test Software Prompt
        es_software = resolve_root_prompt("ES", "SOFTWARE_PROMPT")
        self.assertIn("operador de software", es_software)

        # 4. Test Unrecognized Fallback
        fallback = resolve_root_prompt("ES", "UNKNOWN_PROMPT")
        self.assertEqual(fallback, es_general)

    def test_skill_manifest_default_prompt_family(self):
        # By default, a manifest should have prompt_family as None (absence of specialized skill)
        manifest = SkillManifest(
            id="test_skill",
            name="Test",
            description="Testing"
        )
        self.assertIsNone(manifest.prompt_family)

        # But it should be parseable if provided
        manifest_custom = SkillManifest(
            id="test_skill",
            name="Test",
            description="Testing",
            prompt_family="BUSINESS_PROMPT"
        )
        self.assertEqual(manifest_custom.prompt_family, "BUSINESS_PROMPT")

    def test_coordinator_assemble_resolves_correct_prompt(self):
        # Setup mock db
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        MemoryBase.metadata.create_all(engine)
        SessionLocal = sessionmaker(bind=engine)
        db = SessionLocal()

        try:
            # Setup mock skills
            marketing_manifest = SkillManifest(
                id="marketing",
                name="Marketing Assistant",
                description="Marketing",
                prompt_family="BUSINESS_PROMPT"
            )
            mock_loader = MockSkillLoader({
                "marketing": (marketing_manifest, "Marketing prompt content")
            })

            snapshot = SessionSnapshot(session_id="session_1", turn_number=1)

            # Create coordinator contract
            contract = RuntimeContract(
                request_id="req1",
                session_id="session_1",
                model_id="chat",
                user_message="hablemos de marketing", # Will trigger gate in test since we mock analyze_intent or set manual skill
                manual_skill="marketing",
                timestamp=123456.78,
                snapshot=snapshot
            )

            # Assemble prompt
            coordinator = PureCoordinator()
            manifest = coordinator.assemble(db, contract, mock_loader)
            
            # Assert that the system prompt snapshot contains the Business identity prompt
            self.assertIn("experto en negocios", manifest.system_prompt_snapshot)
            self.assertNotIn("DIAGNÓSTICO", manifest.system_prompt_snapshot)

            # Create neutral coordinator contract
            contract_neutral = RuntimeContract(
                request_id="req2",
                session_id="session_1",
                model_id="chat",
                user_message="que se habla en este documento",
                manual_skill=None,
                timestamp=123456.78,
                snapshot=snapshot
            )

            manifest_neutral = coordinator.assemble(db, contract_neutral, mock_loader)
            # Assert that the neutral contract resolves to the General Analytical prompt
            self.assertIn("asistente de inteligencia artificial", manifest_neutral.system_prompt_snapshot)
            self.assertNotIn("DIAGNÓSTICO", manifest_neutral.system_prompt_snapshot)
            self.assertNotIn("experto en negocios", manifest_neutral.system_prompt_snapshot)

        finally:
            db.close()

if __name__ == "__main__":
    unittest.main()
