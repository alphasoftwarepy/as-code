import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
from runtime.skills.models import SkillManifest, SkillStatus
from runtime.capabilities.registry import get_capability_registry

logger = logging.getLogger("as-code.runtime.skills")

class SkillLoader:
    def __init__(self, skills_dir: Optional[str] = None):
        if skills_dir:
            self.skills_dir = Path(skills_dir)
        else:
            self.skills_dir = Path("skills")
            if not self.skills_dir.exists():
                self.skills_dir = Path(__file__).parents[2] / "skills"

        self.skills: Dict[str, Tuple[SkillManifest, str]] = {}

    def load_skills(self) -> None:
        """Scan skills_dir and load manifests and system prompts."""
        self.skills.clear()
        if not self.skills_dir.exists():
            logger.warning(f"Skills directory does not exist: {self.skills_dir}")
            return

        logger.info(f"Scanning for skills in: {self.skills_dir.resolve()}")
        for item in self.skills_dir.iterdir():
            if item.is_dir():
                manifest_path = item / "manifest.json"
                prompt_path = item / "prompt.md"

                if manifest_path.exists() and prompt_path.exists():
                    try:
                        with open(manifest_path, "r", encoding="utf-8") as f:
                            manifest_data = json.load(f)
                        manifest = SkillManifest(**manifest_data)
                        
                        with open(prompt_path, "r", encoding="utf-8") as f:
                            prompt_content = f.read().strip()

                        if not manifest.id:
                            manifest.id = item.name

                        self.skills[manifest.id] = (manifest, prompt_content)
                        logger.info(f"Loaded skill: {manifest.id} ({manifest.name})")
                    except Exception as e:
                        logger.error(f"Failed to load skill from {item.name}: {e}")
        
        logger.info(f"Skills loaded: {', '.join(self.skills.keys())}")

    def get_skill_prompt(self, skill_id: str) -> Optional[str]:
        """Retrieve the system prompt for a specific skill if registered."""
        skill = self.skills.get(skill_id)
        if skill:
            return skill[1]
        return None

    def evaluate_skills(self, settings, app_state=None) -> Dict[str, SkillStatus]:
        """Verify compatibility of loaded skills against dynamic capabilities."""
        registry = get_capability_registry()
        cap_statuses = registry.check_all(settings, app_state)

        # Collect all active scopes (available & enabled)
        active_scopes = set()
        for cap_status in cap_statuses.values():
            if cap_status.available and cap_status.enabled:
                active_scopes.update(cap_status.scopes)

        evaluated = {}
        for skill_id, (manifest, _) in self.skills.items():
            if not manifest.enabled:
                evaluated[skill_id] = SkillStatus(
                    id=manifest.id,
                    name=manifest.name,
                    description=manifest.description,
                    compatible=False,
                    reason="Skill is disabled in manifest",
                    enabled=False
                )
                continue

            # Verify required scopes
            missing_scopes = [scope for scope in manifest.required_scopes if scope not in active_scopes]
            
            if missing_scopes:
                reasons = []
                for ms in missing_scopes:
                    provided_by = None
                    for cap_id, cap_def in registry.capabilities.items():
                        if ms in cap_def.scopes:
                            provided_by = cap_id
                            break
                    if provided_by:
                        cap_status = cap_statuses.get(provided_by)
                        if cap_status:
                            if not cap_status.available:
                                reasons.append(f"{ms} (capability '{provided_by}' is unavailable: {cap_status.reason or 'missing dependencies'})")
                            elif not cap_status.enabled:
                                reasons.append(f"{ms} (capability '{provided_by}' is disabled by user overrides)")
                            else:
                                reasons.append(f"{ms} (capability '{provided_by}' status is {cap_status.status})")
                        else:
                            reasons.append(f"{ms} (missing capability '{provided_by}')")
                    else:
                        reasons.append(ms)

                reason_str = f"Missing required scopes: {', '.join(reasons)}"
                compatible = False
            else:
                reason_str = None
                compatible = True

            evaluated[skill_id] = SkillStatus(
                id=manifest.id,
                name=manifest.name,
                description=manifest.description,
                compatible=compatible,
                reason=reason_str,
                enabled=True
            )

        return evaluated

# Global loader instance
_global_loader = SkillLoader()

def get_skill_loader() -> SkillLoader:
    """Access the global SkillLoader singleton."""
    return _global_loader
