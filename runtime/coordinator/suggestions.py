import logging
from typing import List, Optional
from sqlalchemy.orm import Session
from config.settings import get_settings
from runtime.skills.loader import get_skill_loader
from runtime.coordinator.intent import analyze_intent
from runtime.coordinator.models import WorkflowState

logger = logging.getLogger("as-code.runtime.coordinator.suggestions")

def get_suggested_skills(
    db: Session,
    session_id: str,
    user_message: str,
    workflow_state: WorkflowState
) -> List[str]:
    """
    Produce a list of recommended skills based on intent, memory state, and workflow.
    Ensures recommended skills are compatible and enabled in the runtime.
    """
    # 1. Get skills detected by intent analyzer
    detected = analyze_intent(user_message, db, session_id)

    # 2. Get list of compatible/enabled skills from the SkillLoader
    loader = get_skill_loader()
    # evaluate_skills returns Dict[str, SkillStatus]
    evaluated = loader.evaluate_skills(get_settings())
    compatible_skills = {
        sid for sid, status in evaluated.items()
        if status.enabled and getattr(status, "compatible", True)
    }

    # 3. Compile suggestions
    suggestions = []
    
    # Priority 1: If there is an inferred skill that is different from currently active skill
    for skill_id in detected:
        if skill_id in compatible_skills and skill_id != workflow_state.active_skill:
            if skill_id not in suggestions:
                suggestions.append(skill_id)

    # Priority 2: Fallback to active skill if it's compatible but not manually active in header
    # (Just in case, but usually suggestions are to switch/add skills)
    
    # Priority 3: Default fallback suggestions if no matches
    if not suggestions:
        # Suggest top available compatible skills
        for default_skill in ["programming", "business", "marketing", "sales", "legal", "content_creator"]:
            if default_skill in compatible_skills and default_skill != workflow_state.active_skill:
                suggestions.append(default_skill)

    return suggestions[:3]  # Return max 3 recommendations
