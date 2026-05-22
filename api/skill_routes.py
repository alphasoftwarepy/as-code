from fastapi import APIRouter, Depends, Request
from typing import Dict
from config.settings import Settings, get_settings
from runtime.skills.loader import get_skill_loader
from runtime.skills.models import SkillStatus

skills_router = APIRouter(
    prefix="/v1/skills",
    tags=["skills"],
    redirect_slashes=False,   # Prevent 307→404 for empty-suffix routes
)

@skills_router.get("", response_model=Dict[str, SkillStatus])
def get_skills(
    request: Request,
    settings: Settings = Depends(get_settings)
):
    """Evaluate and retrieve all loaded skills and their capability compatibility."""
    loader = get_skill_loader()
    app_state = request.app.state
    return loader.evaluate_skills(settings, app_state)
