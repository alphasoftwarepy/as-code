from fastapi import APIRouter, Depends, Request
from typing import Dict
from config.settings import Settings, get_settings
from runtime.capabilities.registry import get_capability_registry
from runtime.capabilities.models import CapabilityStatus

capabilities_router = APIRouter(prefix="/v1", tags=["Capabilities"])

@capabilities_router.get("/capabilities", response_model=Dict[str, CapabilityStatus])
def get_capabilities(
    request: Request,
    settings: Settings = Depends(get_settings)
):
    """Evaluate and retrieve current runtime capabilities dynamically."""
    registry = get_capability_registry()
    # Pass the app state to allow service checking (like active rag_service instance)
    app_state = request.app.state
    return registry.check_all(settings, app_state)
