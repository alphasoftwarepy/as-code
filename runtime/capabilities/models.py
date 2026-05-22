from pydantic import BaseModel
from typing import Optional, List

class CapabilityStatus(BaseModel):
    id: str
    name: str
    description: str
    category: str
    version: str
    available: bool
    enabled: bool
    provider: Optional[str] = None
    status: str = "healthy"  # healthy | degraded | offline
    reason: Optional[str] = None
    scopes: List[str] = []
