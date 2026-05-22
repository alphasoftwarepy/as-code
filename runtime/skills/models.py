from pydantic import BaseModel
from typing import List, Optional

class SkillManifest(BaseModel):
    id: str
    name: str
    description: str
    required_scopes: List[str] = []
    enabled: bool = True

class SkillStatus(BaseModel):
    id: str
    name: str
    description: str
    compatible: bool
    reason: Optional[str] = None
    enabled: bool
