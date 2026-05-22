from typing import Dict, Optional
from runtime.capabilities.base import BaseCapability
from runtime.capabilities.models import CapabilityStatus
from runtime.capabilities.documents import DocumentsCapability
from runtime.capabilities.rag import RAGCapability
from runtime.capabilities.git import GitCapability
from runtime.capabilities.terminal import TerminalCapability

class CapabilityRegistry:
    def __init__(self):
        self.capabilities: Dict[str, BaseCapability] = {}
        # Register current 4 real capabilities
        self.register(DocumentsCapability())
        self.register(RAGCapability())
        self.register(GitCapability())
        self.register(TerminalCapability())

    def register(self, capability: BaseCapability) -> None:
        self.capabilities[capability.id] = capability

    def get(self, capability_id: str) -> Optional[BaseCapability]:
        return self.capabilities.get(capability_id)

    def check_all(self, settings, app_state=None) -> Dict[str, CapabilityStatus]:
        """Perform dynamic/lazy evaluation of all registered capabilities."""
        statuses = {}
        for cap_id, cap in self.capabilities.items():
            try:
                statuses[cap_id] = cap.check(settings, app_state)
            except Exception as e:
                # Graceful fallback if checking a specific capability raises an error
                statuses[cap_id] = CapabilityStatus(
                    id=cap_id,
                    name=cap.name,
                    description=cap.description,
                    category=cap.category,
                    version=cap.version,
                    available=False,
                    enabled=False,
                    provider="none",
                    status="offline",
                    reason=f"Check execution failed: {str(e)}",
                    scopes=cap.scopes
                )
        return statuses

# Global registry instance
_global_registry = CapabilityRegistry()

def get_capability_registry() -> CapabilityRegistry:
    """Access the global CapabilityRegistry singleton."""
    return _global_registry
