from runtime.capabilities.base import BaseCapability
from runtime.capabilities.models import CapabilityStatus

class RAGCapability(BaseCapability):
    id = "rag"
    name = "RAG Memory"
    description = "NotebookLM-style semantic and keyword search memory over ingested files"
    category = "core"
    version = "1.0.0"
    scopes = ["rag.retrieve", "rag.index"]

    def check(self, settings, app_state=None) -> CapabilityStatus:
        enable_rag_mode = getattr(settings, "enable_rag_mode", False)
        
        available = enable_rag_mode
        reason = None
        status = "healthy"

        if not enable_rag_mode:
            available = False
            reason = "RAG mode is disabled globally (ASCODE_ENABLE_RAG_MODE=false)"
            status = "offline"
        elif app_state is not None:
            # Check runtime state
            rag_service = getattr(app_state, "rag_service", None)
            if rag_service is None:
                status = "degraded"
                reason = "RAG service failed to initialize on startup"

        # Check override (defaults to True)
        enabled = available
        if hasattr(settings, "capability_overrides"):
            enabled = enabled and settings.capability_overrides.get(self.id, True)

        return CapabilityStatus(
            id=self.id,
            name=self.name,
            description=self.description,
            category=self.category,
            version=self.version,
            available=available,
            enabled=enabled,
            provider="faiss/sqlite/bge",
            status=status,
            reason=reason,
            scopes=self.scopes
        )
