import importlib
from runtime.capabilities.base import BaseCapability
from runtime.capabilities.models import CapabilityStatus

class DocumentsCapability(BaseCapability):
    id = "documents"
    name = "Document Parsing"
    description = "Extract and parse plain text, PDF, and DOCX files"
    category = "documents"
    version = "1.0.0"
    scopes = ["documents.read", "documents.write"]

    def check(self, settings, app_state=None) -> CapabilityStatus:
        missing = []
        for pkg in ["pypdf", "docx"]:
            try:
                importlib.import_module(pkg)
            except ImportError:
                missing.append(pkg)

        available = len(missing) == 0
        reason = None
        if not available:
            reason = f"Missing Python libraries: {', '.join(missing)}"

        # Check user config override (defaults to True)
        enabled = available
        if hasattr(settings, "capability_overrides"):
            enabled = settings.capability_overrides.get(self.id, True) and available
        else:
            enabled = available

        return CapabilityStatus(
            id=self.id,
            name=self.name,
            description=self.description,
            category=self.category,
            version=self.version,
            available=available,
            enabled=enabled,
            provider="python-docx/pypdf",
            status="healthy" if available else "offline",
            reason=reason,
            scopes=self.scopes
        )
