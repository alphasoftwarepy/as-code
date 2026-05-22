import sys
from runtime.capabilities.base import BaseCapability
from runtime.capabilities.models import CapabilityStatus

class TerminalCapability(BaseCapability):
    id = "terminal"
    name = "Terminal Execution"
    description = "Run terminal commands and shell processes directly on host system"
    category = "developer"
    version = "1.0.0"
    scopes = ["terminal.execute"]

    def check(self, settings, app_state=None) -> CapabilityStatus:
        available = True
        reason = None
        
        # Security decision: defaults to False (must be explicitly enabled in overrides)
        enabled = False
        if hasattr(settings, "capability_overrides"):
            enabled = settings.capability_overrides.get(self.id, False)
        
        if not enabled:
            reason = "Disabled for security reasons. Can be enabled via capability overrides."

        return CapabilityStatus(
            id=self.id,
            name=self.name,
            description=self.description,
            category=self.category,
            version=self.version,
            available=available,
            enabled=enabled,
            provider="powershell" if sys.platform == "win32" else "bash",
            status="healthy" if enabled else "offline",
            reason=reason,
            scopes=self.scopes
        )
