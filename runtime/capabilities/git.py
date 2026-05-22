import shutil
from runtime.capabilities.base import BaseCapability
from runtime.capabilities.models import CapabilityStatus

class GitCapability(BaseCapability):
    id = "git"
    name = "Git Integration"
    description = "Read repository status, diffs, and manage branches/commits locally"
    category = "tools"
    version = "1.0.0"
    scopes = ["git.read", "git.write"]

    def check(self, settings, app_state=None) -> CapabilityStatus:
        git_path = shutil.which("git")
        available = git_path is not None
        reason = None
        if not available:
            reason = "Git CLI executable not found on host machine path"

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
            provider="git-cli",
            status="healthy" if available else "offline",
            reason=reason,
            scopes=self.scopes
        )
