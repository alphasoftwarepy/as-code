from runtime.capabilities.models import CapabilityStatus

class BaseCapability:
    id: str
    name: str
    description: str
    category: str
    version: str = "1.0.0"
    scopes: list[str] = []

    def check(self, settings, app_state=None) -> CapabilityStatus:
        """Perform lightweight validation of the capability and return its status.
        
        Args:
            settings: The application settings instance.
            app_state: FastAPI app state (optional, for checking active services).
        """
        raise NotImplementedError
