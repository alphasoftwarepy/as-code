"""AS Code — Core package"""

from core.hardware import HardwareInfo, detect_hardware
from core.engine import EngineManager

__all__ = ["HardwareInfo", "detect_hardware", "EngineManager"]
