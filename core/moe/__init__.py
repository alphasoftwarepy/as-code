"""
AS-Core — MoE Residency Engine Core Package
============================================
Exporta las abstracciones fundamentales para modelado, indexación y gestión de residencia
de modelos Mixture-of-Experts.
"""

from core.moe.cuda_driver import CUDADriver, CUDADriverError
from core.moe.cublas_backend import CuBLASBackend, CuBLASError
from core.moe.expert_tensor import (
    ExpertTensor,
    ExpertTensorSlice,
    ResidencyTier,
    TensorRole,
)
from core.moe.model_profile import ModelProfile
from core.moe.expert_registry import ExpertRegistry
from core.moe.vram_pool import VRAMExpertPool, VRAMSlot
from core.moe.ram_warm_pool import RAMWarmPool, WarmSlot
from core.moe.single_expert_executor import (
    SingleExpertExecutionResult,
    SingleExpertExecutor,
)
from core.moe.multi_expert_executor import (
    MultiExpertExecutionResult,
    MultiExpertExecutor,
)
from core.moe.router import (
    RealRouter,
    RoutedExpert,
    RoutedLayerDispatch,
    RoutingDecision,
)
from core.moe.residency_manager import (
    ResidencyDecision,
    ResidencyLayerDispatch,
    ResidencyManager,
)
from core.moe.dynamic_residency_engine import (
    DynamicLayerDispatch,
    DynamicResidencyDecision,
    DynamicResidencyEngine,
    ResidencySource,
)
from core.moe.layer_executor import (
    MoELayerExecutionResult,
    MoELayerExecutor,
)
from core.moe.routing_tracer import (
    RoutingTraceEvent,
    RoutingTracer,
)
from core.moe.frequency_analyzer import (
    FrequencyAnalyzer,
    LayerFrequencyStats,
)

__all__ = [
    "CUDADriver",
    "CUDADriverError",
    "CuBLASBackend",
    "CuBLASError",
    "ExpertTensor",
    "ExpertTensorSlice",
    "ResidencyTier",
    "TensorRole",
    "ModelProfile",
    "ExpertRegistry",
    "VRAMExpertPool",
    "VRAMSlot",
    "RAMWarmPool",
    "WarmSlot",
    "SingleExpertExecutor",
    "SingleExpertExecutionResult",
    "MultiExpertExecutor",
    "MultiExpertExecutionResult",
    "RealRouter",
    "RoutingDecision",
    "RoutedExpert",
    "RoutedLayerDispatch",
    "ResidencyDecision",
    "ResidencyLayerDispatch",
    "ResidencyManager",
    "DynamicResidencyEngine",
    "DynamicResidencyDecision",
    "DynamicLayerDispatch",
    "ResidencySource",
    "MoELayerExecutor",
    "MoELayerExecutionResult",
    "RoutingTracer",
    "RoutingTraceEvent",
    "FrequencyAnalyzer",
    "LayerFrequencyStats",
]
