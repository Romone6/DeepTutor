"""
Degradation Service Package
===========================

Provides graceful degradation when components fail.
"""

from .service import (
    ComponentType,
    ComponentStatus,
    DegradationRule,
    DegradationService,
    get_degradation_service,
    reset_degradation_service,
)

__all__ = [
    "ComponentType",
    "ComponentStatus",
    "DegradationRule",
    "DegradationService",
    "get_degradation_service",
    "reset_degradation_service",
]
