"""Extensions router module."""

from .intent_router import (
    IntentRouter,
    IntentClassification,
    RouterResponse,
    Subject,
    ConfidenceLevel,
    create_intent_router,
    classify_message,
    route_message,
)
from .model_router import (
    ModelRouter,
    ModelRouterResponse,
    ModelTarget,
    ModelType,
    RoutingContext,
    PolicyMode,
    ModelRouterConfig,
    create_model_router,
    route_to_model,
)

__all__ = [
    # Intent Router
    "IntentRouter",
    "IntentClassification",
    "RouterResponse",
    "Subject",
    "ConfidenceLevel",
    "create_intent_router",
    "classify_message",
    "route_message",
    # Model Router
    "ModelRouter",
    "ModelRouterResponse",
    "ModelTarget",
    "ModelType",
    "RoutingContext",
    "PolicyMode",
    "ModelRouterConfig",
    "create_model_router",
    "route_to_model",
]
