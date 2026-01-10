from pydantic import BaseModel, Field
from typing import Literal


class RouteDecision(BaseModel):
    """Shared schema for router decisions across extensions."""

    target: Literal["router", "voice", "knowledge", "policies", "agents"] = Field(
        ...,
        description="Extension category to route to",
    )
    handler: str = Field(
        ...,
        description="Specific handler name",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Additional routing metadata",
    )
