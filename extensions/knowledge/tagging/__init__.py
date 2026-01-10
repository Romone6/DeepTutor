"""Question Tagging Module.

Provides archetype and error trap tagging for past-paper questions.
"""

from .base import (
    QuestionArchetype,
    ErrorTrap,
    DifficultyLevel,
    TagSet,
    TaggedQuestion,
    TagManager,
    suggest_archetype,
    suggest_error_traps,
)

__all__ = [
    "QuestionArchetype",
    "ErrorTrap",
    "DifficultyLevel",
    "TagSet",
    "TaggedQuestion",
    "TagManager",
    "suggest_archetype",
    "suggest_error_traps",
]
