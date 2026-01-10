"""Extensions agents module.

Agents for specialized tasks like marking and feedback.

Usage:
    from extensions.agents import Examiner, MarkingResult, should_trigger_examiner

    # Check if examiner should run
    if should_trigger_examiner(intent, policy_id, strictness):
        result = evaluate_answer(answer, guidelines, subject="mathematics")
"""

from .examiner import (
    Examiner,
    MarkingResult,
    MarkBreakdown,
    MarkAllocation,
    FixItem,
    RewriteTarget,
    MarkingCriterion,
    AnswerEvaluation,
    IntentType,
    should_trigger_examiner,
    evaluate_answer,
)

__all__ = [
    "Examiner",
    "MarkingResult",
    "MarkBreakdown",
    "MarkAllocation",
    "FixItem",
    "RewriteTarget",
    "MarkingCriterion",
    "AnswerEvaluation",
    "IntentType",
    "should_trigger_examiner",
    "evaluate_answer",
]
