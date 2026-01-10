"""
Memory and Weakness Tracking API Router
=======================================

Handles weakness tracking, review scheduling, and interaction logging.
"""

import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

_project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from extensions.memory import (
    ErrorType,
    ConfidenceLevel,
    get_memory_store,
    WeaknessRecord,
)

router = APIRouter()


class AddWeaknessRequest(BaseModel):
    subject: str
    topic: str
    subtopic: str = ""
    error_type: str
    description: str = ""
    question_id: str | None = None
    user_answer: str = ""
    correct_answer: str = ""
    confidence: str = "medium"


class LogInteractionRequest(BaseModel):
    session_id: str
    subject: str
    topic: str
    question_id: str | None = None
    correct: bool
    confidence: str
    duration_ms: float | None = None
    metadata: dict[str, Any] | None = None


class CompleteReviewRequest(BaseModel):
    quality: int  # 0-5 rating
    correct: bool


class AddToQueueRequest(BaseModel):
    priority: float = 0.5
    hours: float = 24.0


@router.get("/memory/stats")
def get_memory_stats():
    """Get overall memory store statistics."""
    store = get_memory_store()
    return store.get_stats()


@router.get("/memory/weak-spots")
def get_weak_spots(
    subject: str | None = None,
    min_mastery: float = 0.0,
    max_mastery: float = 1.0,
    limit: int = 50,
):
    """Get list of weak topics, optionally filtered by subject."""
    store = get_memory_store()
    return store.get_weakness_topics(
        subject=subject,
        min_mastery=min_mastery,
        max_mastery=max_mastery,
        limit=limit,
    )


@router.get("/memory/weakness-records")
def get_weakness_records(
    subject: str | None = None,
    topic: str | None = None,
    error_type: str | None = None,
    limit: int = 100,
):
    """Get weakness records with optional filters."""
    store = get_memory_store()
    error_enum = None
    if error_type:
        try:
            error_enum = ErrorType(error_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid error_type: {error_type}")

    records = store.get_weakness_records(
        subject=subject,
        topic=topic,
        error_type=error_enum,
        limit=limit,
    )
    return [r.to_dict() for r in records]


@router.post("/memory/weakness")
def add_weakness(request: AddWeaknessRequest):
    """Add a new weakness record."""
    store = get_memory_store()
    try:
        error_enum = ErrorType(request.error_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid error_type: {request.error_type}")

    try:
        confidence_enum = ConfidenceLevel(request.confidence)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid confidence: {request.confidence}")

    weakness_id = store.add_weakness(
        subject=request.subject,
        topic=request.topic,
        subtopic=request.subtopic,
        error_type=error_enum,
        description=request.description,
        question_id=request.question_id,
        user_answer=request.user_answer,
        correct_answer=request.correct_answer,
        confidence=confidence_enum,
    )
    return {"id": weakness_id, "message": "Weakness recorded successfully"}


@router.post("/memory/review-queue")
def add_to_review_queue(weakness_id: int, request: AddToQueueRequest):
    """Add a weakness to the review queue."""
    store = get_memory_store()

    records = store.get_weakness_records(limit=1)
    if not any(r.id == weakness_id for r in records):
        raise HTTPException(status_code=404, detail="Weakness not found")

    record = next(r for r in records if r.id == weakness_id)

    review_id = store.add_to_review_queue(
        weakness_id=weakness_id,
        subject=record.subject,
        topic=record.topic,
        priority=request.priority,
        hours=request.hours,
    )
    return {"id": review_id, "message": "Added to review queue"}


@router.get("/memory/review-queue")
def get_review_queue(limit: int = 20):
    """Get items due for review."""
    store = get_memory_store()
    queue = store.get_review_queue(limit=limit)
    return [item.to_dict() for item in queue]


@router.post("/memory/review/{review_id}/complete")
def complete_review(review_id: int, request: CompleteReviewRequest):
    """Mark a review as complete and update spaced repetition."""
    store = get_memory_store()

    if not (0 <= request.quality <= 5):
        raise HTTPException(status_code=400, detail="Quality must be between 0 and 5")

    store.complete_review(
        review_id=review_id,
        quality=request.quality,
        correct=request.correct,
    )
    return {"message": "Review completed successfully"}


@router.post("/memory/interaction")
def log_interaction(request: LogInteractionRequest):
    """Log an interaction for tracking."""
    store = get_memory_store()

    try:
        confidence_enum = ConfidenceLevel(request.confidence)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid confidence: {request.confidence}")

    log_id = store.log_interaction(
        session_id=request.session_id,
        subject=request.subject,
        topic=request.topic,
        question_id=request.question_id,
        correct=request.correct,
        confidence=confidence_enum,
        duration_ms=request.duration_ms,
        metadata=request.metadata,
    )
    return {"id": log_id, "message": "Interaction logged successfully"}


@router.get("/memory/interaction-stats")
def get_interaction_stats(subject: str | None = None, days: int = 7):
    """Get interaction statistics."""
    store = get_memory_store()
    return store.get_interaction_stats(subject=subject, days=days)


@router.get("/memory/topic-mastery")
def get_topic_mastery(subject: str | None = None):
    """Get mastery scores per topic."""
    store = get_memory_store()
    return store.get_topic_mastery(subject=subject)


@router.get("/memory/weakness-counts")
def get_weakness_counts():
    """Get count of weaknesses per subject."""
    store = get_memory_store()
    return store.get_all_weakness_counts()


@router.post("/memory/weakness/{weakness_id}/mastery")
def update_mastery(weakness_id: int, correct: bool):
    """Update mastery score for a weakness."""
    store = get_memory_store()
    store.update_mastery(weakness_id=weakness_id, correct=correct)
    return {"message": "Mastery updated successfully"}
