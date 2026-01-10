"""
Progress Dashboard API Router
============================

REST endpoints for the per-subject progress dashboard.

Features:
- Subject-level metrics (accuracy, time spent)
- Topic-level metrics (weak topics)
- Accuracy trend over time
- Time allocation vs syllabus weight

Requires PID-32.
"""

import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

_project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from src.logging import get_logger
from src.services.config import load_config_with_main
from extensions.memory import get_memory_store

project_root = Path(__file__).parent.parent.parent.parent
config = load_config_with_main("solve_config.yaml", project_root)
log_dir = config.get("paths", {}).get("user_log_dir") or config.get("logging", {}).get("log_dir")
logger = get_logger("DashboardAPI", level="INFO", log_dir=log_dir)

router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================


class RecordMetricRequest(BaseModel):
    """Request to record a study metric."""

    session_id: str
    subject: str
    topic: str
    difficulty: str = "medium"
    activity_type: Optional[str] = None
    time_spent_seconds: float = 0
    questions_attempted: int = 0
    questions_correct: int = 0
    confidence_avg: float = 0.0
    plan_id: Optional[str] = None
    day_number: Optional[int] = None


class DashboardSummaryResponse(BaseModel):
    """Dashboard summary response."""

    period_days: int
    subject: Optional[str]
    overall_stats: dict
    by_subject: list[dict]
    weak_topics: list[dict]
    accuracy_trend: list[dict]
    time_allocation: list[dict]


class SubjectMetricsResponse(BaseModel):
    """Subject-level metrics response."""

    subjects: list[dict]
    total_subjects: int


class TopicMetricsResponse(BaseModel):
    """Topic-level metrics response."""

    topics: list[dict]


class AccuracyTrendResponse(BaseModel):
    """Accuracy trend response."""

    trend: list[dict]


class TimeAllocationResponse(BaseModel):
    """Time allocation response."""

    allocation: list[dict]


# =============================================================================
# Helper Functions
# =============================================================================


def get_memory():
    """Get memory store instance."""
    return get_memory_store()


# =============================================================================
# Endpoints
# =============================================================================


@router.post("/metrics/record")
async def record_metric(request: RecordMetricRequest):
    """
    Record a study session metric.

    Call this after completing study activities to track progress.
    """
    try:
        store = get_memory()

        metric_id = store.record_study_metric(
            session_id=request.session_id,
            subject=request.subject,
            topic=request.topic,
            difficulty=request.difficulty,
            activity_type=request.activity_type,
            time_spent_seconds=request.time_spent_seconds,
            questions_attempted=request.questions_attempted,
            questions_correct=request.questions_correct,
            confidence_avg=request.confidence_avg,
            plan_id=request.plan_id,
            day_number=request.day_number,
        )

        logger.info(f"Recorded metric {metric_id}: {request.subject}/{request.topic}")

        return {
            "metric_id": metric_id,
            "message": "Metric recorded successfully",
            "subject": request.subject,
            "topic": request.topic,
        }

    except Exception as e:
        logger.error(f"Failed to record metric: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to record metric: {str(e)}",
        )


@router.get("/dashboard/summary", response_model=DashboardSummaryResponse)
async def get_dashboard_summary(
    subject: Optional[str] = None,
    days: int = Query(default=30, ge=1, le=365),
):
    """
    Get complete dashboard summary.

    Includes overall stats, subject breakdown, weak topics, accuracy trend,
    and time allocation.
    """
    try:
        store = get_memory()
        summary = store.get_dashboard_summary(subject=subject, days=days)

        return DashboardSummaryResponse(**summary)

    except Exception as e:
        logger.error(f"Failed to get dashboard summary: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get dashboard summary: {str(e)}",
        )


@router.get("/metrics/subjects", response_model=SubjectMetricsResponse)
async def get_subject_metrics(
    subject: Optional[str] = None,
    days: int = Query(default=30, ge=1, le=365),
):
    """
    Get metrics grouped by subject.

    Returns accuracy, time spent, and question counts per subject.
    """
    try:
        store = get_memory()
        result = store.get_subject_metrics(subject=subject, days=days)

        return SubjectMetricsResponse(**result)

    except Exception as e:
        logger.error(f"Failed to get subject metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get subject metrics: {str(e)}",
        )


@router.get("/metrics/topics", response_model=TopicMetricsResponse)
async def get_topic_metrics(
    subject: Optional[str] = None,
    days: int = Query(default=30, ge=1, le=365),
    limit: int = Query(default=20, ge=1, le=100),
):
    """
    Get metrics per topic.

    Returns accuracy and time spent per topic, sorted by accuracy (worst first).
    """
    try:
        store = get_memory()
        topics = store.get_topic_metrics(subject=subject, days=days, limit=limit)

        return TopicMetricsResponse(topics=topics)

    except Exception as e:
        logger.error(f"Failed to get topic metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get topic metrics: {str(e)}",
        )


@router.get("/metrics/weak-topics")
async def get_weak_topics(
    subject: Optional[str] = None,
    accuracy_threshold: float = Query(default=70.0, ge=0, le=100),
    days: int = Query(default=30, ge=1, le=365),
    limit: int = Query(default=10, ge=1, le=50),
):
    """
    Get topics with accuracy below threshold.

    These are the "weak" topics that need more practice.
    """
    try:
        store = get_memory()
        topics = store.get_weak_topics(
            subject=subject,
            accuracy_threshold=accuracy_threshold,
            days=days,
            limit=limit,
        )

        return {
            "topics": topics,
            "count": len(topics),
            "threshold": accuracy_threshold,
            "period_days": days,
        }

    except Exception as e:
        logger.error(f"Failed to get weak topics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get weak topics: {str(e)}",
        )


@router.get("/metrics/trend", response_model=AccuracyTrendResponse)
async def get_accuracy_trend(
    subject: Optional[str] = None,
    days: int = Query(default=14, ge=1, le=90),
):
    """
    Get daily accuracy trend over time.

    Shows how accuracy changes day by day.
    """
    try:
        store = get_memory()
        trend = store.get_accuracy_trend(subject=subject, days=days)

        return AccuracyTrendResponse(trend=trend)

    except Exception as e:
        logger.error(f"Failed to get accuracy trend: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get accuracy trend: {str(e)}",
        )


@router.get("/metrics/time-allocation", response_model=TimeAllocationResponse)
async def get_time_allocation(
    subject: Optional[str] = None,
    days: int = Query(default=30, ge=1, le=365),
):
    """
    Get time allocation per topic.

    Shows how study time is distributed across topics.
    """
    try:
        store = get_memory()
        allocation = store.get_time_allocation(subject=subject, days=days)

        return TimeAllocationResponse(allocation=allocation)

    except Exception as e:
        logger.error(f"Failed to get time allocation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get time allocation: {str(e)}",
        )


@router.get("/metrics/overview")
async def get_metrics_overview(days: int = Query(default=7, ge=1, le=90)):
    """
    Get a quick overview of all metrics.

    Returns summary stats for the dashboard home.
    """
    try:
        store = get_memory()

        subject_metrics = store.get_subject_metrics(days=days)
        weak_topics = store.get_weak_topics(days=days, limit=5)
        recent_trend = store.get_accuracy_trend(days=min(days, 7))

        total_questions = sum(s.get("total_questions", 0) for s in subject_metrics["subjects"])
        correct_questions = sum(s.get("correct_questions", 0) for s in subject_metrics["subjects"])
        total_time = sum(s.get("total_time_minutes", 0) for s in subject_metrics["subjects"])

        overall_accuracy = (
            (correct_questions / total_questions * 100) if total_questions > 0 else 0.0
        )

        latest_accuracy = recent_trend[-1]["accuracy"] if recent_trend else 0.0

        return {
            "period_days": days,
            "overall": {
                "total_questions": total_questions,
                "correct_questions": correct_questions,
                "accuracy": overall_accuracy,
                "study_time_minutes": total_time,
            },
            "subjects": subject_metrics["subjects"],
            "weak_topics_count": len(weak_topics),
            "latest_daily_accuracy": latest_accuracy,
        }

    except Exception as e:
        logger.error(f"Failed to get metrics overview: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get metrics overview: {str(e)}",
        )
