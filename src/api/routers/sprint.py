"""
Sprint Plan API Router
====================

REST endpoints for creating, viewing, and managing 21-day sprint study plans.

Features:
- Create sprint plans with subject selection
- View plan details and progress
- Update daily completion status
- Track sprint statistics
"""

import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

_project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from src.logging import get_logger
from src.services.config import load_config_with_main
from extensions.memory import get_memory_store
from extensions.agents.sprint_planner import (
    SprintPlanner,
    Subject,
    DifficultyLevel,
)

project_root = Path(__file__).parent.parent.parent.parent
config = load_config_with_main("solve_config.yaml", project_root)
log_dir = config.get("paths", {}).get("user_log_dir") or config.get("logging", {}).get("log_dir")
logger = get_logger("SprintAPI", level="INFO", log_dir=log_dir)

router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================


class CreateSprintRequest(BaseModel):
    """Request to create a new sprint plan."""

    subjects: list[str]
    days: int = 21
    daily_hours: float = 2.0
    start_date: Optional[str] = None
    name: Optional[str] = None
    kb_name: str = "default"


class UpdateDayRequest(BaseModel):
    """Request to update a sprint day."""

    completion_status: str  # pending, in_progress, completed
    notes: str = ""


class UpdatePlanStatusRequest(BaseModel):
    """Request to update sprint plan status."""

    status: str  # draft, active, completed, paused


class SprintPlanResponse(BaseModel):
    """Sprint plan response model."""

    id: str
    name: str
    subjects: list[str]
    total_days: int
    daily_hours: float
    start_date: str
    end_date: str
    status: str
    coverage_summary: dict


class SprintListResponse(BaseModel):
    """List of sprint plans."""

    plans: list[dict]
    total: int


# =============================================================================
# Helper Functions
# =============================================================================


def get_sprint_planner(kb_name: str = "default") -> SprintPlanner:
    """Get SprintPlanner instance."""
    return SprintPlanner(kb_name=kb_name)


def get_memory():
    """Get memory store instance."""
    return get_memory_store()


def calculate_coverage_summary(plan_data: dict, subjects: list[str]) -> dict:
    """Calculate topic coverage summary from plan data."""
    coverage = plan_data.get("topic_coverage", {})
    covered_count = sum(1 for v in coverage.values() if v)
    total_count = len(coverage)

    subject_summary = {}
    for subject in subjects:
        subject_topics = [k for k in coverage.keys() if subject in k]
        subject_covered = sum(1 for t in subject_topics if coverage.get(t, False))
        subject_summary[subject] = {
            "covered": subject_covered,
            "total": len(subject_topics),
            "percentage": ((subject_covered / len(subject_topics) * 100) if subject_topics else 0),
        }

    return {
        "total_covered": covered_count,
        "total_topics": total_count,
        "overall_percentage": ((covered_count / total_count * 100) if total_count > 0 else 0),
        "by_subject": subject_summary,
    }


# =============================================================================
# Endpoints
# =============================================================================


@router.post("/sprint/create", response_model=SprintPlanResponse)
async def create_sprint_plan(request: CreateSprintRequest):
    """
    Create a new 21-day sprint study plan.

    Generates a day-by-day plan covering all syllabus topics with
    active recall checkpoints and mini-quizzes.
    """
    try:
        logger.info(f"Creating sprint plan: {request.subjects} for {request.days} days")

        planner = get_sprint_planner(request.kb_name)
        plan = planner.create_plan(
            subjects=request.subjects,
            days=request.days,
            daily_hours=request.daily_hours,
            start_date=request.start_date,
            name=request.name,
        )

        plan_dict = plan.to_dict()
        plan_id = plan_dict["id"]

        store = get_memory()
        store.save_sprint_plan(plan_dict)

        coverage_summary = calculate_coverage_summary(plan_dict, request.subjects)

        logger.success(f"Sprint plan created: {plan_id}")

        return SprintPlanResponse(
            id=plan_id,
            name=plan.name,
            subjects=request.subjects,
            total_days=request.days,
            daily_hours=request.daily_hours,
            start_date=plan.start_date,
            end_date=plan.end_date,
            status=plan.status,
            coverage_summary=coverage_summary,
        )

    except Exception as e:
        logger.error(f"Failed to create sprint plan: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create sprint plan: {str(e)}",
        )


@router.get("/sprint/list")
async def list_sprint_plans():
    """
    List all sprint plans.

    Returns basic info about each plan (ID, name, subjects, status).
    """
    try:
        store = get_memory()
        plans = store.get_all_sprint_plans()

        return {
            "plans": plans,
            "total": len(plans),
        }

    except Exception as e:
        logger.error(f"Failed to list sprint plans: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list sprint plans: {str(e)}",
        )


@router.get("/sprint/{plan_id}")
async def get_sprint_plan(plan_id: str):
    """
    Get full details of a sprint plan.

    Includes day-by-day schedule with activities, quizzes, and checkpoints.
    """
    try:
        store = get_memory()
        plan = store.get_sprint_plan(plan_id)

        if not plan:
            raise HTTPException(status_code=404, detail="Sprint plan not found")

        coverage_summary = calculate_coverage_summary(plan, plan["subjects"])

        return {
            "plan": plan,
            "coverage_summary": coverage_summary,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get sprint plan {plan_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get sprint plan: {str(e)}",
        )


@router.get("/sprint/{plan_id}/summary")
async def get_sprint_summary(plan_id: str):
    """
    Get a quick summary of sprint plan progress.

    Returns overall progress, subject breakdown, and completion stats.
    """
    try:
        store = get_memory()
        plan = store.get_sprint_plan(plan_id)

        if not plan:
            raise HTTPException(status_code=404, detail="Sprint plan not found")

        days = plan.get("days", [])
        total_days = len(days)
        completed_days = sum(1 for d in days if d.get("completion_status") == "completed")
        in_progress_days = sum(1 for d in days if d.get("completion_status") == "in_progress")

        subject_progress = {}
        for subject in plan["subjects"]:
            subject_days = [d for d in days if d.get("subject") == subject]
            subject_completed = sum(
                1 for d in subject_days if d.get("completion_status") == "completed"
            )
            subject_progress[subject] = {
                "total": len(subject_days),
                "completed": subject_completed,
                "percentage": (
                    (subject_completed / len(subject_days) * 100) if subject_days else 0
                ),
            }

        return {
            "plan_id": plan_id,
            "name": plan["name"],
            "status": plan["status"],
            "overall_progress": {
                "total_days": total_days,
                "completed": completed_days,
                "in_progress": in_progress_days,
                "pending": total_days - completed_days - in_progress_days,
                "percentage": ((completed_days / total_days * 100) if total_days > 0 else 0),
            },
            "subject_progress": subject_progress,
            "coverage_summary": calculate_coverage_summary(plan, plan["subjects"]),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get sprint summary {plan_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get sprint summary: {str(e)}",
        )


@router.put("/sprint/{plan_id}/day/{day_number}")
async def update_sprint_day(plan_id: str, day_number: int, request: UpdateDayRequest):
    """
    Update the completion status of a specific day in a sprint plan.

    Use this to mark days as in_progress or completed.
    """
    try:
        if request.completion_status not in ["pending", "in_progress", "completed"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid status. Must be: pending, in_progress, or completed",
            )

        store = get_memory()
        plan = store.get_sprint_plan(plan_id)

        if not plan:
            raise HTTPException(status_code=404, detail="Sprint plan not found")

        success = store.update_sprint_day_status(
            plan_id, day_number, request.completion_status, request.notes
        )

        if not success:
            raise HTTPException(status_code=404, detail=f"Day {day_number} not found in plan")

        return {
            "message": f"Day {day_number} updated successfully",
            "completion_status": request.completion_status,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update sprint day {plan_id}/{day_number}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update sprint day: {str(e)}",
        )


@router.put("/sprint/{plan_id}/status")
async def update_sprint_status(plan_id: str, request: UpdatePlanStatusRequest):
    """
    Update the overall status of a sprint plan.

    Status options:
    - draft: Plan is being configured
    - active: Currently following the plan
    - paused: Temporarily paused
    - completed: Finished the plan
    """
    try:
        if request.status not in ["draft", "active", "completed", "paused"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid status. Must be: draft, active, paused, or completed",
            )

        store = get_memory()
        plan = store.get_sprint_plan(plan_id)

        if not plan:
            raise HTTPException(status_code=404, detail="Sprint plan not found")

        success = store.update_sprint_plan_status(plan_id, request.status)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to update status")

        return {
            "message": f"Plan status updated to {request.status}",
            "status": request.status,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update sprint status {plan_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update sprint status: {str(e)}",
        )


@router.delete("/sprint/{plan_id}")
async def delete_sprint_plan(plan_id: str):
    """
    Delete a sprint plan and all its days.
    """
    try:
        store = get_memory()
        plan = store.get_sprint_plan(plan_id)

        if not plan:
            raise HTTPException(status_code=404, detail="Sprint plan not found")

        success = store.delete_sprint_plan(plan_id)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete plan")

        return {
            "message": f"Sprint plan {plan_id} deleted successfully",
            "deleted": True,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete sprint plan {plan_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete sprint plan: {str(e)}",
        )


@router.get("/sprint/stats")
async def get_sprint_stats():
    """
    Get overall sprint statistics.

    Returns counts of plans, active plans, and completion rates.
    """
    try:
        store = get_memory()
        stats = store.get_sprint_stats()

        return stats

    except Exception as e:
        logger.error(f"Failed to get sprint stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get sprint stats: {str(e)}",
        )


@router.get("/sprint/subjects")
async def get_available_subjects():
    """
    Get list of available subjects for sprint planning.

    Returns subjects with their topic counts from the knowledge base.
    """
    try:
        planner = get_sprint_planner()
        extractor = planner.extractor

        subjects = []
        for subject in Subject:
            topics = extractor.get_topics_for_subject(subject.value)
            subjects.append(
                {
                    "id": subject.value,
                    "name": subject.value.title(),
                    "topic_count": len(topics),
                    "topics": [t.name for t in topics],
                }
            )

        return {
            "subjects": subjects,
            "total": len(subjects),
        }

    except Exception as e:
        logger.error(f"Failed to get subjects: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get subjects: {str(e)}",
        )


# =============================================================================
# Sprint Session Runner Endpoints (PID-31)
# =============================================================================


class QuizAnswerRequest(BaseModel):
    """Request to submit quiz answers."""

    question_id: str
    answer: str
    is_correct: bool
    time_spent_seconds: int = 0


class CheckpointRatingRequest(BaseModel):
    """Request to submit checkpoint self-ratings."""

    topic: str
    understanding_rating: int  # 1-5
    confidence_rating: int  # 1-5
    notes: str = ""


class CompleteSessionRequest(BaseModel):
    """Request to complete a sprint session."""

    quiz_answers: list[QuizAnswerRequest] = []
    checkpoint_ratings: list[CheckpointRatingRequest] = []
    total_time_seconds: int = 1800  # Default 30 minutes


class TodaySessionResponse(BaseModel):
    """Response for today's session."""

    session: dict
    catch_up_sessions: list[dict] = []
    is_catch_up: bool = False
    missed_days: int = 0


@router.get("/sprint/{plan_id}/today", response_model=TodaySessionResponse)
async def get_today_session(plan_id: str, target_date: str = None):
    """
    Get today's sprint session.

    If today has no session (e.g., weekend), returns the next available session.
    If days were missed, returns catch-up suggestions.
    """
    try:
        from datetime import datetime
        from extensions.agents.sprint_session_runner import SprintSessionRunner

        store = get_memory()
        plan = store.get_sprint_plan(plan_id)

        if not plan:
            raise HTTPException(status_code=404, detail="Sprint plan not found")

        runner = SprintSessionRunner(memory_store=store)

        target = target_date or datetime.now().isoformat()[:10]

        session = runner.load_today_session(plan_id, target)

        if session:
            return TodaySessionResponse(
                session=session.to_dict(),
                catch_up_sessions=[],
                is_catch_up=False,
                missed_days=0,
            )

        catch_up_sessions = runner.check_and_create_catch_up_sessions(plan_id)

        if catch_up_sessions:
            next_session = catch_up_sessions[0]
            return TodaySessionResponse(
                session=next_session.to_dict(),
                catch_up_sessions=[s.to_dict() for s in catch_up_sessions],
                is_catch_up=True,
                missed_days=len(catch_up_sessions),
            )

        return TodaySessionResponse(
            session=None,
            catch_up_sessions=[],
            is_catch_up=False,
            missed_days=0,
            message="No sessions available for the requested date",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get today session for {plan_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get today session: {str(e)}",
        )


@router.post("/sprint/{plan_id}/day/{day_number}/complete")
async def complete_sprint_session(
    plan_id: str,
    day_number: int,
    request: CompleteSessionRequest,
):
    """
    Complete a sprint session.

    Submits quiz answers and checkpoint ratings, updates weakness tracking,
    and schedules reviews for D+2 and D+7.
    """
    try:
        from extensions.agents.sprint_session_runner import (
            SprintSessionRunner,
            QuizAnswer,
            CheckpointRating,
        )

        store = get_memory()
        plan = store.get_sprint_plan(plan_id)

        if not plan:
            raise HTTPException(status_code=404, detail="Sprint plan not found")

        runner = SprintSessionRunner(memory_store=store)

        session = runner.load_today_session(plan_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        quiz_answers = [
            QuizAnswer(
                question_id=a.question_id,
                answer=a.answer,
                is_correct=a.is_correct,
                time_spent_seconds=a.time_spent_seconds,
            )
            for a in request.quiz_answers
        ]

        checkpoint_ratings = [
            CheckpointRating(
                topic=r.topic,
                understanding_rating=r.understanding_rating,
                confidence_rating=r.confidence_rating,
                notes=r.notes,
            )
            for r in request.checkpoint_ratings
        ]

        result = runner.complete_session(
            session=session,
            quiz_answers=quiz_answers,
            checkpoint_ratings=checkpoint_ratings,
            total_time_seconds=request.total_time_seconds,
        )

        store.update_sprint_day_status(plan_id, day_number, "completed", int(result.percentage))

        if result.percentage >= 70:
            store.update_sprint_plan_status(plan_id, "active")

        return {
            "result": result.to_dict(),
            "message": "Session completed successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to complete session {plan_id}/{day_number}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to complete session: {str(e)}",
        )


@router.get("/sprint/{plan_id}/catch-up")
async def get_catch_up_sessions(plan_id: str):
    """
    Get list of missed sessions that need to be caught up.

    Returns all pending days before today with their sessions.
    """
    try:
        from extensions.agents.sprint_session_runner import SprintSessionRunner

        store = get_memory()
        plan = store.get_sprint_plan(plan_id)

        if not plan:
            raise HTTPException(status_code=404, detail="Sprint plan not found")

        runner = SprintSessionRunner(memory_store=store)
        catch_up_sessions = runner.check_and_create_catch_up_sessions(plan_id)

        return {
            "sessions": [s.to_dict() for s in catch_up_sessions],
            "count": len(catch_up_sessions),
            "message": f"You have {len(catch_up_sessions)} missed session(s) to catch up on.",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get catch-up sessions for {plan_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get catch-up sessions: {str(e)}",
        )


@router.get("/sprint/{plan_id}/review-queue")
async def get_review_queue(plan_id: str, limit: int = 10):
    """
    Get the review queue for a sprint plan.

    Returns scheduled reviews based on weakness tracking.
    """
    try:
        store = get_memory()
        plan = store.get_sprint_plan(plan_id)

        if not plan:
            raise HTTPException(status_code=404, detail="Sprint plan not found")

        reviews = store.get_review_queue(limit=limit)

        sprint_reviews = [r.to_dict() for r in reviews if r.subject in plan.get("subjects", [])]

        return {
            "reviews": sprint_reviews,
            "count": len(sprint_reviews),
        }

    except Exception as e:
        logger.error(f"Failed to get review queue for {plan_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get review queue: {str(e)}",
        )


# =============================================================================
# Voice Command Endpoints (PID-33)
# =============================================================================


class VoiceCommandRequest(BaseModel):
    """Request to process a voice command."""

    transcript: str
    plan_id: Optional[str] = None
    day_number: Optional[int] = None
    current_phase: str = "overview"
    difficulty_level: str = "medium"
    tts_speed: float = 1.0
    is_paused: bool = False
    session_active: bool = False


class VoiceCommandResponse(BaseModel):
    """Response for a voice command."""

    success: bool
    action: str
    message: str
    command: dict
    context: dict
    phase: Optional[str] = None
    difficulty: Optional[str] = None
    tts_speed: Optional[float] = None


@router.post("/sprint/voice/command", response_model=VoiceCommandResponse)
async def process_voice_command(request: VoiceCommandRequest):
    """
    Process a voice command for sprint session control.

    Voice macros:
    - "start today" - Start today's session
    - "next checkpoint" - Move to next phase
    - "previous" - Go back to previous phase
    - "repeat" - Repeat current content
    - "slower" - Slow down speech
    - "faster" - Speed up speech
    - "harder" - Increase difficulty
    - "easier" - Decrease difficulty
    - "mark this" / "done" - Mark item complete
    - "skip" - Skip current item
    - "pause" - Pause session
    - "resume" - Continue session
    - "stop" / "end" - End session
    - "help" - Show available commands
    """
    try:
        from extensions.agents.sprint_session_runner import process_sprint_voice_command

        result = process_sprint_voice_command(
            transcript=request.transcript,
            plan_id=request.plan_id,
            day_number=request.day_number,
            current_phase=request.current_phase,
            difficulty_level=request.difficulty_level,
            tts_speed=request.tts_speed,
            is_paused=request.is_paused,
            session_active=request.session_active,
        )

        return VoiceCommandResponse(
            success=result["result"].get("success", False),
            action=result["result"].get("action", "unknown"),
            message=result["result"].get("message", ""),
            command=result["command"],
            context=result["context"],
            phase=result["context"].get("current_phase"),
            difficulty=result["context"].get("difficulty_level"),
            tts_speed=result["context"].get("tts_speed"),
        )

    except Exception as e:
        logger.error(f"Failed to process voice command: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process voice command: {str(e)}",
        )


@router.get("/sprint/voice/help")
async def get_voice_commands_help():
    """
    Get list of available voice commands for sprint sessions.

    Returns all supported voice macros and their descriptions.
    """
    return {
        "commands": {
            "start today": {
                "description": "Start today's sprint session",
                "alternatives": ["start now", "begin", "ready to start"],
            },
            "next checkpoint": {
                "description": "Move to the next phase of the session",
                "alternatives": ["continue", "proceed", "next phase", "next step"],
            },
            "previous": {
                "description": "Go back to the previous phase",
                "alternatives": ["go back", "back", "previous phase"],
            },
            "repeat": {
                "description": "Repeat the current content",
                "alternatives": ["say again", "again", "once more"],
            },
            "slower": {
                "description": "Slow down text-to-speech speed",
                "alternatives": ["slow down", "too fast"],
            },
            "faster": {
                "description": "Speed up text-to-speech",
                "alternatives": ["speed up", "too slow"],
            },
            "harder": {
                "description": "Increase difficulty level",
                "alternatives": ["more difficult", "challenge me", "tougher"],
            },
            "easier": {
                "description": "Decrease difficulty level",
                "alternatives": ["more simple", "simpler", "too hard"],
            },
            "mark this": {
                "description": "Mark current item as complete",
                "alternatives": ["done", "complete", "finished", "that's it"],
            },
            "skip": {
                "description": "Skip current item",
                "alternatives": ["skip this", "next one", "come back later"],
            },
            "pause": {
                "description": "Pause the session",
                "alternatives": ["wait", "hold on", "stop for a sec"],
            },
            "resume": {
                "description": "Resume a paused session",
                "alternatives": ["continue", "keep going", "start again"],
            },
            "stop": {
                "description": "End the session",
                "alternatives": ["end", "quit", "finish", "done for today"],
            },
            "help": {
                "description": "Show available voice commands",
                "alternatives": ["what can I say", "commands", "options"],
            },
        },
        "phases": ["overview", "checkpoint", "quiz", "complete"],
        "difficulty_levels": ["easy", "medium", "hard", "advanced"],
        "tts_speed_range": {"min": 0.5, "max": 2.0, "default": 1.0},
    }
