"""
Sprint Session Runner Agent
============================

Executes daily sprint sessions with:
- 10-minute overview generation
- 10-minute guided checkpoints
- 10-minute quiz with automatic weak-spot tracking
- Spaced repetition review scheduling (D+2, D+7)

Usage:
    from extensions.agents.sprint_session_runner import SprintSessionRunner, SessionResult

    runner = SprintSessionRunner()
    session = runner.load_today_session(plan_id)
    result = runner.complete_session(session, quiz_answers, checkpoint_ratings)
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class SessionPhase(str, Enum):
    """Phases of a daily sprint session."""

    OVERVIEW = "overview"
    CHECKPOINT = "checkpoint"
    QUIZ = "quiz"
    COMPLETE = "complete"


class QuizAnswer(BaseModel):
    """User's answer to a quiz question."""

    question_id: str
    answer: str
    is_correct: bool
    time_spent_seconds: int = 0


class CheckpointRating(BaseModel):
    """User's self-rating for a checkpoint."""

    topic: str
    understanding_rating: int  # 1-5
    confidence_rating: int  # 1-5
    notes: str = ""


@dataclass
class SessionOverview:
    """10-minute overview content for a session."""

    phase: str = SessionPhase.OVERVIEW
    title: str = ""
    summary: str = ""
    key_concepts: list[str] = field(default_factory=list)
    learning_objectives: list[str] = field(default_factory=list)
    estimated_time_minutes: int = 10
    resources: list[str] = field(default_factory=list)
    previous_review_notes: str = ""

    def to_dict(self) -> dict:
        return {
            "phase": self.phase.value if isinstance(self.phase, SessionPhase) else self.phase,
            "title": self.title,
            "summary": self.summary,
            "key_concepts": self.key_concepts,
            "learning_objectives": self.learning_objectives,
            "estimated_time_minutes": self.estimated_time_minutes,
            "resources": self.resources,
            "previous_review_notes": self.previous_review_notes,
        }


@dataclass
class SessionCheckpoint:
    """10-minute guided checkpoint activity."""

    phase: str = SessionPhase.CHECKPOINT
    title: str = ""
    instructions: str = ""
    questions: list[dict] = field(default_factory=list)
    estimated_time_minutes: int = 10
    guidance_tips: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "phase": self.phase.value if isinstance(self.phase, SessionPhase) else self.phase,
            "title": self.title,
            "instructions": self.instructions,
            "questions": self.questions,
            "estimated_time_minutes": self.estimated_time_minutes,
            "guidance_tips": self.guidance_tips,
        }


@dataclass
class SessionQuiz:
    """10-minute quiz for the session."""

    phase: str = SessionPhase.QUIZ
    title: str = ""
    questions: list[dict] = field(default_factory=list)
    total_marks: int = 0
    time_limit_minutes: int = 10
    passing_score: int = 70

    def to_dict(self) -> dict:
        return {
            "phase": self.phase.value if isinstance(self.phase, SessionPhase) else self.phase,
            "title": self.title,
            "questions": self.questions,
            "total_marks": self.total_marks,
            "time_limit_minutes": self.time_limit_minutes,
            "passing_score": self.passing_score,
        }


@dataclass
class SprintSession:
    """Complete daily sprint session with all phases."""

    plan_id: str
    day_number: int
    date: str
    subject: str
    topics_covered: list[str]
    overview: SessionOverview
    checkpoint: SessionCheckpoint
    quiz: SessionQuiz
    current_phase: str = SessionPhase.OVERVIEW
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    quiz_score: Optional[int] = None
    quiz_total: int = 0

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "day_number": self.day_number,
            "date": self.date,
            "subject": self.subject,
            "topics_covered": self.topics_covered,
            "overview": self.overview.to_dict(),
            "checkpoint": self.checkpoint.to_dict(),
            "quiz": self.quiz.to_dict(),
            "current_phase": self.current_phase,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "quiz_score": self.quiz_score,
            "quiz_total": self.quiz_total,
        }


@dataclass
class SessionResult:
    """Result of completing a sprint session."""

    success: bool
    plan_id: str
    day_number: int
    quiz_score: int
    quiz_total: int
    percentage: float
    weaknesses_found: list[dict] = field(default_factory=list)
    review_scheduled: list[dict] = field(default_factory=list)
    catch_up_suggestions: list[str] = field(default_factory=list)
    next_session_date: Optional[str] = None
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "plan_id": self.plan_id,
            "day_number": self.day_number,
            "quiz_score": self.quiz_score,
            "quiz_total": self.quiz_total,
            "percentage": self.percentage,
            "weaknesses_found": self.weaknesses_found,
            "review_scheduled": self.review_scheduled,
            "catch_up_suggestions": self.catch_up_suggestions,
            "next_session_date": self.next_session_date,
            "message": self.message,
        }


class SprintSessionRunner:
    """
    Executes daily sprint sessions with automatic weak-spot tracking.

    Features:
    - Generates 10-minute overviews from topic content
    - Creates guided checkpoint activities
    - Runs quizzes with automatic grading
    - Tracks weaknesses and schedules reviews (D+2, D+7)
    - Handles catch-up when days are missed
    """

    def __init__(self, memory_store=None):
        self.memory_store = memory_store
        self._topic_content_cache: dict[str, dict] = {}

    def load_today_session(
        self, plan_id: str, target_date: Optional[str] = None
    ) -> Optional[SprintSession]:
        """
        Load today's session for a given plan.

        Args:
            plan_id: The sprint plan ID
            target_date: Specific date to load (defaults to today)

        Returns:
            SprintSession if available, None if no session for today
        """
        from extensions.memory.store import MemoryStore

        if self.memory_store is None:
            self.memory_store = MemoryStore()

        with self.memory_store._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            target = target_date or datetime.now().isoformat()[:10]

            cursor.execute(
                """
                SELECT * FROM sprint_plans WHERE id = ?
            """,
                (plan_id,),
            )
            plan_row = cursor.fetchone()
            if not plan_row:
                return None

            plan_data = json.loads(plan_row["plan_data"])

            cursor.execute(
                """
                SELECT * FROM sprint_days
                WHERE plan_id = ? AND date = ?
            """,
                (plan_id, target),
            )
            day_row = cursor.fetchone()

            if not day_row:
                return None

            day_data = json.loads(day_row["activities"]) if day_row["activities"] else []
            quiz_data = json.loads(day_row["quiz"]) if day_row["quiz"] else None

            overview = self._generate_overview(plan_data, day_data, day_row)
            checkpoint = self._generate_checkpoint(plan_data, day_data, day_row)
            quiz = self._generate_quiz(quiz_data, day_row)

            return SprintSession(
                plan_id=plan_id,
                day_number=day_row["day_number"],
                date=day_row["date"],
                subject=day_row["subject"],
                topics_covered=json.loads(day_row["topics_covered"]),
                overview=overview,
                checkpoint=checkpoint,
                quiz=quiz,
            )

    def load_next_pending_session(self, plan_id: str) -> Optional[SprintSession]:
        """
        Load the next pending session (handles catch-up).

        Args:
            plan_id: The sprint plan ID

        Returns:
            SprintSession for the next incomplete day
        """
        from extensions.memory.store import MemoryStore

        if self.memory_store is None:
            self.memory_store = MemoryStore()

        with self.memory_store._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM sprint_days
                WHERE plan_id = ? AND completion_status != 'completed'
                ORDER BY day_number ASC
                LIMIT 1
            """,
                (plan_id,),
            )
            day_row = cursor.fetchone()

            if not day_row:
                return None

            cursor.execute(
                """
                SELECT * FROM sprint_plans WHERE id = ?
            """,
                (plan_id,),
            )
            plan_row = cursor.fetchone()
            if not plan_row:
                return None

            plan_data = json.loads(plan_row["plan_data"])
            day_data = json.loads(day_row["activities"]) if day_row["activities"] else []
            quiz_data = json.loads(day_row["quiz"]) if day_row["quiz"] else None

            overview = self._generate_overview(plan_data, day_data, day_row, catch_up_mode=True)
            checkpoint = self._generate_checkpoint(plan_data, day_data, day_row)
            quiz = self._generate_quiz(quiz_data, day_row)

            return SprintSession(
                plan_id=plan_id,
                day_number=day_row["day_number"],
                date=day_row["date"],
                subject=day_row["subject"],
                topics_covered=json.loads(day_row["topics_covered"]),
                overview=overview,
                checkpoint=checkpoint,
                quiz=quiz,
            )

    def _generate_overview(
        self, plan_data: dict, activities: list, day_row: sqlite3.Row, catch_up_mode: bool = False
    ) -> SessionOverview:
        """Generate 10-minute overview for the session."""
        topics = json.loads(day_row["topics_covered"])
        subject = day_row["subject"]

        topic_names = [t.replace("_", " ").title() for t in topics]

        overview = SessionOverview(
            title=f"Day {day_row['day_number']}: {', '.join(topic_names)}",
            summary=f"Today's session covers {len(topics)} key topics in {subject.title()}. "
            f"Expected time: 30 minutes total.",
            key_concepts=topic_names,
            learning_objectives=[
                f"Understand the core concepts of {topic_names[0]}"
                if topic_names
                else "Review main concepts",
                f"Apply key formulas and methods for {topic_names[0]}"
                if topic_names
                else "Practice problems",
                f"Complete checkpoint self-assessment",
                f"Pass the daily quiz with 70% or higher",
            ],
            resources=["See knowledge base for detailed notes", "Practice problems in notebook"],
            previous_review_notes="",
        )

        if catch_up_mode:
            overview.previous_review_notes = (
                "⚠️ You're catching up on this session. Focus on the key concepts first, "
                "then proceed to the quiz. You can always retry later."
            )

        return overview

    def _generate_checkpoint(
        self, plan_data: dict, activities: list, day_row: sqlite3.Row
    ) -> SessionCheckpoint:
        """Generate 10-minute guided checkpoint."""
        topics = json.loads(day_row["topics_covered"])

        questions = []
        for i, topic in enumerate(topics):
            questions.append(
                {
                    "id": f"cp_{day_row['day_number']}_{i}",
                    "topic": topic,
                    "question": f"Rate your understanding of {topic.replace('_', ' ').title()} (1-5):",
                    "type": "self-assessment",
                    "prompt": f"Before we test your knowledge, rate how well you understand {topic.replace('_', ' ').title()}. "
                    "1 = Very uncertain, 5 = Very confident",
                }
            )
            questions.append(
                {
                    "id": f"ar_{day_row['day_number']}_{i}",
                    "topic": topic,
                    "question": f"Without looking at notes, explain the main concept of {topic.replace('_', ' ').title()}:",
                    "type": "active_recall",
                    "prompt": f"Write 2-3 sentences explaining {topic.replace('_', ' ').title()} in your own words. "
                    "Don't worry about perfection - this is for self-assessment.",
                }
            )

        return SessionCheckpoint(
            title=f"Day {day_row['day_number']} Checkpoint",
            instructions="Complete the self-assessment and active recall questions below. "
            "Be honest with yourself - this helps identify areas that need more practice.",
            questions=questions,
            guidance_tips=[
                "Be honest in your self-assessment",
                "Try active recall before checking answers",
                "Note any topics you found difficult",
            ],
        )

    def _generate_quiz(self, quiz_data: Optional[dict], day_row: sqlite3.Row) -> SessionQuiz:
        """Generate 10-minute quiz."""
        if not quiz_data:
            return SessionQuiz(
                title=f"Day {day_row['day_number']} Quiz",
                questions=[],
                total_marks=0,
            )

        return SessionQuiz(
            title=f"Day {day_row['day_number']} Quiz",
            questions=quiz_data.get("questions", []),
            total_marks=quiz_data.get("total_marks", 0),
            time_limit_minutes=10,
            passing_score=70,
        )

    def complete_session(
        self,
        session: SprintSession,
        quiz_answers: list[QuizAnswer],
        checkpoint_ratings: list[CheckpointRating],
        total_time_seconds: int = 1800,
    ) -> SessionResult:
        """
        Complete a sprint session and update tracking.

        Args:
            session: The completed session
            quiz_answers: User's answers to quiz questions
            checkpoint_ratings: User's self-ratings for checkpoints
            total_time_seconds: Total time spent on session

        Returns:
            SessionResult with quiz score, weaknesses, and review schedule
        """
        from extensions.memory.store import MemoryStore, ErrorType, ConfidenceLevel

        if self.memory_store is None:
            self.memory_store = MemoryStore()

        correct_count = sum(1 for a in quiz_answers if a.is_correct)
        total_questions = len(quiz_answers)
        percentage = (correct_count / total_questions * 100) if total_questions > 0 else 0

        weaknesses = []
        review_scheduled = []

        if quiz_answers:
            for answer in quiz_answers:
                if not answer.is_correct:
                    weakness = self.memory_store.add_weakness(
                        subject=session.subject,
                        topic=answer.question_id.split("_")[0]
                        if "_" in answer.question_id
                        else session.topics_covered[0]
                        if session.topics_covered
                        else "",
                        error_type=ErrorType.CONCEPTUAL_MISUNDERSTANDING,
                        description=f"Incorrect answer to: {answer.question_id}",
                        question_id=answer.question_id,
                        user_answer=answer.answer,
                        correct_answer="",  # Would need to store correct answer
                        confidence=ConfidenceLevel.LOW,
                    )
                    weaknesses.append(
                        {
                            "id": weakness,
                            "topic": session.topics_covered[0] if session.topics_covered else "",
                            "question_id": answer.question_id,
                        }
                    )

        for rating in checkpoint_ratings:
            if rating.understanding_rating <= 2:
                weakness = self.memory_store.add_weakness(
                    subject=session.subject,
                    topic=rating.topic,
                    error_type=ErrorType.COMPREHENSION_GAP,
                    description=f"Low understanding rating ({rating.understanding_rating}/5): {rating.notes}",
                    confidence=ConfidenceLevel(
                        ["very_low", "low", "medium", "high", "very_high"][
                            rating.understanding_rating - 1
                        ]
                    ),
                )

                if weakness:
                    d2_review = self.memory_store.add_to_review_queue(
                        weakness_id=weakness,
                        subject=session.subject,
                        topic=rating.topic,
                        priority=1.0 - (rating.understanding_rating / 5),
                        hours=48,
                    )
                    d7_review = self.memory_store.add_to_review_queue(
                        weakness_id=weakness,
                        subject=session.subject,
                        topic=rating.topic,
                        priority=0.8 - (rating.understanding_rating / 6),
                        hours=168,
                    )
                    review_scheduled.append(
                        {
                            "id": weakness,
                            "topic": rating.topic,
                            "d2_review_id": d2_review,
                            "d7_review_id": d7_review,
                        }
                    )

        catch_up_suggestions = []
        if percentage < 70:
            catch_up_suggestions = [
                "Review the key concepts from today's session",
                "Focus on topics where you scored lowest",
                "Try the practice problems in the notebook",
                "Schedule a 15-minute review before the next session",
            ]

        next_date = self._get_next_session_date(session)

        return SessionResult(
            success=True,
            plan_id=session.plan_id,
            day_number=session.day_number,
            quiz_score=correct_count,
            quiz_total=total_questions,
            percentage=percentage,
            weaknesses_found=weaknesses,
            review_scheduled=review_scheduled,
            catch_up_suggestions=catch_up_suggestions,
            next_session_date=next_date,
            message=self._generate_completion_message(percentage, len(weaknesses)),
        )

    def _get_next_session_date(self, session: SprintSession) -> Optional[str]:
        """Get the date for the next session."""
        from extensions.memory.store import MemoryStore

        if self.memory_store is None:
            self.memory_store = MemoryStore()

        with self.memory_store._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT date FROM sprint_days
                WHERE plan_id = ? AND day_number > ?
                ORDER BY day_number ASC
                LIMIT 1
            """,
                (session.plan_id, session.day_number),
            )
            row = cursor.fetchone()
            return row["date"] if row else None

    def _generate_completion_message(self, percentage: float, weakness_count: int) -> str:
        """Generate appropriate completion message based on performance."""
        if percentage >= 90:
            return "Excellent work! You've mastered today's material. Keep up the great progress!"
        elif percentage >= 70:
            return "Good job! You've passed today's quiz. Review the areas you missed for even better retention."
        elif percentage >= 50:
            return "You completed today's session, but there's room for improvement. Focus on the suggested review topics."
        else:
            return "Today's quiz was challenging. Don't worry - the review schedule will help reinforce these concepts."

    def update_day_status(
        self, plan_id: str, day_number: int, status: str, quiz_score: Optional[int] = None
    ) -> bool:
        """Update the completion status of a sprint day."""
        from extensions.memory.store import MemoryStore

        if self.memory_store is None:
            self.memory_store = MemoryStore()

        with self.memory_store._get_connection() as conn:
            cursor = conn.cursor()

            if quiz_score is not None:
                cursor.execute(
                    """
                    UPDATE sprint_days
                    SET completion_status = ?, notes = ?
                    WHERE plan_id = ? AND day_number = ?
                """,
                    (status, f"Quiz score: {quiz_score}%", plan_id, day_number),
                )
            else:
                cursor.execute(
                    """
                    UPDATE sprint_days
                    SET completion_status = ?
                    WHERE plan_id = ? AND day_number = ?
                """,
                    (status, plan_id, day_number),
                )

            conn.commit()
            return cursor.rowcount > 0

    def check_and_create_catch_up_sessions(self, plan_id: str) -> list[SprintSession]:
        """
        Check for missed days and create catch-up sessions.

        Returns:
            List of catch-up sessions to complete
        """
        from extensions.memory.store import MemoryStore
        from datetime import datetime

        if self.memory_store is None:
            self.memory_store = MemoryStore()

        with self.memory_store._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            today = datetime.now().isoformat()[:10]

            cursor.execute(
                """
                SELECT * FROM sprint_days
                WHERE plan_id = ? AND date < ? AND completion_status != 'completed'
                ORDER BY date ASC
            """,
                (plan_id, today),
            )
            missed_days = cursor.fetchall()

            catch_up_sessions = []
            for day_row in missed_days:
                session = self.load_next_pending_session(plan_id)
                if session:
                    catch_up_sessions.append(session)

            return catch_up_sessions


import sqlite3

__all__ = [
    "SprintSessionRunner",
    "SprintSession",
    "SessionResult",
    "SessionOverview",
    "SessionCheckpoint",
    "SessionQuiz",
    "SessionPhase",
    "QuizAnswer",
    "CheckpointRating",
]


# =============================================================================
# Voice Controller Integration (PID-33)
# =============================================================================


def create_voice_controller():
    """Create a voice controller for sprint sessions."""
    from extensions.agents.voice_controller import (
        VoiceControllerAgent,
        SprintSessionContext,
        SprintVoiceAction,
    )

    controller = VoiceControllerAgent()

    def handle_start_today(ctx: SprintSessionContext) -> dict:
        return {
            "success": True,
            "action": "start_today",
            "message": "Starting today's sprint session. Let's begin with the overview.",
            "phase": "overview",
            "difficulty": ctx.difficulty_level,
            "tts_speed": ctx.tts_speed,
        }

    def handle_next_phase(ctx: SprintSessionContext) -> dict:
        phases = ["overview", "checkpoint", "quiz", "complete"]
        if ctx.current_phase in phases:
            current_idx = phases.index(ctx.current_phase)
            next_idx = min(current_idx + 1, len(phases) - 1)
            next_phase = phases[next_idx]
            ctx.current_phase = next_phase
            messages = {
                "overview": "Moving to the checkpoint phase. Time to test your understanding.",
                "checkpoint": "Moving to the quiz. Ready to test your knowledge?",
                "quiz": "Great work completing the quiz! Let's see your results.",
                "complete": "Session complete! Great job today.",
            }
            return {
                "success": True,
                "action": "next_phase",
                "message": messages.get(next_phase, "Moving to next phase."),
                "phase": next_phase,
                "difficulty": ctx.difficulty_level,
                "tts_speed": ctx.tts_speed,
            }
        return {"success": False, "action": "next_phase", "message": "Invalid phase"}

    def handle_previous_phase(ctx: SprintSessionContext) -> dict:
        phases = ["overview", "checkpoint", "quiz", "complete"]
        if ctx.current_phase in phases:
            current_idx = phases.index(ctx.current_phase)
            prev_idx = max(current_idx - 1, 0)
            prev_phase = phases[prev_idx]
            ctx.current_phase = prev_phase
            messages = {
                "overview": "Going back to the overview.",
                "checkpoint": "Going back to the checkpoint phase.",
                "quiz": "Going back to the quiz.",
                "complete": "Let's continue with the session.",
            }
            return {
                "success": True,
                "action": "previous_phase",
                "message": messages.get(prev_phase, "Going back."),
                "phase": prev_phase,
                "difficulty": ctx.difficulty_level,
                "tts_speed": ctx.tts_speed,
            }
        return {"success": False, "action": "previous_phase", "message": "Invalid phase"}

    def handle_pause(ctx: SprintSessionContext) -> dict:
        ctx.is_paused = True
        return {
            "success": True,
            "action": "pause",
            "message": "Session paused. Say 'resume' or 'continue' when you're ready.",
            "is_paused": True,
        }

    def handle_resume(ctx: SprintSessionContext) -> dict:
        ctx.is_paused = False
        ctx.session_active = True
        return {
            "success": True,
            "action": "resume",
            "message": "Resuming the session.",
            "is_paused": False,
            "phase": ctx.current_phase,
        }

    def handle_stop(ctx: SprintSessionContext) -> dict:
        ctx.session_active = False
        ctx.is_paused = False
        return {
            "success": True,
            "action": "stop",
            "message": "Session ended. Great work today!",
            "session_active": False,
        }

    def handle_repeat(ctx: SprintSessionContext) -> dict:
        repeat_messages = {
            "overview": "I'll repeat the overview. Today's session covers key concepts and learning objectives.",
            "checkpoint": "Let's go through the checkpoint questions again. Remember to be honest with your self-assessment.",
            "quiz": "I'll repeat the quiz questions. Take your time and answer carefully.",
            "complete": "Here are your session results again.",
        }
        return {
            "success": True,
            "action": "repeat",
            "message": repeat_messages.get(ctx.current_phase, "Repeating current content."),
            "phase": ctx.current_phase,
            "tts_speed": ctx.tts_speed,
        }

    def handle_slower(ctx: SprintSessionContext) -> dict:
        ctx.tts_speed = max(0.5, ctx.tts_speed - 0.1)
        return {
            "success": True,
            "action": "slower",
            "message": f" slowing down. New speed: {ctx.tts_speed:.1f}x",
            "tts_speed": ctx.tts_speed,
        }

    def handle_faster(ctx: SprintSessionContext) -> dict:
        ctx.tts_speed = min(2.0, ctx.tts_speed + 0.1)
        return {
            "success": True,
            "action": "faster",
            "message": f"Speeding up. New speed: {ctx.tts_speed:.1f}x",
            "tts_speed": ctx.tts_speed,
        }

    def handle_harder(ctx: SprintSessionContext) -> dict:
        difficulty_levels = ["easy", "medium", "hard", "advanced"]
        if ctx.difficulty_level in difficulty_levels:
            current_idx = difficulty_levels.index(ctx.difficulty_level)
            if current_idx < len(difficulty_levels) - 1:
                ctx.difficulty_level = difficulty_levels[current_idx + 1]
        return {
            "success": True,
            "action": "harder",
            "message": f" difficulty. Moving to {ctx.difficulty_level} level.",
            "difficulty": ctx.difficulty_level,
        }

    def handle_easier(ctx: SprintSessionContext) -> dict:
        difficulty_levels = ["easy", "medium", "hard", "advanced"]
        if ctx.difficulty_level in difficulty_levels:
            current_idx = difficulty_levels.index(ctx.difficulty_level)
            if current_idx > 0:
                ctx.difficulty_level = difficulty_levels[current_idx - 1]
        return {
            "success": True,
            "action": "easier",
            "message": f"Making it easier. Moving to {ctx.difficulty_level} level.",
            "difficulty": ctx.difficulty_level,
        }

    def handle_mark_complete(ctx: SprintSessionContext) -> dict:
        return {
            "success": True,
            "action": "mark_complete",
            "message": "Marked as complete. Moving to next item.",
            "phase": ctx.current_phase,
        }

    def handle_skip(ctx: SprintSessionContext) -> dict:
        return {
            "success": True,
            "action": "skip",
            "message": "Skipping this item.",
            "phase": ctx.current_phase,
        }

    def handle_help(ctx: SprintSessionContext) -> dict:
        return {
            "success": True,
            "action": "help",
            "message": "Voice commands available during sprint sessions:",
            "commands": {
                "start today": "Start today's session",
                "next checkpoint": "Move to next phase",
                "previous": "Go back to previous phase",
                "repeat": "Repeat current content",
                "slower": "Slow down speech",
                "faster": "Speed up speech",
                "harder": "Increase difficulty",
                "easier": "Decrease difficulty",
                "mark this / done": "Mark item complete",
                "skip": "Skip current item",
                "pause": "Pause session",
                "resume": "Continue session",
                "stop / end": "End session",
                "help": "Show this help",
            },
        }

    controller.register_handler(SprintVoiceAction.START_TODAY, handle_start_today)
    controller.register_handler(SprintVoiceAction.NEXT_PHASE, handle_next_phase)
    controller.register_handler(SprintVoiceAction.PREVIOUS_PHASE, handle_previous_phase)
    controller.register_handler(SprintVoiceAction.REPEAT, handle_repeat)
    controller.register_handler(SprintVoiceAction.SLOWER, handle_slower)
    controller.register_handler(SprintVoiceAction.FASTER, handle_faster)
    controller.register_handler(SprintVoiceAction.HARDER, handle_harder)
    controller.register_handler(SprintVoiceAction.EASIER, handle_easier)
    controller.register_handler(SprintVoiceAction.MARK_COMPLETE, handle_mark_complete)
    controller.register_handler(SprintVoiceAction.SKIP_ITEM, handle_skip)
    controller.register_handler(SprintVoiceAction.PAUSE, handle_pause)
    controller.register_handler(SprintVoiceAction.RESUME, handle_resume)
    controller.register_handler(SprintVoiceAction.STOP, handle_stop)
    controller.register_handler(SprintVoiceAction.HELP, handle_help)

    return controller


def process_sprint_voice_command(
    transcript: str,
    plan_id: Optional[str] = None,
    day_number: Optional[int] = None,
    current_phase: str = "overview",
    difficulty_level: str = "medium",
    tts_speed: float = 1.0,
    is_paused: bool = False,
    session_active: bool = False,
) -> dict:
    """
    Process a voice command for sprint session control.

    Args:
        transcript: The transcribed voice text
        plan_id: Current sprint plan ID
        day_number: Current day number
        current_phase: Current session phase
        difficulty_level: Current difficulty level
        tts_speed: Current TTS speed
        is_paused: Whether session is paused
        session_active: Whether session is active

    Returns:
        Result dictionary with action result and updated context
    """
    controller = create_voice_controller()

    context = SprintSessionContext(
        plan_id=plan_id,
        day_number=day_number,
        current_phase=current_phase,
        difficulty_level=difficulty_level,
        tts_speed=tts_speed,
        is_paused=is_paused,
        session_active=session_active,
    )

    controller.set_context(context)

    match = controller.process_command(transcript)
    result = controller.execute_action(match, context)

    return {
        "result": result,
        "context": {
            "plan_id": context.plan_id,
            "day_number": context.day_number,
            "current_phase": context.current_phase,
            "difficulty_level": context.difficulty_level,
            "tts_speed": context.tts_speed,
            "is_paused": context.is_paused,
            "session_active": context.session_active,
        },
        "command": match.to_dict(),
    }
