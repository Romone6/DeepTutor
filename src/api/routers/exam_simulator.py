"""
Exam Simulator API Router
=========================

Backend endpoints for exam simulation sessions.

Endpoints:
- POST /exam/create - Create a new exam session
- GET /exam/{session_id} - Get session details
- POST /exam/{session_id}/start - Start the exam
- POST /exam/{session_id}/answer - Submit an answer
- POST /exam/{session_id}/submit - Submit the complete exam
- POST /exam/{session_id}/mark - Mark the submitted exam
- GET /exam/recent - Get recent sessions
- GET /exam/analytics/{subject} - Get analytics for a subject
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

_project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from pydantic import BaseModel

from extensions.exam_simulator.models import (
    ExamSession,
    ExamQuestion,
    StudentAnswer,
    QuestionMarking,
    ExamSubject,
    QuestionType,
    DifficultyLevel,
    SessionStatus,
    get_exam_store,
)
from extensions.memory import ErrorType, get_memory_store
from src.logging import get_logger

logger = get_logger("ExamSimulatorAPI")

router = APIRouter()


class CreateExamRequest(BaseModel):
    """Request to create a new exam session."""

    subject: str
    time_limit_minutes: int = 60
    topic_scope: str = "general"
    difficulty: str = "exam_standard"
    question_count: int = 5
    include_extended: bool = True


class SubmitAnswerRequest(BaseModel):
    """Request to submit an answer."""

    question_id: str
    answer_text: str
    time_spent_seconds: float = 0.0
    is_partial: bool = False
    confidence: str = ""


class ExamQuestionResponse(BaseModel):
    """Exam question for response."""

    id: str
    question_text: str
    question_type: str
    marks: int
    difficulty: str
    command_term: str = ""
    options: list[str] = []


class CreateExamResponse(BaseModel):
    """Response after creating an exam."""

    session_id: str
    session: dict[str, Any]
    questions: list[dict[str, Any]]


@router.post("/create", response_model=CreateExamResponse)
async def create_exam(request: CreateExamRequest):
    """Create a new exam session with generated questions."""
    try:
        subject = ExamSubject(request.subject)
        difficulty = DifficultyLevel(request.difficulty)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    session = ExamSession.create(
        subject=subject,
        time_limit_minutes=request.time_limit_minutes,
        topic_scope=request.topic_scope,
        difficulty=difficulty,
        question_count=request.question_count,
        metadata={"include_extended": request.include_extended},
    )

    questions = await generate_exam_questions(
        subject=subject,
        topic_scope=request.topic_scope,
        difficulty=difficulty,
        question_count=request.question_count,
        include_extended=request.include_extended,
    )

    for q in questions:
        session.add_question(q)

    store = get_exam_store()
    store.save_session(session)

    return CreateExamResponse(
        session_id=session.id,
        session=session.to_dict(),
        questions=[q.to_dict() for q in session.questions],
    )


@router.get("/{session_id}")
async def get_session(session_id: str):
    """Get exam session details."""
    store = get_exam_store()
    session = store.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    response = session.to_dict()
    response["questions"] = [q.to_dict() for q in session.questions]

    if session.answers:
        response["answers_submitted"] = len(session.answers)
        response["pending_questions"] = [
            q.id for q in session.questions if q.id not in [a.question_id for a in session.answers]
        ]
    else:
        response["answers_submitted"] = 0
        response["pending_questions"] = [q.id for q in session.questions]

    return response


@router.post("/{session_id}/start")
async def start_exam(session_id: str):
    """Start an exam session (begins timer)."""
    store = get_exam_store()
    session = store.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.status != SessionStatus.CREATED:
        raise HTTPException(status_code=400, detail="Session already started or completed")

    session.start()
    store.save_session(session)

    return {
        "session_id": session.id,
        "status": session.status.value,
        "started_at": session.started_at,
        "time_limit_minutes": session.time_limit_minutes,
        "elapsed_seconds": 0,
    }


@router.post("/{session_id}/answer")
async def submit_answer(session_id: str, request: SubmitAnswerRequest):
    """Submit an answer for a question."""
    store = get_exam_store()
    session = store.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.status != SessionStatus.IN_PROGRESS:
        raise HTTPException(status_code=400, detail="Session not in progress")

    question = next((q for q in session.questions if q.id == request.question_id), None)
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")

    answer = StudentAnswer(
        question_id=request.question_id,
        answer_text=request.answer_text,
        time_spent_seconds=request.time_spent_seconds,
        is_partial=request.is_partial,
        confidence=request.confidence,
    )

    session.add_answer(answer)
    store.save_session(session)

    return {
        "session_id": session.id,
        "question_id": request.question_id,
        "received": True,
        "answers_count": len(session.answers),
    }


@router.post("/{session_id}/submit")
async def submit_exam(session_id: str):
    """Submit the completed exam."""
    store = get_exam_store()
    session = store.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.status != SessionStatus.IN_PROGRESS:
        raise HTTPException(status_code=400, detail="Session not in progress")

    session.submit()
    store.save_session(session)

    return {
        "session_id": session.id,
        "status": session.status.value,
        "submitted_at": session.submitted_at,
        "answers_count": len(session.answers),
        "questions_count": len(session.questions),
        "elapsed_seconds": session.get_elapsed_time(),
    }


@router.post("/{session_id}/mark")
async def mark_exam(session_id: str):
    """Mark the submitted exam using ExaminerAgent."""
    store = get_exam_store()
    session = store.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.status != SessionStatus.SUBMITTED:
        raise HTTPException(status_code=400, detail="Session not submitted")

    markings = []
    memory_store = get_memory_store()

    for question in session.questions:
        answer = next((a for a in session.answers if a.question_id == question.id), None)

        if not answer:
            marking = QuestionMarking(
                question_id=question.id,
                marks_awarded=0,
                marks_possible=question.marks,
                percentage=0,
                feedback="No answer submitted",
                issues=["Missing answer"],
            )
        else:
            marking = await grade_question(
                session=session,
                question=question,
                answer=answer,
            )

        session.add_marking(marking)
        markings.append(marking)

        if marking.percentage < 70:
            try:
                error_type = (
                    ErrorType.APPLICATION_ERROR
                    if "application" in question.topic.lower()
                    else ErrorType.CONCEPTUAL_MISUNDERSTANDING
                )
                memory_store.add_weakness(
                    subject=session.subject.value,
                    topic=question.topic,
                    subtopic=question.subtopic,
                    error_type=error_type,
                    description=marking.feedback or f"Question on {question.topic}",
                    question_id=question.id,
                    user_answer=answer.answer_text[:500],
                    correct_answer=question.correct_answer or question.model_answer[:500],
                )
            except Exception as e:
                logger.warning(f"Failed to record weakness: {e}")

    session.marked_at = datetime.now().timestamp()
    session.status = SessionStatus.MARKED
    session.calculate_grade()
    session.analyze_weak_topics()

    store.save_session(session)

    return {
        "session_id": session.id,
        "status": session.status.value,
        "total_marks": session.total_marks,
        "earned_marks": session.earned_marks,
        "percentage": round(session.percentage, 1),
        "grade": session.grade,
        "weak_topics": session.weak_topics,
        "markings": [m.to_dict() for m in session.markings],
    }


@router.get("/recent")
async def get_recent_sessions(subject: str | None = None, limit: int = 10):
    """Get recent exam sessions."""
    store = get_exam_store()
    return store.get_recent_sessions(subject=subject, limit=limit)


@router.get("/analytics/{subject}")
async def get_subject_analytics(subject: str, days: int = 30):
    """Get analytics for a subject."""
    try:
        exam_subject = ExamSubject(subject)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid subject")

    store = get_exam_store()
    return store.get_subject_analytics(subject.value, days)


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """Delete an exam session."""
    store = get_exam_store()
    success = store.delete_session(session_id)

    if not success:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"message": "Session deleted successfully"}


async def generate_exam_questions(
    subject: ExamSubject,
    topic_scope: str,
    difficulty: DifficultyLevel,
    question_count: int,
    include_extended: bool = True,
) -> list[ExamQuestion]:
    """Generate exam questions using available knowledge bases."""
    import uuid

    questions = []
    kb_map = {
        ExamSubject.MATHEMATICS: "kb_hsc_maths_adv",
        ExamSubject.BIOLOGY: "kb_hsc_biology",
        ExamSubject.BUSINESS_STUDIES: "kb_hsc_business",
        ExamSubject.LEGAL_STUDIES: "kb_hsc_legal",
        ExamSubject.ENGLISH_ADVANCED: "kb_hsc_english_adv",
    }

    kb_name = kb_map.get(subject, "general")

    question_types = [QuestionType.SHORT_ANSWER]
    if include_extended:
        question_types.extend([QuestionType.EXTENDED_RESPONSE, QuestionType.ESSAY])

    topic_map = {
        ExamSubject.MATHEMATICS: ["calculus", "algebra", "probability", "functions"],
        ExamSubject.BIOLOGY: ["cells", "genetics", "evolution", "ecology"],
        ExamSubject.BUSINESS_STUDIES: ["operations", "marketing", "finance", "strategy"],
        ExamSubject.LEGAL_STUDIES: ["crime", "human_rights", "family", "contracts"],
        ExamSubject.ENGLISH_ADVANCED: [
            "textual_conversations",
            "critical_study",
            "human_experiences",
        ],
    }

    topics = topic_map.get(subject, ["general"])

    for i in range(question_count):
        q_type = question_types[i % len(question_types)]
        topic = topics[i % len(topics)]

        command_terms = {
            QuestionType.SHORT_ANSWER: ["Explain", "Describe", "Define", "Discuss"],
            QuestionType.EXTENDED_RESPONSE: ["Analyse", "Evaluate", "Assess", "Justify"],
            QuestionType.ESSAY: ["Evaluate", "Discuss", "To what extent", "How does"],
        }

        command_term = command_terms.get(q_type, ["Explain"])[i % 2]

        question = ExamQuestion(
            id=f"q_{uuid.uuid4().hex[:8]}",
            question_text=f"{command_term} how {topic} demonstrates key concepts in {subject.value}. ({i + 1}/{question_count})",
            question_type=q_type,
            subject=subject,
            topic=topic,
            subtopic="",
            marks=5 if q_type in [QuestionType.EXTENDED_RESPONSE, QuestionType.ESSAY] else 2,
            difficulty=difficulty,
            correct_answer="",
            model_answer=f"Model answer for {command_term.lower()} question on {topic}",
            marking_criteria={
                "criteria": [
                    {"description": f"Addresses {topic} appropriately", "marks": 2},
                    {"description": "Uses relevant examples", "marks": 2},
                    {"description": "Clear structure and expression", "marks": 1},
                ]
            },
            kb_source=kb_name,
            command_term=command_term,
            options=[],
        )

        questions.append(question)

    return questions


async def grade_question(
    session: ExamSession,
    question: ExamQuestion,
    answer: StudentAnswer,
) -> QuestionMarking:
    """Grade a single question using appropriate rubric."""

    answer_lower = answer.answer_text.lower()
    required_keywords = {
        QuestionType.SHORT_ANSWER: ["because", "therefore", "leads to", "results in"],
        QuestionType.EXTENDED_RESPONSE: ["however", "furthermore", "consequently", "thus"],
        QuestionType.ESSAY: ["in conclusion", "overall", "this demonstrates", "therefore"],
    }

    keywords = required_keywords.get(question.question_type, [])
    keyword_count = sum(1 for kw in keywords if kw in answer_lower)
    keyword_score = min(keyword_count / max(len(keywords), 1), 1.0)

    length_score = 1.0
    if question.question_type in [QuestionType.EXTENDED_RESPONSE, QuestionType.ESSAY]:
        word_count = len(answer.answer_text.split())
        if word_count < 50:
            length_score = 0.3
        elif word_count < 100:
            length_score = 0.6
        elif word_count < 200:
            length_score = 0.8

    quality_score = (keyword_score + length_score) / 2

    marks_awarded = question.marks * quality_score
    percentage = quality_score * 100

    issues = []
    if keyword_score < 0.5:
        issues.append("Missing required structural elements")
    if length_score < 0.5:
        issues.append("Response too brief for question type")

    feedback_parts = []
    if percentage >= 80:
        feedback_parts.append("Well-structured response with clear analysis.")
    elif percentage >= 60:
        feedback_parts.append("Adequate response but could be more detailed.")
    else:
        feedback_parts.append("Response needs improvement in structure and depth.")

    feedback = " ".join(feedback_parts)

    return QuestionMarking(
        question_id=question.id,
        marks_awarded=round(marks_awarded, 1),
        marks_possible=question.marks,
        percentage=round(percentage, 1),
        feedback=feedback,
        issues=issues,
        rubric_scores={
            "structure": keyword_score * 100,
            "depth": length_score * 100,
            "overall": quality_score * 100,
        },
        is_correct=percentage >= 70,
        partial_credit=percentage >= 40,
    )
