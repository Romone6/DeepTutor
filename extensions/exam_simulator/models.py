"""
Exam Simulator Models and Storage
=================================

Data models and persistence for exam simulation sessions.

Usage:
    from extensions.exam_simulator.models import (
        ExamSession,
        ExamQuestion,
        ExamResult,
        ExamSessionStore,
    )

    session = ExamSession.create(...)
    session_store.save(session)
"""

import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class ExamSubject(str, Enum):
    """Supported exam subjects."""

    MATHEMATICS = "mathematics"
    BIOLOGY = "biology"
    BUSINESS_STUDIES = "business_studies"
    LEGAL_STUDIES = "legal_studies"
    ENGLISH_ADVANCED = "english_advanced"


class QuestionType(str, Enum):
    """Types of exam questions."""

    SHORT_ANSWER = "short_answer"
    MULTIPLE_CHOICE = "multiple_choice"
    EXTENDED_RESPONSE = "extended_response"
    CALCULATION = "calculation"
    ESSAY = "essay"


class DifficultyLevel(str, Enum):
    """Question difficulty levels."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXAM_STANDARD = "exam_standard"


class SessionStatus(str, Enum):
    """Exam session status."""

    CREATED = "created"
    IN_PROGRESS = "in_progress"
    SUBMITTED = "submitted"
    MARKED = "marked"
    COMPLETED = "completed"


@dataclass
class ExamQuestion:
    """A single exam question."""

    id: str
    question_text: str
    question_type: QuestionType
    subject: ExamSubject
    topic: str
    subtopic: str = ""
    marks: int = 1
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    correct_answer: str = ""
    model_answer: str = ""
    marking_criteria: dict[str, Any] = field(default_factory=dict)
    kb_source: str = ""
    command_term: str = ""
    options: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question_text": self.question_text,
            "question_type": self.question_type.value,
            "subject": self.subject.value,
            "topic": self.topic,
            "subtopic": self.subtopic,
            "marks": self.marks,
            "difficulty": self.difficulty.value,
            "correct_answer": self.correct_answer,
            "model_answer": self.model_answer,
            "marking_criteria": self.marking_criteria,
            "kb_source": self.kb_source,
            "command_term": self.command_term,
            "options": self.options,
        }


@dataclass
class StudentAnswer:
    """A student's answer to an exam question."""

    question_id: str
    answer_text: str
    time_spent_seconds: float = 0.0
    is_partial: bool = False
    confidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "answer_text": self.answer_text,
            "time_spent_seconds": self.time_spent_seconds,
            "is_partial": self.is_partial,
            "confidence": self.confidence,
        }


@dataclass
class QuestionMarking:
    """Marking result for a single question."""

    question_id: str
    marks_awarded: float
    marks_possible: float
    percentage: float
    feedback: str = ""
    issues: list[str] = field(default_factory=list)
    rubric_scores: dict[str, float] = field(default_factory=dict)
    is_correct: bool = False
    partial_credit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "marks_awarded": self.marks_awarded,
            "marks_possible": self.marks_possible,
            "percentage": self.percentage,
            "feedback": self.feedback,
            "issues": self.issues,
            "rubric_scores": self.rubric_scores,
            "is_correct": self.is_correct,
            "partial_credit": self.partial_credit,
        }


@dataclass
class ExamSession:
    """An exam simulation session."""

    id: str
    subject: ExamSubject
    status: SessionStatus
    time_limit_minutes: int
    topic_scope: str
    difficulty: DifficultyLevel
    question_count: int
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    submitted_at: float | None = None
    marked_at: float | None = None
    questions: list[ExamQuestion] = field(default_factory=list)
    answers: list[StudentAnswer] = field(default_factory=list)
    markings: list[QuestionMarking] = field(default_factory=list)
    total_marks: float = 0.0
    earned_marks: float = 0.0
    percentage: float = 0.0
    grade: str = ""
    weak_topics: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        subject: ExamSubject,
        time_limit_minutes: int,
        topic_scope: str,
        difficulty: DifficultyLevel,
        question_count: int,
        metadata: dict[str, Any] = None,
    ) -> "ExamSession":
        import uuid

        return cls(
            id=f"exam_{uuid.uuid4().hex[:12]}",
            subject=subject,
            status=SessionStatus.CREATED,
            time_limit_minutes=time_limit_minutes,
            topic_scope=topic_scope,
            difficulty=difficulty,
            question_count=question_count,
            metadata=metadata or {},
        )

    def start(self) -> None:
        """Mark session as started."""
        self.status = SessionStatus.IN_PROGRESS
        self.started_at = time.time()

    def submit(self) -> None:
        """Mark session as submitted."""
        self.status = SessionStatus.SUBMITTED
        self.submitted_at = time.time()

    def add_question(self, question: ExamQuestion) -> None:
        """Add a question to the session."""
        self.questions.append(question)
        self.total_marks += question.marks

    def add_answer(self, answer: StudentAnswer) -> None:
        """Add a student's answer."""
        self.answers.append(answer)

    def add_marking(self, marking: QuestionMarking) -> None:
        """Add marking result."""
        self.markings.append(marking)
        self.earned_marks += marking.marks_awarded
        self.percentage = (
            (self.earned_marks / self.total_marks * 100) if self.total_marks > 0 else 0
        )

    def calculate_grade(self) -> str:
        """Calculate grade based on percentage."""
        if self.percentage >= 90:
            self.grade = "A"
        elif self.percentage >= 80:
            self.grade = "B"
        elif self.percentage >= 70:
            self.grade = "C"
        elif self.percentage >= 60:
            self.grade = "D"
        else:
            self.grade = "E"
        return self.grade

    def analyze_weak_topics(self) -> list[str]:
        """Analyze and identify weak topics from markings."""
        topic_scores = {}

        for marking in self.markings:
            question = next((q for q in self.questions if q.id == marking.question_id), None)
            if question:
                if question.topic not in topic_scores:
                    topic_scores[question.topic] = []
                topic_scores[question.topic].append(marking.percentage)

        weak = []
        for topic, scores in topic_scores.items():
            avg = sum(scores) / len(scores)
            if avg < 70:
                weak.append(topic)

        self.weak_topics = weak
        return weak

    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self.started_at is None:
            return 0.0
        end_time = self.submitted_at or self.marked_at or time.time()
        return end_time - self.started_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subject": self.subject.value,
            "status": self.status.value,
            "time_limit_minutes": self.time_limit_minutes,
            "topic_scope": self.topic_scope,
            "difficulty": self.difficulty.value,
            "question_count": self.question_count,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "submitted_at": self.submitted_at,
            "marked_at": self.marked_at,
            "total_marks": self.total_marks,
            "earned_marks": self.earned_marks,
            "percentage": self.percentage,
            "grade": self.grade,
            "weak_topics": self.weak_topics,
            "metadata": self.metadata,
            "question_count_actual": len(self.questions),
            "answers_count": len(self.answers),
        }


class ExamSessionStore:
    """SQLite storage for exam sessions."""

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            data_dir = Path(__file__).parent.parent.parent / "data" / "user"
            data_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(data_dir / "exam_sessions.db")
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS exam_sessions (
                    id TEXT PRIMARY KEY,
                    subject TEXT NOT NULL,
                    status TEXT NOT NULL,
                    time_limit_minutes INTEGER,
                    topic_scope TEXT,
                    difficulty TEXT,
                    question_count INTEGER,
                    created_at REAL,
                    started_at REAL,
                    submitted_at REAL,
                    marked_at REAL,
                    total_marks REAL,
                    earned_marks REAL,
                    percentage REAL,
                    grade TEXT,
                    weak_topics TEXT,
                    metadata TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS exam_questions (
                    session_id TEXT,
                    question_id TEXT,
                    question_text TEXT,
                    question_type TEXT,
                    subject TEXT,
                    topic TEXT,
                    subtopic TEXT,
                    marks INTEGER,
                    difficulty TEXT,
                    correct_answer TEXT,
                    model_answer TEXT,
                    marking_criteria TEXT,
                    kb_source TEXT,
                    command_term TEXT,
                    options TEXT,
                    PRIMARY KEY (session_id, question_id),
                    FOREIGN KEY (session_id) REFERENCES exam_sessions(id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS exam_answers (
                    session_id TEXT,
                    question_id TEXT,
                    answer_text TEXT,
                    time_spent_seconds REAL,
                    is_partial INTEGER,
                    confidence TEXT,
                    PRIMARY KEY (session_id, question_id),
                    FOREIGN KEY (session_id, question_id) 
                        REFERENCES exam_questions(session_id, question_id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS exam_markings (
                    session_id TEXT,
                    question_id TEXT,
                    marks_awarded REAL,
                    marks_possible REAL,
                    percentage REAL,
                    feedback TEXT,
                    issues TEXT,
                    rubric_scores TEXT,
                    is_correct INTEGER,
                    partial_credit INTEGER,
                    PRIMARY KEY (session_id, question_id),
                    FOREIGN KEY (session_id, question_id) 
                        REFERENCES exam_questions(session_id, question_id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS exam_analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    subject TEXT,
                    percentage REAL,
                    grade TEXT,
                    weak_topics TEXT,
                    timestamp REAL,
                    FOREIGN KEY (session_id) REFERENCES exam_sessions(id)
                )
            """)

            conn.commit()

    def save_session(self, session: ExamSession) -> None:
        """Save an exam session."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO exam_sessions
                (id, subject, status, time_limit_minutes, topic_scope, difficulty,
                 question_count, created_at, started_at, submitted_at, marked_at,
                 total_marks, earned_marks, percentage, grade, weak_topics, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    session.id,
                    session.subject.value,
                    session.status.value,
                    session.time_limit_minutes,
                    session.topic_scope,
                    session.difficulty.value,
                    session.question_count,
                    session.created_at,
                    session.started_at,
                    session.submitted_at,
                    session.marked_at,
                    session.total_marks,
                    session.earned_marks,
                    session.percentage,
                    session.grade,
                    json.dumps(session.weak_topics),
                    json.dumps(session.metadata),
                ),
            )

            for q in session.questions:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO exam_questions
                    (session_id, question_id, question_text, question_type, subject,
                     topic, subtopic, marks, difficulty, correct_answer, model_answer,
                     marking_criteria, kb_source, command_term, options)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        session.id,
                        q.id,
                        q.question_text,
                        q.question_type.value,
                        q.subject.value,
                        q.topic,
                        q.subtopic,
                        q.marks,
                        q.difficulty.value,
                        q.correct_answer,
                        q.model_answer,
                        json.dumps(q.marking_criteria),
                        q.kb_source,
                        q.command_term,
                        json.dumps(q.options),
                    ),
                )

            for a in session.answers:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO exam_answers
                    (session_id, question_id, answer_text, time_spent_seconds,
                     is_partial, confidence)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        session.id,
                        a.question_id,
                        a.answer_text,
                        a.time_spent_seconds,
                        1 if a.is_partial else 0,
                        a.confidence,
                    ),
                )

            for m in session.markings:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO exam_markings
                    (session_id, question_id, marks_awarded, marks_possible, percentage,
                     feedback, issues, rubric_scores, is_correct, partial_credit)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        session.id,
                        m.question_id,
                        m.marks_awarded,
                        m.marks_possible,
                        m.percentage,
                        m.feedback,
                        json.dumps(m.issues),
                        json.dumps(m.rubric_scores),
                        1 if m.is_correct else 0,
                        1 if m.partial_credit else 0,
                    ),
                )

            conn.commit()

    def get_session(self, session_id: str) -> ExamSession | None:
        """Get a session by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM exam_sessions WHERE id = ?", (session_id,))
            row = cursor.fetchone()
            if not row:
                return None

            session = ExamSession(
                id=row[0],
                subject=ExamSubject(row[1]),
                status=SessionStatus(row[2]),
                time_limit_minutes=row[3],
                topic_scope=row[4],
                difficulty=DifficultyLevel(row[5]),
                question_count=row[6],
                created_at=row[7],
                started_at=row[8],
                submitted_at=row[9],
                marked_at=row[10],
                total_marks=row[11],
                earned_marks=row[12],
                percentage=row[13],
                grade=row[14],
                weak_topics=json.loads(row[15]) if row[15] else [],
                metadata=json.loads(row[16]) if row[16] else {},
            )

            cursor.execute("SELECT * FROM exam_questions WHERE session_id = ?", (session_id,))
            for q_row in cursor.fetchall():
                session.questions.append(
                    ExamQuestion(
                        id=q_row[1],
                        question_text=q_row[2],
                        question_type=QuestionType(q_row[3]),
                        subject=ExamSubject(q_row[4]),
                        topic=q_row[5],
                        subtopic=q_row[6] or "",
                        marks=q_row[7],
                        difficulty=DifficultyLevel(q_row[8]),
                        correct_answer=q_row[9],
                        model_answer=q_row[10],
                        marking_criteria=json.loads(q_row[11]) if q_row[11] else {},
                        kb_source=q_row[12],
                        command_term=q_row[13],
                        options=json.loads(q_row[14]) if q_row[14] else [],
                    )
                )

            cursor.execute("SELECT * FROM exam_answers WHERE session_id = ?", (session_id,))
            for a_row in cursor.fetchall():
                session.answers.append(
                    StudentAnswer(
                        question_id=a_row[1],
                        answer_text=a_row[2],
                        time_spent_seconds=a_row[3],
                        is_partial=bool(a_row[4]),
                        confidence=a_row[5],
                    )
                )

            cursor.execute("SELECT * FROM exam_markings WHERE session_id = ?", (session_id,))
            for m_row in cursor.fetchall():
                session.markings.append(
                    QuestionMarking(
                        question_id=m_row[1],
                        marks_awarded=m_row[2],
                        marks_possible=m_row[3],
                        percentage=m_row[4],
                        feedback=m_row[5],
                        issues=json.loads(m_row[6]) if m_row[6] else [],
                        rubric_scores=json.loads(m_row[7]) if m_row[7] else {},
                        is_correct=bool(m_row[8]),
                        partial_credit=bool(m_row[9]),
                    )
                )

            return session

    def get_recent_sessions(
        self,
        subject: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get recent session summaries."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if subject:
                cursor.execute(
                    "SELECT * FROM exam_sessions WHERE subject = ? ORDER BY created_at DESC LIMIT ?",
                    (subject, limit),
                )
            else:
                cursor.execute(
                    "SELECT * FROM exam_sessions ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                )

            return [
                dict(
                    zip(
                        [
                            "id",
                            "subject",
                            "status",
                            "time_limit_minutes",
                            "topic_scope",
                            "difficulty",
                            "question_count",
                            "created_at",
                            "started_at",
                            "submitted_at",
                            "marked_at",
                            "total_marks",
                            "earned_marks",
                            "percentage",
                            "grade",
                            "weak_topics",
                            "metadata",
                        ],
                        row,
                    )
                )
                for row in cursor.fetchall()
            ]

    def get_subject_analytics(
        self,
        subject: str,
        days: int = 30,
    ) -> dict[str, Any]:
        """Get analytics for a subject."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT COUNT(*), AVG(percentage), AVG(earned_marks), AVG(total_marks)
                FROM exam_sessions
                WHERE subject = ? AND created_at > ?
            """,
                (subject, time.time() - (days * 24 * 60 * 60)),
            )

            count, avg_pct, avg_earned, avg_total = cursor.fetchone() or (0, 0, 0, 0)

            cursor.execute(
                """
                SELECT weak_topics, COUNT(*) as freq
                FROM exam_sessions
                WHERE subject = ? AND weak_topics != '[]' AND created_at > ?
                GROUP BY weak_topics
                ORDER BY freq DESC
                LIMIT 10
            """,
                (subject, time.time() - (days * 24 * 60 * 60)),
            )

            weak_topic_counts = {}
            for row in cursor.fetchall():
                topics = json.loads(row[0]) if row[0] else []
                for topic in topics:
                    weak_topic_counts[topic] = weak_topic_counts.get(topic, 0) + row[1]

            sorted_weak = sorted(weak_topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            top_weak_topics = [{"topic": t, "count": c} for t, c in sorted_weak]

            return {
                "subject": subject,
                "days": days,
                "session_count": count or 0,
                "average_percentage": round(avg_pct, 1) if avg_pct else 0,
                "average_marks": f"{avg_earned or 0:.1f}/{avg_total or 0:.1f}",
                "top_weak_topics": top_weak_topics,
            }

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all related data."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("DELETE FROM exam_markings WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM exam_answers WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM exam_questions WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM exam_sessions WHERE id = ?", (session_id,))
            cursor.execute("DELETE FROM exam_analytics WHERE session_id = ?", (session_id,))

            return cursor.rowcount > 0


def get_exam_store() -> ExamSessionStore:
    """Get the exam session store instance."""
    return ExamSessionStore()
