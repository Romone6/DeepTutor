"""
Memory and Weakness Tracking SQLite Store

Lightweight persistence for tracking student weaknesses, spaced repetition,
and review scheduling.

Database Schema:
- weakness_records: Individual weakness entries
- interaction_logs: History of all interactions
- review_queue: Spaced repetition queue
- topic_mastery: Per-topic mastery scores
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any


class ErrorType(str, Enum):
    """Types of errors that can be tracked."""

    CONCEPTUAL_MISUNDERSTANDING = "conceptual"
    CALCULATION_ERROR = "calculation"
    FORMULA_FORGOT = "formula"
    DEFINITION_WRONG = "definition"
    PROCEDURE_ERROR = "procedure"
    LOGIC_ERROR = "logic"
    APPLICATION_ERROR = "application"
    COMPREHENSION_GAP = "comprehension"
    CONFUSION = "confusion"
    OTHER = "other"


class ConfidenceLevel(str, Enum):
    """Confidence levels for self-assessment."""

    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class WeaknessRecord:
    """A recorded weakness or difficulty."""

    id: int | None = None
    subject: str = ""
    topic: str = ""
    subtopic: str = ""
    error_type: ErrorType = ErrorType.OTHER
    description: str = ""
    question_id: str | None = None
    user_answer: str = ""
    correct_answer: str = ""
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    timestamp: float = field(default_factory=time.time)
    review_count: int = 0
    last_reviewed: float | None = None
    mastery_score: float = 0.0  # 0.0 to 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subject": self.subject,
            "topic": self.topic,
            "subtopic": self.subtopic,
            "error_type": self.error_type.value,
            "description": self.description,
            "question_id": self.question_id,
            "user_answer": self.user_answer,
            "correct_answer": self.correct_answer,
            "confidence": self.confidence.value,
            "timestamp": self.timestamp,
            "review_count": self.review_count,
            "last_reviewed": self.last_reviewed,
            "mastery_score": self.mastery_score,
        }


@dataclass
class ReviewItem:
    """An item in the spaced repetition review queue."""

    weakness_id: int
    subject: str
    topic: str
    scheduled_date: float  # When to review (timestamp)
    priority: float = 0.0  # Higher = more urgent
    interval_days: float = 1.0  # Days since last review
    ease_factor: float = 2.5  # SM-2 algorithm ease factor
    interval_hours: float = field(default_factory=lambda: 24.0)  # Hours until next review
    id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "weakness_id": self.weakness_id,
            "subject": self.subject,
            "topic": self.topic,
            "priority": self.priority,
            "scheduled_date": self.scheduled_date,
            "interval_days": self.interval_days,
            "ease_factor": self.ease_factor,
            "interval_hours": self.interval_hours,
        }


class MemoryStore:
    """SQLite-based memory store for weakness tracking."""

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            data_dir = Path(__file__).parent.parent.parent / "data" / "user"
            data_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(data_dir / "memory.db")
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS weakness_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    subtopic TEXT,
                    error_type TEXT NOT NULL DEFAULT 'other',
                    description TEXT,
                    question_id TEXT,
                    user_answer TEXT,
                    correct_answer TEXT,
                    confidence TEXT DEFAULT 'medium',
                    timestamp REAL NOT NULL,
                    review_count INTEGER DEFAULT 0,
                    last_reviewed REAL,
                    mastery_score REAL DEFAULT 0.0,
                    UNIQUE(subject, topic, question_id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS review_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    weakness_id INTEGER NOT NULL,
                    subject TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    priority REAL DEFAULT 0.0,
                    scheduled_date REAL NOT NULL,
                    interval_days REAL DEFAULT 1.0,
                    ease_factor REAL DEFAULT 2.5,
                    interval_hours REAL DEFAULT 24.0,
                    FOREIGN KEY (weakness_id) REFERENCES weakness_records(id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS topic_mastery (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    mastery_score REAL DEFAULT 0.0,
                    interactions INTEGER DEFAULT 0,
                    correct_count INTEGER DEFAULT 0,
                    last_interaction REAL,
                    UNIQUE(subject, topic)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interaction_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    subject TEXT,
                    topic TEXT,
                    question_id TEXT,
                    correct BOOLEAN,
                    confidence TEXT,
                    timestamp REAL NOT NULL,
                    duration_ms REAL,
                    metadata TEXT
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_weakness_subject_topic
                ON weakness_records(subject, topic)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_review_scheduled
                ON review_queue(scheduled_date)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_mastery_subject
                ON topic_mastery(subject)
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sprint_plans (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    subjects TEXT NOT NULL,
                    total_days INTEGER NOT NULL,
                    daily_hours REAL NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    plan_data TEXT NOT NULL,
                    topic_coverage TEXT,
                    status TEXT DEFAULT 'draft',
                    created_at REAL NOT NULL,
                    updated_at REAL
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sprint_days (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plan_id TEXT NOT NULL,
                    day_number INTEGER NOT NULL,
                    date TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    topics_covered TEXT NOT NULL,
                    activities TEXT,
                    quiz TEXT,
                    checkpoint TEXT,
                    completion_status TEXT DEFAULT 'pending',
                    notes TEXT,
                    FOREIGN KEY (plan_id) REFERENCES sprint_plans(id)
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sprint_plan_id
                ON sprint_days(plan_id)
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS study_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    plan_id TEXT,
                    day_number INTEGER,
                    subject TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    difficulty TEXT DEFAULT 'medium',
                    activity_type TEXT,
                    time_spent_seconds REAL DEFAULT 0,
                    questions_attempted INTEGER DEFAULT 0,
                    questions_correct INTEGER DEFAULT 0,
                    accuracy REAL DEFAULT 0.0,
                    confidence_avg REAL DEFAULT 0.0,
                    timestamp REAL NOT NULL,
                    date TEXT NOT NULL
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_subject_date
                ON study_metrics(subject, date)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_session
                ON study_metrics(session_id)
            """)

            conn.commit()

            conn.commit()

    def add_weakness(
        self,
        subject: str,
        topic: str,
        error_type: ErrorType | str,
        description: str = "",
        question_id: str | None = None,
        user_answer: str = "",
        correct_answer: str = "",
        confidence: ConfidenceLevel | str = ConfidenceLevel.MEDIUM,
    ) -> int:
        """Add a new weakness record."""

        error_type = ErrorType(error_type) if isinstance(error_type, str) else error_type
        confidence = ConfidenceLevel(confidence) if isinstance(confidence, str) else confidence

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO weakness_records
                (subject, topic, error_type, description, question_id,
                 user_answer, correct_answer, confidence, timestamp, mastery_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    subject,
                    topic,
                    error_type.value,
                    description,
                    question_id,
                    user_answer,
                    correct_answer,
                    confidence.value,
                    time.time(),
                    self._calculate_initial_mastery(confidence),
                ),
            )

            conn.commit()
            last_id = cursor.lastrowid
            return last_id if last_id is not None else 0

    def _calculate_initial_mastery(self, confidence: ConfidenceLevel) -> float:
        """Calculate initial mastery score based on confidence."""
        confidence_scores = {
            ConfidenceLevel.VERY_LOW: 0.1,
            ConfidenceLevel.LOW: 0.25,
            ConfidenceLevel.MEDIUM: 0.5,
            ConfidenceLevel.HIGH: 0.75,
            ConfidenceLevel.VERY_HIGH: 0.9,
        }
        return confidence_scores.get(confidence, 0.5)

    def update_mastery(
        self,
        weakness_id: int,
        correct: bool,
        new_confidence: ConfidenceLevel | str | None = None,
    ) -> None:
        """Update mastery score after a review attempt."""

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT mastery_score, review_count FROM weakness_records WHERE id = ?",
                (weakness_id,),
            )
            row = cursor.fetchone()
            if not row:
                return

            current_mastery, review_count = row

            if correct:
                new_mastery = min(1.0, current_mastery + (1.0 - current_mastery) * 0.2)
            else:
                new_mastery = max(0.0, current_mastery - 0.1)

            review_count += 1

            cursor.execute(
                """
                UPDATE weakness_records
                SET mastery_score = ?, review_count = ?, last_reviewed = ?
                WHERE id = ?
            """,
                (new_mastery, review_count, time.time(), weakness_id),
            )

            conn.commit()

    def get_weakness_topics(
        self,
        subject: str | None = None,
        min_mastery: float = 0.0,
        max_mastery: float = 1.0,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get list of weakness topics, optionally filtered."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if subject:
                cursor.execute(
                    """
                    SELECT subject, topic, COUNT(*) as count,
                           AVG(mastery_score) as avg_mastery,
                           SUM(review_count) as total_reviews
                    FROM weakness_records
                    WHERE subject = ? AND mastery_score BETWEEN ? AND ?
                    GROUP BY subject, topic
                    ORDER BY avg_mastery ASC
                    LIMIT ?
                """,
                    (subject, min_mastery, max_mastery, limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT subject, topic, COUNT(*) as count,
                           AVG(mastery_score) as avg_mastery,
                           SUM(review_count) as total_reviews
                    FROM weakness_records
                    WHERE mastery_score BETWEEN ? AND ?
                    GROUP BY subject, topic
                    ORDER BY avg_mastery ASC
                    LIMIT ?
                """,
                    (min_mastery, max_mastery, limit),
                )

            return [dict(row) for row in cursor.fetchall()]

    def get_weakness_records(
        self,
        subject: str | None = None,
        topic: str | None = None,
        error_type: ErrorType | str | None = None,
        limit: int = 100,
    ) -> list[WeaknessRecord]:
        """Get weakness records with optional filters."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = "SELECT * FROM weakness_records WHERE 1=1"
            params = []

            if subject:
                query += " AND subject = ?"
                params.append(subject)
            if topic:
                query += " AND topic = ?"
                params.append(topic)
            if error_type:
                error_type_val = (
                    error_type.value if isinstance(error_type, ErrorType) else error_type
                )
                query += " AND error_type = ?"
                params.append(error_type_val)

            query += " ORDER BY mastery_score ASC, timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)

            records = []
            for row in cursor.fetchall():
                record = WeaknessRecord(
                    id=row["id"],
                    subject=row["subject"],
                    topic=row["topic"],
                    subtopic=row["subtopic"],
                    error_type=ErrorType(row["error_type"]),
                    description=row["description"],
                    question_id=row["question_id"],
                    user_answer=row["user_answer"],
                    correct_answer=row["correct_answer"],
                    confidence=ConfidenceLevel(row["confidence"]),
                    timestamp=row["timestamp"],
                    review_count=row["review_count"],
                    last_reviewed=row["last_reviewed"],
                    mastery_score=row["mastery_score"],
                )
                records.append(record)

            return records

    def add_to_review_queue(
        self,
        weakness_id: int,
        subject: str,
        topic: str,
        priority: float | None = None,
        hours: float | None = None,
    ) -> int:
        """Add a weakness to the review queue with spaced repetition."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT ease_factor, interval_hours FROM review_queue WHERE weakness_id = ? ORDER BY id DESC",
                (weakness_id,),
            )
            last_review = cursor.fetchone()

            if last_review:
                ease_factor = last_review["ease_factor"]
                interval_hours = last_review["interval_hours"]
            else:
                ease_factor = 2.5
                interval_hours = 24.0

            delay_hours = hours if hours is not None else interval_hours
            scheduled_date = time.time() + delay_hours * 3600

            cursor.execute(
                """
                INSERT INTO review_queue
                (weakness_id, subject, topic, priority, scheduled_date, ease_factor, interval_hours)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    weakness_id,
                    subject,
                    topic,
                    priority or 0.5,
                    scheduled_date,
                    ease_factor,
                    interval_hours,
                ),
            )

            conn.commit()
            last_id = cursor.lastrowid
            return last_id if last_id is not None else 0

    def get_review_queue(self, limit: int = 20) -> list[ReviewItem]:
        """Get items due for review."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM review_queue
                WHERE scheduled_date <= ?
                ORDER BY priority DESC, scheduled_date ASC
                LIMIT ?
            """,
                (time.time(), limit),
            )

            items = []
            for row in cursor.fetchall():
                item = ReviewItem(
                    id=row["id"],
                    weakness_id=row["weakness_id"],
                    subject=row["subject"],
                    topic=row["topic"],
                    priority=row["priority"],
                    scheduled_date=row["scheduled_date"],
                    interval_days=row["interval_days"],
                    ease_factor=row["ease_factor"],
                    interval_hours=row["interval_hours"],
                )
                items.append(item)

            return items

    def complete_review(
        self,
        review_id: int,
        quality: int,  # 0-5 quality rating (SM-2)
        correct: bool,
    ) -> None:
        """Mark a review as complete and update spaced repetition."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT weakness_id, ease_factor, interval_hours FROM review_queue WHERE id = ?",
                (review_id,),
            )
            review = cursor.fetchone()
            if not review:
                return

            weakness_id, ease_factor, current_interval = review

            if quality < 3:
                ease_factor = max(1.3, ease_factor - 0.2)
                new_interval = current_interval * 0.5
            else:
                new_interval = current_interval * ease_factor if correct else current_interval

            new_interval = min(new_interval, 720.0)  # Max 30 days
            new_interval = max(new_interval, 1.0)  # Min 1 hour

            cursor.execute(
                """
                UPDATE review_queue
                SET ease_factor = ?, interval_hours = ?, scheduled_date = ?
                WHERE id = ?
            """,
                (ease_factor, new_interval, time.time() + new_interval * 3600, review_id),
            )

            conn.commit()

        self.update_mastery(weakness_id, correct)

    def log_interaction(
        self,
        session_id: str,
        subject: str,
        topic: str,
        question_id: str | None,
        correct: bool,
        confidence: ConfidenceLevel | str,
        duration_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Log an interaction for tracking."""
        confidence_val = confidence.value if isinstance(confidence, ConfidenceLevel) else confidence

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO interaction_logs
                (session_id, subject, topic, question_id, correct, confidence, timestamp, duration_ms, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    session_id,
                    subject,
                    topic,
                    question_id,
                    correct,
                    confidence_val,
                    time.time(),
                    duration_ms,
                    json.dumps(metadata) if metadata else None,
                ),
            )

            conn.commit()
            last_id = cursor.lastrowid
            return last_id if last_id is not None else 0

    def get_interaction_stats(
        self,
        subject: str | None = None,
        days: int = 7,
    ) -> dict[str, Any]:
        """Get interaction statistics."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            since = time.time() - (days * 24 * 3600)

            if subject:
                cursor.execute(
                    """
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN correct THEN 1 ELSE 0 END) as correct_count,
                        AVG(duration_ms) as avg_duration
                    FROM interaction_logs
                    WHERE subject = ? AND timestamp >= ?
                """,
                    (subject, since),
                )
            else:
                cursor.execute(
                    """
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN correct THEN 1 ELSE 0 END) as correct_count,
                        AVG(duration_ms) as avg_duration
                    FROM interaction_logs
                    WHERE timestamp >= ?
                """,
                    (since,),
                )

            row = cursor.fetchone()

            total = row["total"] or 0
            correct_count = row["correct_count"] or 0

            return {
                "total_interactions": total,
                "correct_count": correct_count,
                "accuracy": (correct_count / total * 100) if total > 0 else 0.0,
                "avg_duration_ms": row["avg_duration"] or 0,
                "days": days,
            }

    def get_topic_mastery(self, subject: str | None = None) -> list[dict[str, Any]]:
        """Get mastery scores per topic."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if subject:
                cursor.execute(
                    """
                    SELECT * FROM topic_mastery
                    WHERE subject = ?
                    ORDER BY mastery_score ASC
                """,
                    (subject,),
                )
            else:
                cursor.execute("""
                    SELECT * FROM topic_mastery
                    ORDER BY mastery_score ASC
                """)

            return [dict(row) for row in cursor.fetchall()]

    def cleanup_old_records(self, days: int = 90) -> int:
        """Remove old records beyond retention period."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cutoff = time.time() - (days * 24 * 3600)

            cursor.execute("DELETE FROM interaction_logs WHERE timestamp < ?", (cutoff,))
            deleted = cursor.rowcount

            conn.commit()
            return deleted

    def get_all_weakness_counts(self) -> dict[str, int]:
        """Get count of weaknesses per subject."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT subject, COUNT(*) as count
                FROM weakness_records
                GROUP BY subject
            """)

            return {row["subject"]: row["count"] for row in cursor.fetchall()}

    # =========================================================================
    # Sprint Plan Methods
    # =========================================================================

    def save_sprint_plan(self, plan_data: dict) -> str:
        """Save a sprint plan to the database."""
        import json

        with self._get_connection() as conn:
            cursor = conn.cursor()

            plan_id = plan_data["id"]
            subjects = json.dumps(plan_data["subjects"])
            plan_json = json.dumps(plan_data)
            topic_coverage = json.dumps(plan_data.get("topic_coverage", {}))
            created_at = plan_data.get("created_at", time.time())

            cursor.execute(
                """
                INSERT OR REPLACE INTO sprint_plans
                (id, name, subjects, total_days, daily_hours, start_date, end_date,
                 plan_data, topic_coverage, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    plan_id,
                    plan_data["name"],
                    subjects,
                    plan_data["total_days"],
                    plan_data["daily_hours"],
                    plan_data["start_date"],
                    plan_data["end_date"],
                    plan_json,
                    topic_coverage,
                    plan_data.get("status", "draft"),
                    created_at,
                    time.time(),
                ),
            )

            for day in plan_data.get("days", []):
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO sprint_days
                    (plan_id, day_number, date, subject, topics_covered,
                     activities, quiz, checkpoint, completion_status, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        plan_id,
                        day["day_number"],
                        day["date"],
                        day["subject"],
                        json.dumps(day["topics_covered"]),
                        json.dumps(day.get("activities", [])),
                        json.dumps(day.get("quiz")),
                        json.dumps(day.get("checkpoint")),
                        day.get("completion_status", "pending"),
                        day.get("notes", ""),
                    ),
                )

            conn.commit()
            return plan_id

    def get_sprint_plan(self, plan_id: str) -> dict | None:
        """Retrieve a sprint plan by ID."""
        import json

        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM sprint_plans WHERE id = ?", (plan_id,))
            plan_row = cursor.fetchone()

            if not plan_row:
                return None

            plan = dict(plan_row)
            plan["subjects"] = json.loads(plan["subjects"])
            plan["topic_coverage"] = json.loads(plan["topic_coverage"] or "{}")

            cursor.execute(
                "SELECT * FROM sprint_days WHERE plan_id = ? ORDER BY day_number",
                (plan_id,),
            )
            days = []
            for row in cursor.fetchall():
                day = dict(row)
                day["topics_covered"] = json.loads(day["topics_covered"] or "[]")
                day["activities"] = json.loads(day["activities"] or "[]")
                day["quiz"] = json.loads(day["quiz"] or "null")
                day["checkpoint"] = json.loads(day["checkpoint"] or "null")
                days.append(day)

            plan["days"] = days
            return plan

    def get_all_sprint_plans(self) -> list[dict]:
        """Get all sprint plans."""
        import json

        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                "SELECT id, name, subjects, total_days, status, created_at, start_date, end_date "
                "FROM sprint_plans ORDER BY created_at DESC"
            )

            plans = []
            for row in cursor.fetchall():
                plans.append(
                    {
                        "id": row["id"],
                        "name": row["name"],
                        "subjects": json.loads(row["subjects"]),
                        "total_days": row["total_days"],
                        "status": row["status"],
                        "created_at": row["created_at"],
                        "start_date": row["start_date"],
                        "end_date": row["end_date"],
                    }
                )

            return plans

    def update_sprint_day_status(
        self, plan_id: str, day_number: int, completion_status: str, notes: str = ""
    ) -> bool:
        """Update the completion status of a sprint day."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE sprint_days
                SET completion_status = ?, notes = ?
                WHERE plan_id = ? AND day_number = ?
            """,
                (completion_status, notes, plan_id, day_number),
            )

            conn.commit()
            return cursor.rowcount > 0

    def update_sprint_plan_status(self, plan_id: str, status: str) -> bool:
        """Update the status of a sprint plan."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE sprint_plans
                SET status = ?, updated_at = ?
                WHERE id = ?
            """,
                (status, time.time(), plan_id),
            )

            conn.commit()
            return cursor.rowcount > 0

    def delete_sprint_plan(self, plan_id: str) -> bool:
        """Delete a sprint plan and its days."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("DELETE FROM sprint_days WHERE plan_id = ?", (plan_id,))
            cursor.execute("DELETE FROM sprint_plans WHERE id = ?", (plan_id,))

            conn.commit()
            return cursor.rowcount > 0

    def get_sprint_stats(self) -> dict[str, Any]:
        """Get sprint plan statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM sprint_plans")
            total_plans = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM sprint_plans WHERE status = 'active'")
            active_plans = cursor.fetchone()[0]

            cursor.execute(
                """
                SELECT COUNT(*) FROM sprint_days
                WHERE completion_status = 'completed'
            """
            )
            completed_days = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM sprint_days")
            total_days = cursor.fetchone()[0]

            return {
                "total_plans": total_plans,
                "active_plans": active_plans,
                "total_days": total_days,
                "completed_days": completed_days,
                "completion_rate": ((completed_days / total_days * 100) if total_days > 0 else 0.0),
            }

    # =========================================================================
    # Study Metrics Methods (PID-32)
    # =========================================================================

    def record_study_metric(
        self,
        session_id: str,
        subject: str,
        topic: str,
        difficulty: str = "medium",
        activity_type: str | None = None,
        time_spent_seconds: float = 0,
        questions_attempted: int = 0,
        questions_correct: int = 0,
        confidence_avg: float = 0.0,
        plan_id: str | None = None,
        day_number: int | None = None,
    ) -> int:
        """Record a study session metric."""
        accuracy = (
            (questions_correct / questions_attempted * 100) if questions_attempted > 0 else 0.0
        )
        date_str = datetime.now().isoformat()[:10]

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO study_metrics
                (session_id, plan_id, day_number, subject, topic, difficulty, activity_type,
                 time_spent_seconds, questions_attempted, questions_correct, accuracy,
                 confidence_avg, timestamp, date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    session_id,
                    plan_id,
                    day_number,
                    subject,
                    topic,
                    difficulty,
                    activity_type,
                    time_spent_seconds,
                    questions_attempted,
                    questions_correct,
                    accuracy,
                    confidence_avg,
                    time.time(),
                    date_str,
                ),
            )

            conn.commit()
            last_id = cursor.lastrowid
            return last_id if last_id is not None else 0

    def get_subject_metrics(
        self,
        subject: str | None = None,
        days: int = 30,
    ) -> dict[str, Any]:
        """Get metrics grouped by subject."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            since = time.time() - (days * 24 * 3600)

            if subject:
                cursor.execute(
                    """
                    SELECT
                        subject,
                        COUNT(*) as total_sessions,
                        SUM(time_spent_seconds) as total_time,
                        SUM(questions_attempted) as total_questions,
                        SUM(questions_correct) as correct_questions,
                        AVG(accuracy) as avg_accuracy,
                        AVG(confidence_avg) as avg_confidence
                    FROM study_metrics
                    WHERE subject = ? AND timestamp >= ?
                    GROUP BY subject
                """,
                    (subject, since),
                )
            else:
                cursor.execute(
                    """
                    SELECT
                        subject,
                        COUNT(*) as total_sessions,
                        SUM(time_spent_seconds) as total_time,
                        SUM(questions_attempted) as total_questions,
                        SUM(questions_correct) as correct_questions,
                        AVG(accuracy) as avg_accuracy,
                        AVG(confidence_avg) as avg_confidence
                    FROM study_metrics
                    WHERE timestamp >= ?
                    GROUP BY subject
                """,
                    (since,),
                )

            results = []
            for row in cursor.fetchall():
                total_questions = row["total_questions"] or 0
                correct_questions = row["correct_questions"] or 0
                results.append(
                    {
                        "subject": row["subject"],
                        "total_sessions": row["total_sessions"] or 0,
                        "total_time_minutes": (row["total_time"] or 0) / 60,
                        "total_questions": total_questions,
                        "correct_questions": correct_questions,
                        "accuracy": (correct_questions / total_questions * 100)
                        if total_questions > 0
                        else 0.0,
                        "avg_accuracy": row["avg_accuracy"] or 0.0,
                        "avg_confidence": row["avg_confidence"] or 0.0,
                    }
                )

            return {"subjects": results, "total_subjects": len(results)}

    def get_topic_metrics(
        self,
        subject: str | None = None,
        days: int = 30,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get metrics per topic, sorted by accuracy (worst first)."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            since = time.time() - (days * 24 * 3600)

            if subject:
                cursor.execute(
                    """
                    SELECT
                        topic,
                        subject,
                        COUNT(*) as sessions,
                        SUM(time_spent_seconds) as total_time,
                        SUM(questions_attempted) as total_questions,
                        SUM(questions_correct) as correct_questions,
                        AVG(accuracy) as avg_accuracy,
                        AVG(confidence_avg) as avg_confidence
                    FROM study_metrics
                    WHERE subject = ? AND timestamp >= ?
                    GROUP BY topic
                    ORDER BY avg_accuracy ASC
                    LIMIT ?
                """,
                    (subject, since, limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT
                        topic,
                        subject,
                        COUNT(*) as sessions,
                        SUM(time_spent_seconds) as total_time,
                        SUM(questions_attempted) as total_questions,
                        SUM(questions_correct) as correct_questions,
                        AVG(accuracy) as avg_accuracy,
                        AVG(confidence_avg) as avg_confidence
                    FROM study_metrics
                    WHERE timestamp >= ?
                    GROUP BY topic
                    ORDER BY avg_accuracy ASC
                    LIMIT ?
                """,
                    (since, limit),
                )

            return [dict(row) for row in cursor.fetchall()]

    def get_accuracy_trend(
        self,
        subject: str | None = None,
        days: int = 14,
    ) -> list[dict[str, Any]]:
        """Get daily accuracy trend."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            since = time.time() - (days * 24 * 3600)

            if subject:
                cursor.execute(
                    """
                    SELECT
                        date,
                        COUNT(*) as sessions,
                        SUM(questions_attempted) as total_questions,
                        SUM(questions_correct) as correct_questions,
                        AVG(accuracy) as avg_accuracy
                    FROM study_metrics
                    WHERE subject = ? AND timestamp >= ?
                    GROUP BY date
                    ORDER BY date ASC
                """,
                    (subject, since),
                )
            else:
                cursor.execute(
                    """
                    SELECT
                        date,
                        COUNT(*) as sessions,
                        SUM(questions_attempted) as total_questions,
                        SUM(questions_correct) as correct_questions,
                        AVG(accuracy) as avg_accuracy
                    FROM study_metrics
                    WHERE timestamp >= ?
                    GROUP BY date
                    ORDER BY date ASC
                """,
                    (since,),
                )

            results = []
            for row in cursor.fetchall():
                total = row["total_questions"] or 0
                correct = row["correct_questions"] or 0
                results.append(
                    {
                        "date": row["date"],
                        "sessions": row["sessions"] or 0,
                        "total_questions": total,
                        "correct_questions": correct,
                        "accuracy": (correct / total * 100) if total > 0 else 0.0,
                        "avg_accuracy": row["avg_accuracy"] or 0.0,
                    }
                )

            return results

    def get_time_allocation(
        self,
        subject: str | None = None,
        days: int = 30,
    ) -> list[dict[str, Any]]:
        """Get time allocation per topic vs estimated syllabus weight."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            since = time.time() - (days * 24 * 3600)

            if subject:
                cursor.execute(
                    """
                    SELECT
                        topic,
                        SUM(time_spent_seconds) as total_time,
                        COUNT(*) as sessions,
                        AVG(accuracy) as accuracy
                    FROM study_metrics
                    WHERE subject = ? AND timestamp >= ?
                    GROUP BY topic
                    ORDER BY total_time DESC
                """,
                    (subject, since),
                )
            else:
                cursor.execute(
                    """
                    SELECT
                        topic,
                        subject,
                        SUM(time_spent_seconds) as total_time,
                        COUNT(*) as sessions,
                        AVG(accuracy) as accuracy
                    FROM study_metrics
                    WHERE timestamp >= ?
                    GROUP BY topic, subject
                    ORDER BY total_time DESC
                """,
                    (since,),
                )

            total_time = 0
            results = []
            for row in cursor.fetchall():
                time_seconds = row["total_time"] or 0
                total_time += time_seconds
                results.append(
                    {
                        "topic": row["topic"],
                        "subject": row.get("subject"),
                        "time_minutes": time_seconds / 60,
                        "sessions": row["sessions"] or 0,
                        "accuracy": row["accuracy"] or 0.0,
                    }
                )

            for r in results:
                r["percentage"] = (
                    (r["time_minutes"] / (total_time / 60) * 100) if total_time > 0 else 0.0
                )

            return results

    def get_weak_topics(
        self,
        subject: str | None = None,
        accuracy_threshold: float = 70.0,
        days: int = 30,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get topics with accuracy below threshold (weak areas)."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            since = time.time() - (days * 24 * 3600)

            if subject:
                cursor.execute(
                    """
                    SELECT
                        topic,
                        subject,
                        COUNT(*) as sessions,
                        SUM(questions_attempted) as total_questions,
                        SUM(questions_correct) as correct_questions,
                        AVG(accuracy) as avg_accuracy,
                        AVG(confidence_avg) as avg_confidence
                    FROM study_metrics
                    WHERE subject = ? AND timestamp >= ?
                    GROUP BY topic
                    HAVING avg_accuracy < ?
                    ORDER BY avg_accuracy ASC
                    LIMIT ?
                """,
                    (subject, since, accuracy_threshold, limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT
                        topic,
                        subject,
                        COUNT(*) as sessions,
                        SUM(questions_attempted) as total_questions,
                        SUM(questions_correct) as correct_questions,
                        AVG(accuracy) as avg_accuracy,
                        AVG(confidence_avg) as avg_confidence
                    FROM study_metrics
                    WHERE timestamp >= ?
                    GROUP BY topic
                    HAVING avg_accuracy < ?
                    ORDER BY avg_accuracy ASC
                    LIMIT ?
                """,
                    (since, accuracy_threshold, limit),
                )

            return [dict(row) for row in cursor.fetchall()]

    def get_dashboard_summary(
        self,
        subject: str | None = None,
        days: int = 30,
    ) -> dict[str, Any]:
        """Get complete dashboard summary."""
        subject_metrics = self.get_subject_metrics(subject, days)
        weak_topics = self.get_weak_topics(subject, days=days, limit=10)
        accuracy_trend = self.get_accuracy_trend(subject, days=min(days, 14))
        time_allocation = self.get_time_allocation(subject, days)

        overall_total_questions = sum(
            s.get("total_questions", 0) for s in subject_metrics["subjects"]
        )
        overall_correct = sum(s.get("correct_questions", 0) for s in subject_metrics["subjects"])
        overall_accuracy = (
            (overall_correct / overall_total_questions * 100)
            if overall_total_questions > 0
            else 0.0
        )

        total_time = sum(t["time_minutes"] for t in time_allocation)

        return {
            "period_days": days,
            "subject": subject,
            "overall_stats": {
                "total_questions": overall_total_questions,
                "correct_questions": overall_correct,
                "accuracy": overall_accuracy,
                "total_study_time_minutes": total_time,
            },
            "by_subject": subject_metrics["subjects"],
            "weak_topics": weak_topics,
            "accuracy_trend": accuracy_trend,
            "time_allocation": time_allocation,
        }

    def get_stats(self) -> dict[str, Any]:
        """Get overall memory store statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM weakness_records")
            weakness_count = cursor.fetchone()[0]

            cursor.execute(
                "SELECT COUNT(*) FROM review_queue WHERE scheduled_date <= ?", (time.time(),)
            )
            review_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM topic_mastery")
            topic_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM interaction_logs")
            interaction_count = cursor.fetchone()[0]

            return {
                "total_weaknesses": weakness_count,
                "pending_reviews": review_count,
                "tracked_topics": topic_count,
                "total_interactions": interaction_count,
            }


_store: MemoryStore | None = None


def get_memory_store(db_path: str | None = None) -> MemoryStore:
    """Get the global memory store instance."""
    global _store
    if _store is None:
        _store = MemoryStore(db_path=db_path)
    return _store


def reset_memory_store() -> None:
    """Reset the global memory store (for testing)."""
    global _store
    _store = None
