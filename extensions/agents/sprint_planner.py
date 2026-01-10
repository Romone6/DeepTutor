"""
Holiday Sprint Planner Agent
============================

Generates 21-day study plans that provide bird's-eye map coverage of syllabus content
with active recall checkpoints and daily mini-quizzes.

Usage:
    from extensions.agents.sprint_planner import SprintPlanner, SprintPlan, SprintDay

    planner = SprintPlanner()
    plan = planner.create_plan(
        subjects=["mathematics", "physics"],
        days=21,
        daily_hours=2.0,
        kb_name="hsc_mathematics",
    )
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class Subject(str, Enum):
    """Supported subjects for sprint planning."""

    MATHEMATICS = "mathematics"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"
    ENGLISH = "english"
    HISTORY = "history"
    GEOGRAPHY = "geography"
    ECONOMICS = "economics"


class DifficultyLevel(str, Enum):
    """Difficulty levels for topics."""

    FOUNDATION = "foundation"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXTENSION = "extension"


class ActivityType(str, Enum):
    """Types of learning activities."""

    PRE_READ = "pre_read"
    VIDEO_LECTURE = "video_lecture"
    CONCEPT_MAP = "concept_map"
    PRACTICE_PROBLEMS = "practice_problems"
    ACTIVE_RECALL = "active_recall"
    MINI_QUIZ = "mini_quiz"
    PAST_PAPER = "past_paper"
    REVIEW = "review"


@dataclass
class TopicNode:
    """A topic/subtopic in the syllabus."""

    id: str
    name: str
    subject: str
    parent_id: Optional[str] = None
    difficulty: DifficultyLevel = DifficultyLevel.INTERMEDIATE
    estimated_hours: float = 1.0
    prerequisites: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    is_core: bool = True


@dataclass
class LearningActivity:
    """A single learning activity for a day."""

    id: str
    type: ActivityType
    title: str
    description: str
    duration_minutes: int
    topic_id: str
    checkpoint_question: Optional[str] = None
    resources: list[str] = field(default_factory=list)


@dataclass
class MiniQuiz:
    """Daily mini-quiz for a sprint day."""

    id: str
    topic_ids: list[str]
    questions: list[dict] = field(default_factory=list)
    total_marks: int = 0
    time_limit_minutes: int = 10


@dataclass
class SprintDay:
    """A single day in the sprint plan."""

    day_number: int
    date: str
    subject: str
    topics_covered: list[str]
    activities: list[LearningActivity] = field(default_factory=list)
    quiz: Optional[MiniQuiz] = None
    checkpoint: Optional[dict] = None
    completion_status: str = "pending"  # pending, in_progress, completed
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "day_number": self.day_number,
            "date": self.date,
            "subject": self.subject,
            "topics_covered": self.topics_covered,
            "activities": [
                {
                    "id": a.id,
                    "type": a.type.value,
                    "title": a.title,
                    "description": a.description,
                    "duration_minutes": a.duration_minutes,
                    "topic_id": a.topic_id,
                    "checkpoint_question": a.checkpoint_question,
                    "resources": a.resources,
                }
                for a in self.activities
            ],
            "quiz": self.quiz.to_dict() if self.quiz else None,
            "checkpoint": self.checkpoint,
            "completion_status": self.completion_status,
            "notes": self.notes,
        }


@dataclass
class SprintPlan:
    """Complete 21-day sprint plan."""

    id: str
    name: str
    subjects: list[str]
    total_days: int
    daily_hours: float
    start_date: str
    end_date: str
    days: list[SprintDay] = field(default_factory=list)
    topic_coverage: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    status: str = "draft"  # draft, active, completed, paused

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "subjects": self.subjects,
            "total_days": self.total_days,
            "daily_hours": self.daily_hours,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "days": [d.to_dict() for d in self.days],
            "topic_coverage": self.topic_coverage,
            "created_at": self.created_at,
            "status": self.status,
        }


class SyllabusExtractor:
    """Extracts topic structure from knowledge base."""

    def __init__(self, kb_name: str = "default"):
        self.kb_name = kb_name
        self._topics: list[TopicNode] = []
        self._load_default_topics()

    def _load_default_topics(self) -> None:
        """Load default topic structures for common subjects."""
        self._topics = [
            # Mathematics HSC
            TopicNode(
                id="math_calc_1",
                name="Differentiation Basics",
                subject="mathematics",
                difficulty=DifficultyLevel.INTERMEDIATE,
                estimated_hours=2.0,
                keywords=["derivative", "gradient", "rate of change"],
            ),
            TopicNode(
                id="math_calc_2",
                name="Differentiation Rules",
                subject="mathematics",
                parent_id="math_calc_1",
                difficulty=DifficultyLevel.INTERMEDIATE,
                estimated_hours=2.5,
                prerequisites=["math_calc_1"],
                keywords=["chain rule", "product rule", "quotient rule"],
            ),
            TopicNode(
                id="math_calc_3",
                name="Applications of Differentiation",
                subject="mathematics",
                parent_id="math_calc_2",
                difficulty=DifficultyLevel.ADVANCED,
                estimated_hours=3.0,
                prerequisites=["math_calc_2"],
                keywords=["maxima", "minima", "optimisation"],
            ),
            TopicNode(
                id="math_calc_4",
                name="Integration",
                subject="mathematics",
                difficulty=DifficultyLevel.INTERMEDIATE,
                estimated_hours=3.0,
                keywords=["anti-derivative", "area", "accumulation"],
            ),
            TopicNode(
                id="math_stats_1",
                name="Probability",
                subject="mathematics",
                difficulty=DifficultyLevel.INTERMEDIATE,
                estimated_hours=2.0,
                keywords=["outcome", "event", "conditional"],
            ),
            TopicNode(
                id="math_stats_2",
                name="Statistical Analysis",
                subject="mathematics",
                difficulty=DifficultyLevel.INTERMEDIATE,
                estimated_hours=2.0,
                prerequisites=["math_stats_1"],
                keywords=["mean", "median", "mode", "standard deviation"],
            ),
            # Physics HSC
            TopicNode(
                id="phys_mech_1",
                name="Linear Motion",
                subject="physics",
                difficulty=DifficultyLevel.FOUNDATION,
                estimated_hours=1.5,
                keywords=["velocity", "acceleration", "displacement"],
            ),
            TopicNode(
                id="phys_mech_2",
                name="Forces and Newton's Laws",
                subject="physics",
                parent_id="phys_mech_1",
                difficulty=DifficultyLevel.INTERMEDIATE,
                estimated_hours=2.5,
                prerequisites=["phys_mech_1"],
                keywords=["newton", "force", "mass", "acceleration"],
            ),
            TopicNode(
                id="phys_mech_3",
                name="Momentum and Energy",
                subject="physics",
                parent_id="phys_mech_2",
                difficulty=DifficultyLevel.ADVANCED,
                estimated_hours=3.0,
                prerequisites=["phys_mech_2"],
                keywords=["momentum", "kinetic energy", "conservation"],
            ),
            TopicNode(
                id="phys_waves_1",
                name="Wave Properties",
                subject="physics",
                difficulty=DifficultyLevel.FOUNDATION,
                estimated_hours=1.5,
                keywords=["frequency", "wavelength", "amplitude"],
            ),
            TopicNode(
                id="phys_elect_1",
                name="Electrostatics",
                subject="physics",
                difficulty=DifficultyLevel.INTERMEDIATE,
                estimated_hours=2.0,
                keywords=["charge", "field", "potential"],
            ),
            # Chemistry HSC
            TopicNode(
                id="chem_struc_1",
                name="Atomic Structure",
                subject="chemistry",
                difficulty=DifficultyLevel.FOUNDATION,
                estimated_hours=1.5,
                keywords=["proton", "electron", "orbital"],
            ),
            TopicNode(
                id="chem_struc_2",
                name="Periodic Trends",
                subject="chemistry",
                parent_id="chem_struc_1",
                difficulty=DifficultyLevel.INTERMEDIATE,
                estimated_hours=2.0,
                keywords=["electronegativity", "ionisation energy", "atomic radius"],
            ),
            TopicNode(
                id="chem_react_1",
                name="Chemical Reactions",
                subject="chemistry",
                difficulty=DifficultyLevel.INTERMEDIATE,
                estimated_hours=2.5,
                keywords=["stoichiometry", "yield", "limiting reagent"],
            ),
            TopicNode(
                id="chem_react_2",
                name="Equilibrium",
                subject="chemistry",
                parent_id="chem_react_1",
                difficulty=DifficultyLevel.ADVANCED,
                estimated_hours=3.0,
                keywords=["le chatelier", "equilibrium constant", "reaction quotient"],
            ),
            # Biology HSC
            TopicNode(
                id="bio_cell_1",
                name="Cell Structure",
                subject="biology",
                difficulty=DifficultyLevel.FOUNDATION,
                estimated_hours=1.5,
                keywords=["organelle", "membrane", "cytoplasm"],
            ),
            TopicNode(
                id="bio_cell_2",
                name="Cell Processes",
                subject="biology",
                parent_id="bio_cell_1",
                difficulty=DifficultyLevel.INTERMEDIATE,
                estimated_hours=2.0,
                keywords=["photosynthesis", "respiration", "transport"],
            ),
            TopicNode(
                id="bio_gen_1",
                name="Genetics",
                subject="biology",
                difficulty=DifficultyLevel.INTERMEDIATE,
                estimated_hours=2.5,
                keywords=["DNA", "RNA", "protein synthesis", "mutation"],
            ),
            TopicNode(
                id="bio_gen_2",
                name="Evolution",
                subject="biology",
                parent_id="bio_gen_1",
                difficulty=DifficultyLevel.ADVANCED,
                estimated_hours=2.0,
                keywords=["natural selection", "adaptation", "speciation"],
            ),
        ]

    def get_topics_for_subject(self, subject: str) -> list[TopicNode]:
        """Get all topics for a given subject."""
        return [t for t in self._topics if t.subject == subject]

    def get_topics_by_difficulty(
        self, subject: str, difficulty: DifficultyLevel
    ) -> list[TopicNode]:
        """Get topics filtered by difficulty."""
        return [t for t in self._topics if t.subject == subject and t.difficulty == difficulty]

    def get_topic_graph(self, subject: str) -> dict[str, list[str]]:
        """Get topic dependency graph."""
        topics = self.get_topics_for_subject(subject)
        graph = {}
        for topic in topics:
            graph[topic.id] = topic.prerequisites or []
        return graph

    def get_leaf_topics(self, subject: str) -> list[TopicNode]:
        """Get topics with no prerequisites (starting points)."""
        all_ids = {t.id for t in self._topics}
        leaf_topics = []
        for topic in self._topics:
            if topic.subject == subject:
                prereqs = set(topic.prerequisites or [])
                if not prereqs.issubset(all_ids):
                    leaf_topics.append(topic)
        return leaf_topics


class CheckpointGenerator:
    """Generates active recall checkpoints and quiz questions."""

    def __init__(self):
        self._question_templates = {
            ActivityType.ACTIVE_RECALL: [
                "Explain the key concept of {topic} in 2-3 sentences.",
                "What are the main applications of {topic}?",
                "Compare and contrast {topic} with related concepts.",
                "Describe a real-world example where {topic} is important.",
                "What are the common misconceptions about {topic}?",
            ],
            ActivityType.MINI_QUIZ: [
                {
                    "question": "What is the primary purpose of {topic}?",
                    "options": [
                        "A) Purpose A",
                        "B) Purpose B",
                        "C) Purpose C",
                        "D) Purpose D",
                    ],
                    "answer": "A",
                    "marks": 1,
                },
                {
                    "question": "Which statement about {topic} is correct?",
                    "options": [
                        "A) Statement A",
                        "B) Statement B",
                        "C) Statement C",
                        "D) Statement D",
                    ],
                    "answer": "B",
                    "marks": 1,
                },
            ],
        }

    def generate_checkpoint(self, topic: TopicNode, activity_type: ActivityType) -> str | dict:
        """Generate a checkpoint question for a topic."""
        if activity_type == ActivityType.ACTIVE_RECALL:
            template = random.choice(self._question_templates[activity_type])
            return template.format(topic=topic.name)

        elif activity_type == ActivityType.MINI_QUIZ:
            questions = []
            for _ in range(3):
                q_template = random.choice(self._question_templates[ActivityType.MINI_QUIZ])
                q = {
                    "question": q_template["question"].format(topic=topic.name),
                    "options": q_template["options"],
                    "answer": q_template["answer"],
                    "marks": q_template["marks"],
                }
                questions.append(q)
            return questions

        return f"Review the key concepts of {topic.name} and explain them in your own words."

    def generate_quiz(self, topics: list[TopicNode]) -> MiniQuiz:
        """Generate a mini-quiz covering multiple topics."""
        questions = []
        total_marks = 0

        for topic in topics[:3]:  # Limit to 3 topics
            checkpoint_q = self.generate_checkpoint(topic, ActivityType.ACTIVE_RECALL)
            questions.append(
                {
                    "type": "short_answer",
                    "question": f"Explain {topic.name}: {checkpoint_q}",
                    "topic_id": topic.id,
                    "marks": 2,
                }
            )
            total_marks += 2

            if random.random() > 0.5:
                mcq = self.generate_checkpoint(topic, ActivityType.MINI_QUIZ)
                questions.append(
                    {
                        "type": "multiple_choice",
                        "question": mcq[0]["question"],
                        "options": mcq[0]["options"],
                        "answer": mcq[0]["answer"],
                        "topic_id": topic.id,
                        "marks": 1,
                    }
                )
                total_marks += 1

        return MiniQuiz(
            id=f"quiz_{int(time.time())}",
            topic_ids=[t.id for t in topics],
            questions=questions,
            total_marks=total_marks,
            time_limit_minutes=10,
        )


class SprintPlanner:
    """
    Generates 21-day study sprint plans for syllabus coverage.

    Features:
    - Bird's-eye map of all syllabus topics
    - Logical ordering (prerequisites first)
    - Daily time blocks with varied activities
    - Active recall checkpoints
    - Mini-quizzes at end of each day
    """

    def __init__(self, kb_name: str = "default"):
        self.kb_name = kb_name
        self.extractor = SyllabusExtractor(kb_name)
        self.checkpoint_gen = CheckpointGenerator()

    def create_plan(
        self,
        subjects: list[str],
        days: int = 21,
        daily_hours: float = 2.0,
        start_date: Optional[str] = None,
        name: Optional[str] = None,
    ) -> SprintPlan:
        """
        Create a complete sprint plan.

        Args:
            subjects: List of subjects to cover
            days: Number of days in the sprint (default 21)
            daily_hours: Daily study time in hours
            start_date: Plan start date (ISO format, defaults to today)
            name: Optional plan name

        Returns:
            Complete SprintPlan with day-by-day schedule
        """
        plan_id = f"sprint_{int(time.time())}"
        start = datetime.fromisoformat(start_date or datetime.now().isoformat())

        all_topics: list[TopicNode] = []
        for subject in subjects:
            all_topics.extend(self.extractor.get_topics_for_subject(subject))

        topic_coverage = {t.id: False for t in all_topics}

        daily_minutes = int(daily_hours * 60)
        day_duration = daily_minutes // 4  # 4 time blocks per day

        days_list: list[SprintDay] = []
        topic_index = 0
        subjects_cycle = subjects * ((days // len(subjects)) + 1)

        for day_num in range(1, days + 1):
            date = (start + timedelta(days=day_num - 1)).isoformat()[:10]
            subject = subjects_cycle[day_num - 1]

            subject_topics = self.extractor.get_topics_for_subject(subject)
            remaining = [t for t in subject_topics if not topic_coverage.get(t.id, False)]

            if not remaining:
                remaining = subject_topics

            topics_today = remaining[:2]  # Cover 2 topics per day
            for t in topics_today:
                topic_coverage[t.id] = True

            activities = self._create_activities(topics_today, day_duration)
            quiz = self.checkpoint_gen.generate_quiz(topics_today)
            checkpoint = self._create_daily_checkpoint(topics_today)

            day = SprintDay(
                day_number=day_num,
                date=date,
                subject=subject,
                topics_covered=[t.id for t in topics_today],
                activities=activities,
                quiz=quiz,
                checkpoint=checkpoint,
            )
            days_list.append(day)

        plan = SprintPlan(
            id=plan_id,
            name=name or f"{'-'.join(subjects).title()} Sprint",
            subjects=subjects,
            total_days=days,
            daily_hours=daily_hours,
            start_date=start.isoformat()[:10],
            end_date=(start + timedelta(days=days)).isoformat()[:10],
            days=days_list,
            topic_coverage=topic_coverage,
        )

        return plan

    def _create_activities(
        self, topics: list[TopicNode], block_minutes: int
    ) -> list[LearningActivity]:
        """Create learning activities for a day."""
        activities = []

        for i, topic in enumerate(topics):
            activities.append(
                LearningActivity(
                    id=f"act_{int(time.time())}_{i}",
                    type=ActivityType.PRE_READ,
                    title=f"Read: {topic.name}",
                    description=f"Review the main concepts of {topic.name}",
                    duration_minutes=block_minutes,
                    topic_id=topic.id,
                    checkpoint_question=self.checkpoint_gen.generate_checkpoint(
                        topic, ActivityType.ACTIVE_RECALL
                    ),
                )
            )

            activities.append(
                LearningActivity(
                    id=f"act_{int(time.time())}_{i + 10}",
                    type=ActivityType.PRACTICE_PROBLEMS,
                    title=f"Practice: {topic.name}",
                    description=f"Solve 3-5 practice problems on {topic.name}",
                    duration_minutes=block_minutes,
                    topic_id=topic.id,
                )
            )

        activities.append(
            LearningActivity(
                id=f"act_{int(time.time())}_recall",
                type=ActivityType.ACTIVE_RECALL,
                title="Active Recall Session",
                description="Without looking at notes, recall and write down key concepts from today",
                duration_minutes=block_minutes,
                topic_id=topics[0].id if topics else "",
            )
        )

        return activities

    def _create_daily_checkpoint(self, topics: list[TopicNode]) -> dict:
        """Create end-of-day checkpoint."""
        questions = []
        for topic in topics:
            questions.append(
                {
                    "topic": topic.name,
                    "question": f"Rate your understanding of {topic.name} (1-5):",
                    "type": "self-assessment",
                }
            )

        return {
            "type": "daily_checkpoint",
            "questions": questions,
            "duration_minutes": 5,
        }

    def get_topic_coverage_summary(self, plan: SprintPlan) -> dict:
        """Get summary of topics covered in the plan."""
        covered = sum(1 for v in plan.topic_coverage.values() if v)
        total = len(plan.topic_coverage)

        subject_coverage = {}
        for subject in plan.subjects:
            subject_topics = self.extractor.get_topics_for_subject(subject)
            covered_count = sum(1 for t in subject_topics if plan.topic_coverage.get(t.id, False))
            subject_coverage[subject] = {
                "covered": covered_count,
                "total": len(subject_topics),
                "percentage": (
                    (covered_count / len(subject_topics) * 100) if subject_topics else 0
                ),
            }

        return {
            "total_covered": covered,
            "total_topics": total,
            "overall_percentage": (covered / total * 100) if total > 0 else 0,
            "by_subject": subject_coverage,
        }


def create_sprint_plan(
    subjects: list[str],
    days: int = 21,
    daily_hours: float = 2.0,
    start_date: Optional[str] = None,
    kb_name: str = "default",
) -> SprintPlan:
    """
    Convenience function to create a sprint plan.

    Usage:
        plan = create_sprint_plan(
            subjects=["mathematics", "physics"],
            days=21,
            daily_hours=2.0,
        )
    """
    planner = SprintPlanner(kb_name=kb_name)
    return planner.create_plan(
        subjects=subjects,
        days=days,
        daily_hours=daily_hours,
        start_date=start_date,
    )


__all__ = [
    "SprintPlanner",
    "SprintPlan",
    "SprintDay",
    "TopicNode",
    "LearningActivity",
    "MiniQuiz",
    "Subject",
    "DifficultyLevel",
    "ActivityType",
    "create_sprint_plan",
]
