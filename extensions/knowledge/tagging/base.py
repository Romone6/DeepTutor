"""
Question Archetype and Error Trap Tagging System
=================================================

Provides tagging for past-paper questions with:
- Archetype tags: Question types (calculation, definition, extended response, etc.)
- Error trap tags: Common student mistakes to avoid

Usage:
    from extensions.knowledge.tagging import (
        QuestionArchetype,
        ErrorTrap,
        TagSet,
        TagManager,
    )

    tags = TagSet(
        archetypes=[QuestionArchetype.CALCULATION],
        error_traps=[ErrorTrap.SIGN_ERROR],
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class QuestionArchetype(str, Enum):
    """
    Past-paper question archetypes/patterns.

    Each archetype has specific characteristics that students need to recognize.
    """

    CALCULATION = "calculation"
    """Numerical computation, formula application"""

    DEFINITION = "definition"
    """Define a term or concept"""

    DEFINITION_APPLICATION = "definition_application"
    """Define then apply to specific case"""

    EXPLAIN_ANALYSE = "explain_analyse"
    """Explain a process and analyse its components"""

    DISCUSS_EVALUATE = "discuss_evaluate"
    """Discuss multiple perspectives and evaluate"""

    COMPARE_CONTRAST = "compare_contrast"
    """Compare and contrast two things"""

    DIAGRAM_LABEL = "diagram_label"
    """Label or interpret a diagram"""

    CASE_STUDY = "case_study"
    """Apply knowledge to a specific case study"""

    EXTENDED_RESPONSE = "extended_response"
    """Long-form written response (6-10 marks)"""

    MULTI_STEP = "multi_step"
    """Requires multiple steps, often with sub-parts"""

    DATA_INTERPRETATION = "data_interpretation"
    """Interpret graphs, tables, or experimental data"""

    HYPOTHESIS_TESTING = "hypothesis_testing"
    """Design or evaluate experiments"""

    APPLICATION = "application"
    """Apply concepts to new situations"""

    SYNTHESIS = "synthesis"
    """Combine multiple concepts"""

    CRITICAL_THINKING = "critical_thinking"
    """Evaluate arguments or evidence"""

    PROCEDURE_DESCRIPTION = "procedure_description"
    """Describe experimental procedures"""

    GRAPH_CONSTRUCTION = "graph_construction"
    """Construct or complete graphs"""

    EQUATION_BALANCING = "equation_balancing"
    """Balance chemical or physics equations"""

    KEY_TERM_MATCHING = "key_term_matching"
    """Match key terms to definitions"""

    SHORT_ANSWER = "short_answer"
    """Brief 1-2 sentence answers"""


class ErrorTrap(str, Enum):
    """
    Common error traps in past-paper questions.

    Students often make these mistakes; tagging helps generate specific feedback.
    """

    SIGN_ERROR = "sign_error"
    """Forgot positive/negative signs"""

    UNIT_ERROR = "unit_error"
    """Wrong units or forgot to convert"""

    CALCULATION_ERROR = "calculation_error"
    """Arithmetic mistake"""

    FORMULA_MISUSE = "formula_misuse"
    """Used wrong formula or rearranged incorrectly"""

    CONCEPT_MISUNDERSTANDING = "concept_misunderstanding"
    """Fundamental concept error"""

    DIRECTION_ERROR = "direction_error"
    """Wrong direction in vectors/forces"""

    APPROXIMATION_ERROR = "approximation_error"
    """Rounded too early or too much"""

    VERBAL_ERROR = "verbal_error"
    """Used wrong word (e.g., 'affect' vs 'effect')"""

    DEFINITION_INCOMPLETE = "definition_incomplete"
    """Incomplete or partial definition"""

    STRUCTURE_ERROR = "structure_error"
    """Didn't follow required structure (e.g., PEEL for essays)"""

    DIAGRAM_LABEL_ERROR = "diagram_label_error"
    """Wrong labels or incorrect diagram interpretation"""

    DATA_MISREAD = "data_misread"
    """Misread graph or table data"""

    ASSUMPTION_MADE = "unnecessary_assumption"
    """Made unwarranted assumption"""

    CONTEXT_IGNORED = "context_ignored"
    """Ignored specific context in question"""

    EXPLANATION_TOO_BRIEF = "explanation_too_brief"
    """Answer too short for mark allocation"""

    LOGIC_GAP = "logic_gap"
    """Missing logical step in reasoning"""

    COUNTER_EXAMPLE_MISSED = "counter_example_missed"
    """Didn't provide required counter example"""

    MARK_ALLOC_IGNORED = "mark_alloc_ignored"
    """Didn't allocate enough detail for marks given"""


class DifficultyLevel(str, Enum):
    """Question difficulty levels."""

    EASY = "easy"
    """Straightforward application"""

    MEDIUM = "medium"
    """Requires some reasoning"""

    HARD = "hard"
    """Complex multi-step or synthesis"""

    EXTENSION = "extension"
    """Challenge/innovation level"""


@dataclass
class TagSet:
    """
    Complete tag set for a question or chunk.

    Attributes:
        archetypes: List of question archetypes
        error_traps: List of common error traps to avoid
        difficulty: Difficulty level
        tags: Additional custom tags
    """

    archetypes: list[QuestionArchetype] = field(default_factory=list)
    error_traps: list[ErrorTrap] = field(default_factory=list)
    difficulty: Optional[DifficultyLevel] = None
    marks: Optional[int] = None
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "archetypes": [a.value for a in self.archetypes],
            "error_traps": [e.value for e in self.error_traps],
            "difficulty": self.difficulty.value if self.difficulty else None,
            "marks": self.marks,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TagSet":
        """Create TagSet from dictionary."""
        return cls(
            archetypes=[QuestionArchetype(a) for a in data.get("archetypes", [])],
            error_traps=[ErrorTrap(e) for e in data.get("error_traps", [])],
            difficulty=DifficultyLevel(data["difficulty"]) if data.get("difficulty") else None,
            marks=data.get("marks"),
            tags=data.get("tags", []),
        )

    def has_archetype(self, archetype: QuestionArchetype) -> bool:
        """Check if tag set has a specific archetype."""
        return archetype in self.archetypes

    def has_error_trap(self, trap: ErrorTrap) -> bool:
        """Check if tag set mentions a specific error trap."""
        return trap in self.error_traps

    def get_archetype_labels(self) -> list[str]:
        """Get human-readable archetype labels."""
        labels = {
            QuestionArchetype.CALCULATION: "Calculation",
            QuestionArchetype.DEFINITION: "Definition",
            QuestionArchetype.DEFINITION_APPLICATION: "Definition + Application",
            QuestionArchetype.EXPLAIN_ANALYSE: "Explain/Analyse",
            QuestionArchetype.DISCUSS_EVALUATE: "Discuss/Evaluate",
            QuestionArchetype.COMPARE_CONTRAST: "Compare & Contrast",
            QuestionArchetype.DIAGRAM_LABEL: "Diagram/Label",
            QuestionArchetype.CASE_STUDY: "Case Study",
            QuestionArchetype.EXTENDED_RESPONSE: "Extended Response",
            QuestionArchetype.MULTI_STEP: "Multi-step",
            QuestionArchetype.DATA_INTERPRETATION: "Data Interpretation",
            QuestionArchetype.HYPOTHESIS_TESTING: "Hypothesis Testing",
            QuestionArchetype.APPLICATION: "Application",
            QuestionArchetype.SYNTHESIS: "Synthesis",
            QuestionArchetype.CRITICAL_THINKING: "Critical Thinking",
            QuestionArchetype.PROCEDURE_DESCRIPTION: "Procedure Description",
            QuestionArchetype.GRAPH_CONSTRUCTION: "Graph Construction",
            QuestionArchetype.EQUATION_BALANCING: "Equation Balancing",
            QuestionArchetype.KEY_TERM_MATCHING: "Key Terms",
            QuestionArchetype.SHORT_ANSWER: "Short Answer",
        }
        return [labels.get(a, a.value) for a in self.archetypes]

    def get_error_trap_labels(self) -> list[str]:
        """Get human-readable error trap labels."""
        labels = {
            ErrorTrap.SIGN_ERROR: "Sign errors",
            ErrorTrap.UNIT_ERROR: "Unit errors",
            ErrorTrap.CALCULATION_ERROR: "Calculation errors",
            ErrorTrap.FORMULA_MISUSE: "Formula misuse",
            ErrorTrap.CONCEPT_MISUNDERSTANDING: "Concept misunderstanding",
            ErrorTrap.DIRECTION_ERROR: "Direction errors",
            ErrorTrap.APPROXIMATION_ERROR: "Approximation errors",
            ErrorTrap.VERBAL_ERROR: "Verbal errors",
            ErrorTrap.DEFINITION_INCOMPLETE: "Incomplete definitions",
            ErrorTrap.STRUCTURE_ERROR: "Structure errors",
            ErrorTrap.DIAGRAM_LABEL_ERROR: "Diagram label errors",
            ErrorTrap.DATA_MISREAD: "Data misreading",
            ErrorTrap.ASSUMPTION_MADE: "Unnecessary assumptions",
            ErrorTrap.CONTEXT_IGNORED: "Context ignored",
            ErrorTrap.EXPLANATION_TOO_BRIEF: "Too brief",
            ErrorTrap.LOGIC_GAP: "Logic gaps",
            ErrorTrap.COUNTER_EXAMPLE_MISSED: "Counter example missing",
            ErrorTrap.MARK_ALLOC_IGNORED: "Marks not matched",
        }
        return [labels.get(e, e.value) for e in self.error_traps]


@dataclass
class TaggedQuestion:
    """
    A past-paper question with full tagging information.
    """

    question_id: str
    content: str
    source: str
    year: int
    tags: TagSet
    model_answer: Optional[str] = None
    marking_notes: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "content": self.content,
            "source": self.source,
            "year": self.year,
            "tags": self.tags.to_dict(),
            "model_answer": self.model_answer,
            "marking_notes": self.marking_notes,
        }


class TagManager:
    """
    Manager for question tagging operations.
    """

    def __init__(self, tags_file: Optional[str] = None):
        """
        Initialize tag manager.

        Args:
            tags_file: Optional path to JSON file with tagged questions
        """
        self._tagged_questions: dict[str, TaggedQuestion] = {}
        self._tags_file = tags_file

        if tags_file:
            self.load_from_file(tags_file)

    def add_tagged_question(self, question: TaggedQuestion) -> None:
        """Add a tagged question."""
        self._tagged_questions[question.question_id] = question

    def get_question(self, question_id: str) -> Optional[TaggedQuestion]:
        """Get a tagged question by ID."""
        return self._tagged_questions.get(question_id)

    def get_by_archetype(self, archetype: QuestionArchetype) -> list[TaggedQuestion]:
        """Get all questions with a specific archetype."""
        return [q for q in self._tagged_questions.values() if q.tags.has_archetype(archetype)]

    def get_by_error_trap(self, trap: ErrorTrap) -> list[TaggedQuestion]:
        """Get all questions related to a specific error trap."""
        return [q for q in self._tagged_questions.values() if q.tags.has_error_trap(trap)]

    def get_practice_set(
        self,
        archetype: Optional[QuestionArchetype] = None,
        difficulty: Optional[DifficultyLevel] = None,
        count: int = 5,
    ) -> list[TaggedQuestion]:
        """
        Get a practice set of questions.

        Args:
            archetype: Filter by archetype (None for mixed)
            difficulty: Filter by difficulty
            count: Number of questions to return

        Returns:
            List of tagged questions for practice
        """
        candidates = list(self._tagged_questions.values())

        if archetype:
            candidates = [q for q in candidates if q.tags.has_archetype(archetype)]

        if difficulty:
            candidates = [q for q in candidates if q.tags.difficulty == difficulty]

        return candidates[:count]

    def get_archetype_distribution(self) -> dict[str, int]:
        """Get distribution of archetypes in tagged questions."""
        distribution = {}
        for q in self._tagged_questions.values():
            for archetype in q.tags.archetypes:
                distribution[archetype.value] = distribution.get(archetype.value, 0) + 1
        return distribution

    def save_to_file(self, path: Optional[str] = None) -> None:
        """Save tagged questions to JSON file."""
        import json

        file_path = path or self._tags_file
        if not file_path:
            raise ValueError("No file path specified")

        data = {question_id: q.to_dict() for question_id, q in self._tagged_questions.items()}

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, path: str) -> None:
        """Load tagged questions from JSON file."""
        import json

        self._tags_file = path

        with open(path) as f:
            data = json.load(f)

        for question_id, q_data in data.items():
            tags = TagSet.from_dict(q_data["tags"])
            self._tagged_questions[question_id] = TaggedQuestion(
                question_id=question_id,
                content=q_data["content"],
                source=q_data["source"],
                year=q_data["year"],
                tags=tags,
                model_answer=q_data.get("model_answer"),
                marking_notes=q_data.get("marking_notes"),
            )


def suggest_archetype(content: str) -> list[QuestionArchetype]:
    """
    Suggest possible archetypes based on question content.

    This is a heuristic helper, not definitive classification.
    """
    import re

    content_lower = content.lower()
    suggestions = []

    calculation_indicators = [
        r"\d+\s*[\+\-\*\/]\s*\d+",
        r"calculate",
        r"compute",
        r"solve",
        r"find the (value|answer|result)",
        r"what is \d+",
    ]
    if any(re.search(pattern, content_lower) for pattern in calculation_indicators):
        suggestions.append(QuestionArchetype.CALCULATION)

    definition_indicators = [
        r"define",
        r"what is",
        r"explain the meaning",
        r"state",
    ]
    if any(re.search(pattern, content_lower) for pattern in definition_indicators):
        if "apply" in content_lower or "example" in content_lower:
            suggestions.append(QuestionArchetype.DEFINITION_APPLICATION)
        else:
            suggestions.append(QuestionArchetype.DEFINITION)

    multi_step_indicators = [
        r"\(a\)",
        r"\(b\)",
        r"\(i\)",
        r"\(ii\)",
        r"part \d",
        r"step \d",
    ]
    if any(re.search(pattern, content_lower) for pattern in multi_step_indicators):
        suggestions.append(QuestionArchetype.MULTI_STEP)

    extended_indicators = [
        r"\d+\s*marks?",
        r"explain (why|how)",
        r"discuss",
        r"evaluate",
        r"analyse",
        r"consider",
        r"compare",
        r"contrast",
    ]
    if any(re.search(pattern, content_lower) for pattern in extended_indicators):
        marks_match = re.search(r"(\d+)\s*marks?", content_lower)
        has_marks = marks_match is not None
        marks_count = int(marks_match.group(1)) if has_marks and marks_match else 0

        if has_marks and marks_count >= 5:
            suggestions.append(QuestionArchetype.EXTENDED_RESPONSE)
        elif "compare" in content_lower or "contrast" in content_lower:
            suggestions.append(QuestionArchetype.COMPARE_CONTRAST)
        elif "discuss" in content_lower or "evaluate" in content_lower:
            suggestions.append(QuestionArchetype.DISCUSS_EVALUATE)
        elif "explain" in content_lower:
            suggestions.append(QuestionArchetype.EXPLAIN_ANALYSE)
        elif has_marks:
            suggestions.append(QuestionArchetype.EXTENDED_RESPONSE)

    data_indicators = [
        r"figure \d",
        r"graph",
        r"table",
        r"data",
        r"shown (above|in fig)",
    ]
    if any(re.search(pattern, content_lower) for pattern in data_indicators):
        suggestions.append(QuestionArchetype.DATA_INTERPRETATION)

    diagram_indicators = [
        r"fig\.?\s*\d",
        r"diagram",
        r"draw",
        r"sketch",
        r"label",
    ]
    if any(re.search(pattern, content_lower) for pattern in diagram_indicators):
        suggestions.append(QuestionArchetype.DIAGRAM_LABEL)

    return suggestions if suggestions else [QuestionArchetype.SHORT_ANSWER]


def suggest_error_traps(content: str, archetype: QuestionArchetype) -> list[ErrorTrap]:
    """
    Suggest relevant error traps based on question content and archetype.
    """
    import re

    content_lower = content.lower()
    suggestions = []

    if QuestionArchetype.CALCULATION in archetype:
        if "sign" in content_lower or "negative" in content_lower:
            suggestions.append(ErrorTrap.SIGN_ERROR)
        if "unit" in content_lower or "convert" in content_lower:
            suggestions.append(ErrorTrap.UNIT_ERROR)
        suggestions.append(ErrorTrap.CALCULATION_ERROR)
        suggestions.append(ErrorTrap.FORMULA_MISUSE)

    if (
        QuestionArchetype.DEFINITION in archetype
        or QuestionArchetype.DEFINITION_APPLICATION in archetype
    ):
        suggestions.append(ErrorTrap.DEFINITION_INCOMPLETE)
        suggestions.append(ErrorTrap.VERBAL_ERROR)

    if (
        QuestionArchetype.EXTENDED_RESPONSE in archetype
        or QuestionArchetype.DISCUSS_EVALUATE in archetype
    ):
        suggestions.append(ErrorTrap.STRUCTURE_ERROR)
        suggestions.append(ErrorTrap.EXPLANATION_TOO_BRIEF)
        suggestions.append(ErrorTrap.LOGIC_GAP)

    if QuestionArchetype.DATA_INTERPRETATION in archetype:
        suggestions.append(ErrorTrap.DATA_MISREAD)

    if "marks" in content_lower:
        match = re.search(r"(\d+)\s*marks?", content_lower)
        if match:
            marks = int(match.group(1))
            if marks >= 5:
                suggestions.append(ErrorTrap.MARK_ALLOC_IGNORED)

    return suggestions


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
