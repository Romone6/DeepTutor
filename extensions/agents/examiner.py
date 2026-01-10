"""
Examiner Agent for HSC-style marking and feedback.

Provides strict marking feedback for practice answers and extended responses,
including mark breakdowns, earned/lost marks, ranked fix list, and rewrite targets.

Usage:
    from extensions.agents import Examiner, MarkingResult

    examiner = Examiner()
    result = examiner.evaluate(
        student_answer="The answer text...",
        marking_guidelines=[Snippet(...), ...],
        subject="mathematics",
        policy_config=policy_config,
    )
    print(result.mark_breakdown)
    print(result.rewrite_target)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from extensions.agents.band6_rubrics import (
    evaluate_answer as evaluate_rubric_answer,
    load_rubric,
    Band6RubricResult,
)
from extensions.agents.bio_answer_guardrails import (
    validate_bio_answer,
    BioGuardrailResult,
)
from extensions.agents.business_answer_guardrails import (
    validate_business_answer,
    BusinessGuardrailResult,
)
from extensions.agents.legal_irac_enforcer import (
    validate_irac_structure,
    LegalIRACResult,
)
from extensions.agents.legal_authority_picker import (
    pick_authorities,
    validate_all_citations,
    get_authorities_panel_data,
    suggest_authorities_for_topic,
    AuthorityCandidate,
)
from extensions.agents.english_module_engine import (
    generate_english_response,
    check_quote_discipline,
    evaluate_marker_lens,
    extract_quotes_from_kb,
    EnglishModuleResult,
)
from extensions.agents.command_terms import (
    evaluate_answer_structure,
    generate_scaffold,
    get_command_term_info,
    resolve_command_term,
)
from extensions.knowledge.tagging.base import (
    QuestionArchetype,
    ErrorTrap,
    suggest_archetype,
    suggest_error_traps,
)


class IntentType(str, Enum):
    """Types of practice/intent that trigger examination."""

    PRACTICE = "practice"
    MARK = "mark"
    EXAM = "exam"
    SELF_ASSESS = "self_assess"
    GENERAL = "general"


class MarkingCriterion(BaseModel):
    """A single marking criterion from guidelines."""

    criterion_id: str
    description: str
    max_marks: float
    key_elements: list[str] = Field(default_factory=list)
    common_errors: list[str] = Field(default_factory=list)


class MarkAllocation(BaseModel):
    """Allocation of marks for a criterion."""

    criterion_id: str
    criterion_description: str
    marks_earned: float
    marks_possible: float
    justification: str
    evidence_from_answer: str = ""


class FixItem(BaseModel):
    """A ranked fix item for student improvement."""

    rank: int
    severity: str  # "critical", "major", "minor"
    criterion_id: str
    issue: str
    fix: str
    marks_impact: float  # Positive if fixing earns marks, negative if current issue loses marks


class MarkBreakdown(BaseModel):
    """Complete mark breakdown for an answer."""

    total_marks: float
    possible_marks: float
    percentage: float
    grade: str  # "A", "B", "C", "D", "E" or "HD", "D", "C", "P", "F"
    passed: bool
    allocations: list[MarkAllocation] = Field(default_factory=list)


class RewriteTarget(BaseModel):
    """Target for rewriting the answer."""

    focus_areas: list[str] = Field(default_factory=list)
    key_improvements: list[str] = Field(default_factory=list)
    template: str = ""
    word_count_guidance: str = ""
    structural_suggestions: list[str] = Field(default_factory=list)


class MarkingResult(BaseModel):
    """Complete marking result for an answer."""

    mark_breakdown: MarkBreakdown
    earned_marks_summary: str = ""
    lost_marks_summary: str = ""
    fix_list: list[FixItem] = Field(default_factory=list)
    rewrite_target: RewriteTarget = Field(default_factory=lambda: RewriteTarget())
    overall_feedback: str = ""
    subject: str = ""
    topic: str = ""
    policy_id: str = ""
    quality_score: float = 0.0
    archetype_tags: list[str] = Field(default_factory=list)
    error_traps_found: list[str] = Field(default_factory=list)
    rubric_result: Optional[dict[str, Any]] = None
    band_estimate: str = ""
    top_fixes: list[dict[str, str]] = Field(default_factory=list)
    bio_guardrails_result: Optional[dict[str, Any]] = None
    business_guardrails_result: Optional[dict[str, Any]] = None
    legal_irac_result: Optional[dict[str, Any]] = None
    legal_authorities_panel: Optional[dict[str, Any]] = None
    hallucinated_citations: list[str] = Field(default_factory=list)
    english_marker_lens: Optional[dict[str, Any]] = None
    english_quote_discipline: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mark_breakdown": self.mark_breakdown.model_dump(),
            "earned_marks_summary": self.earned_marks_summary,
            "lost_marks_summary": self.lost_marks_summary,
            "fix_list": [f.model_dump() for f in self.fix_list],
            "rewrite_target": self.rewrite_target.model_dump(),
            "overall_feedback": self.overall_feedback,
            "subject": self.subject,
            "topic": self.topic,
            "policy_id": self.policy_id,
            "quality_score": self.quality_score,
            "archetype_tags": self.archetype_tags,
            "error_traps_found": self.error_traps_found,
            "rubric_result": self.rubric_result,
            "band_estimate": self.band_estimate,
            "top_fixes": self.top_fixes,
            "bio_guardrails_result": self.bio_guardrails_result,
            "business_guardrails_result": self.business_guardrails_result,
            "legal_irac_result": self.legal_irac_result,
            "legal_authorities_panel": self.legal_authorities_panel,
            "hallucinated_citations": self.hallucinated_citations,
            "english_marker_lens": self.english_marker_lens,
            "english_quote_discipline": self.english_quote_discipline,
        }


class AnswerEvaluation(BaseModel):
    """Evaluation of a single answer against a criterion."""

    criterion_id: str
    content_found: bool
    relevant_excerpts: list[str] = Field(default_factory=list)
    missing_elements: list[str] = Field(default_factory=list)
    incorrect_elements: list[str] = Field(default_factory=list)
    partial_elements: list[str] = Field(default_factory=list)
    marks_estimate: float = 0.0


@dataclass
class Examiner:
    """
    Examiner agent for HSC-style marking and feedback.

    Takes student answers and marking guidelines to produce:
    - Mark breakdown
    - What earned marks
    - What lost marks
    - Ranked fix list
    - Rewrite target
    """

    def __init__(self):
        pass

    def should_examine(
        self,
        intent: IntentType | str | None,
        policy_id: str | None,
        policy_strictness: float = 0.5,
    ) -> bool:
        """
        Determine if examination should be triggered.

        Args:
            intent: The detected user intent
            policy_id: Current policy mode ID
            policy_strictness: Evaluation strictness from policy (0.0-1.0)

        Returns:
            True if examination should be performed
        """
        if intent in (
            IntentType.PRACTICE,
            IntentType.MARK,
            IntentType.EXAM,
            IntentType.SELF_ASSESS,
        ):
            return True

        if policy_id == "exam":
            return True

        if policy_strictness >= 0.8:
            return True

        return False

    def evaluate(
        self,
        student_answer: str,
        marking_guidelines: list[dict[str, Any]],
        subject: str = "",
        topic: str = "",
        policy_id: str = "exam",
        evaluation_strictness: float = 0.95,
        pass_threshold: float = 0.7,
        archetype_tags: Optional[list[str]] = None,
        error_traps_found: Optional[list[str]] = None,
        question_content: str = "",
        command_term: Optional[str] = None,
    ) -> MarkingResult:
        """
        Evaluate a student answer against marking guidelines.

        Args:
            student_answer: The student's written answer
            marking_guidelines: Retrieved marking guideline snippets
                Each should have: text, metadata (criterion_id, max_marks, etc.)
            subject: Subject name (e.g., "mathematics", "chemistry")
            topic: Topic name (e.g., "calculus", "organic_chemistry")
            policy_id: Policy mode ID for context
            evaluation_strictness: How strict the evaluation is (0.0-1.0)
            pass_threshold: Minimum percentage to pass (0.0-1.0)
            archetype_tags: Optional list of question archetype tags
            error_traps_found: Optional list of detected error traps
            question_content: Original question content for archetype detection

        Returns:
            MarkingResult with full breakdown
        """
        criteria = self._parse_guidelines(marking_guidelines)
        evaluations = self._evaluate_answer(student_answer, criteria)
        allocations = self._allocate_marks(evaluations, criteria, evaluation_strictness)
        breakdown = self._compute_breakdown(allocations, pass_threshold)
        earned_summary, lost_summary = self._generate_summaries(allocations)

        effective_archetypes = archetype_tags if archetype_tags else []
        effective_error_traps = error_traps_found if error_traps_found else []

        effective_command_term = command_term
        if not effective_command_term and question_content:
            effective_command_term = self._extract_command_term(question_content)

        if not effective_archetypes and question_content:
            suggested = suggest_archetype(question_content)
            effective_archetypes = [a.value for a in suggested]

        if not effective_error_traps and effective_archetypes and question_content:
            archetypes = [QuestionArchetype(a) for a in effective_archetypes]
            for archetype in archetypes:
                suggested_traps = suggest_error_traps(question_content, archetype)
                effective_error_traps.extend([t.value for t in suggested_traps])
            effective_error_traps = list(set(effective_error_traps))

        fix_list = self._generate_fix_list(
            allocations, student_answer, effective_error_traps, effective_command_term, subject
        )
        rewrite_target = self._generate_rewrite_target(
            allocations,
            student_answer,
            subject,
            topic,
            criteria,
            effective_archetypes,
            effective_command_term,
        )
        overall_feedback = self._generate_overall_feedback(
            breakdown,
            subject,
            effective_archetypes,
            effective_error_traps,
            effective_command_term,
            student_answer,
        )

        rubric_result = None
        band_estimate = ""
        rubric_fixes = []
        if subject:
            rubric = load_rubric(subject)
            if rubric:
                rubric_eval = evaluate_rubric_answer(student_answer, subject)
                if rubric_eval:
                    rubric_result = rubric_eval.to_dict()
                    band_estimate = rubric_eval.band_estimate
                    rubric_fixes = [
                        {
                            "dimension": f["dimension"],
                            "criterion": f["criterion"],
                            "issue": f["issue"],
                            "fix": f["fix"],
                            "priority": f["priority"],
                        }
                        for f in rubric_eval.top_fixes[:3]
                    ]

        bio_guardrails_result = None
        if subject == "biology" and student_answer:
            bio_result = validate_bio_answer(
                question=question_content or "",
                answer=student_answer,
                topic=topic or "cells_and_molecules",
            )
            bio_guardrails_result = {
                "is_acceptable": bio_result.is_acceptable,
                "score": bio_result.score,
                "vague_sentences": bio_result.vague_sentences,
                "missing_keywords": bio_result.missing_keywords,
                "missing_links": bio_result.missing_links,
                "process_issues": bio_result.process_issues,
                "rewrite_suggestions": bio_result.rewrite_suggestions,
                "keywords_found": bio_result.keywords_found,
            }

        business_guardrails_result = None
        if subject == "business_studies" and student_answer:
            business_result = validate_business_answer(
                question=question_content or "",
                answer=student_answer,
                topic=topic or "operations",
            )
            business_guardrails_result = {
                "is_acceptable": business_result.is_acceptable,
                "score": business_result.score,
                "missing_concepts": business_result.missing_concepts,
                "missing_evidence": business_result.missing_evidence,
                "missing_judgement": business_result.missing_judgement,
                "missing_syllabus_links": business_result.missing_syllabus_links,
                "structure_issues": business_result.structure_issues,
                "rewrite_suggestions": business_result.rewrite_suggestions,
                "concepts_found": business_result.concepts_found,
                "evidence_found": business_result.evidence_found,
                "exemplar_paragraph": business_result.exemplar_paragraph,
            }

        legal_irac_result = None
        legal_authorities_panel = None
        hallucinated_citations = []
        if subject == "legal_studies" and student_answer:
            irac_result = validate_irac_structure(
                question=question_content or "",
                answer=student_answer,
                topic=topic or "crime",
            )
            legal_irac_result = {
                "is_acceptable": irac_result.is_acceptable,
                "score": irac_result.score,
                "has_issue": irac_result.has_issue,
                "has_rule_law": irac_result.has_rule_law,
                "has_application": irac_result.has_application,
                "has_conclusion": irac_result.has_conclusion,
                "structure_score": irac_result.structure_score,
                "citation_count": irac_result.citation_count,
                "missing_citations": irac_result.missing_citations,
                "structure_issues": irac_result.structure_issues,
                "rewrite_suggestions": irac_result.rewrite_suggestions,
                "irac_scaffold": irac_result.irac_scaffold,
            }

            retrieved_authorities = marking_guidelines if marking_guidelines else []
            authorities = pick_authorities(
                question=question_content or "",
                retrieved_snippets=retrieved_authorities,
                topic=topic or "crime",
                max_authorities=5,
            )
            legal_authorities_panel = get_authorities_panel_data(authorities)

            citation_validations = validate_all_citations(student_answer, authorities)
            hallucinated_citations = [
                v.citation_text for v in citation_validations if v.hallucinated
            ]

        english_marker_lens = None
        english_quote_discipline = None
        if subject == "english_advanced" and student_answer:
            marker_result = evaluate_marker_lens(
                answer=student_answer,
                thesis="",
                question=question_content or "",
                text=topic or "the text",
                module="module_b",
            )
            english_marker_lens = {
                "conceptual_depth": marker_result.conceptual_depth,
                "cohesion": marker_result.cohesion,
                "textual_integrity": marker_result.textual_integrity,
                "audience_purpose": marker_result.audience_purpose,
                "overall_score": marker_result.overall_score,
                "strengths": marker_result.strengths,
                "improvements": marker_result.improvements,
            }

            kb_quotes = extract_quotes_from_kb(marking_guidelines if marking_guidelines else [])
            quote_discipline = check_quote_discipline(student_answer, kb_quotes)
            english_quote_discipline = quote_discipline

        return MarkingResult(
            mark_breakdown=breakdown,
            earned_marks_summary=earned_summary,
            lost_marks_summary=lost_summary,
            fix_list=fix_list[:10],
            rewrite_target=rewrite_target,
            overall_feedback=overall_feedback,
            subject=subject,
            topic=topic,
            policy_id=policy_id,
            quality_score=self._compute_quality_score(allocations, student_answer),
            archetype_tags=effective_archetypes,
            error_traps_found=effective_error_traps,
            rubric_result=rubric_result,
            band_estimate=band_estimate,
            top_fixes=rubric_fixes,
            bio_guardrails_result=bio_guardrails_result,
            business_guardrails_result=business_guardrails_result,
            legal_irac_result=legal_irac_result,
            legal_authorities_panel=legal_authorities_panel,
            hallucinated_citations=hallucinated_citations,
            english_marker_lens=english_marker_lens,
            english_quote_discipline=english_quote_discipline,
        )

    def _parse_guidelines(self, guidelines: list[dict[str, Any]]) -> list[MarkingCriterion]:
        """Parse marking guidelines into structured criteria."""
        criteria = []

        for i, guideline in enumerate(guidelines):
            metadata = guideline.get("metadata", {})
            text = guideline.get("text", "")

            criterion = MarkingCriterion(
                criterion_id=metadata.get("criterion_id", f"criterion_{i}"),
                description=text[:500] if text else "",
                max_marks=float(metadata.get("max_marks", 1.0)),
                key_elements=metadata.get("key_elements", []),
                common_errors=metadata.get("common_errors", []),
            )
            criteria.append(criterion)

        return criteria

    def _evaluate_answer(
        self, answer: str, criteria: list[MarkingCriterion]
    ) -> list[AnswerEvaluation]:
        """Evaluate answer against each criterion."""
        evaluations = []
        answer_lower = answer.lower()

        for criterion in criteria:
            eval_result = AnswerEvaluation(
                criterion_id=criterion.criterion_id,
                content_found=False,
            )

            found_elements = []
            missing_elements = []
            incorrect_elements = []

            for key_element in criterion.key_elements:
                if not key_element:
                    continue
                if key_element.lower() in answer_lower:
                    found_elements.append(key_element)
                else:
                    missing_elements.append(key_element)

            eval_result.content_found = len(found_elements) > 0
            eval_result.relevant_excerpts = found_elements[:3]
            eval_result.missing_elements = missing_elements
            eval_result.incorrect_elements = incorrect_elements

            if criterion.max_marks > 0:
                completeness = (
                    len(found_elements) / len(criterion.key_elements)
                    if criterion.key_elements
                    else 0.5
                )
                eval_result.marks_estimate = round(criterion.max_marks * completeness, 1)

            evaluations.append(eval_result)

        return evaluations

    def _allocate_marks(
        self,
        evaluations: list[AnswerEvaluation],
        criteria: list[MarkingCriterion],
        strictness: float,
    ) -> list[MarkAllocation]:
        """Allocate marks based on evaluations."""
        allocations = []
        criteria_dict = {c.criterion_id: c for c in criteria}

        for eval_result in evaluations:
            criterion = criteria_dict.get(eval_result.criterion_id)
            if not criterion:
                continue

            base_marks = eval_result.marks_estimate

            if strictness >= 0.9:
                adjustment = 0.9
            elif strictness >= 0.7:
                adjustment = 0.95
            else:
                adjustment = 1.0

            marks_earned = round(base_marks * adjustment, 1)
            marks_earned = min(marks_earned, criterion.max_marks)

            justification_parts = []
            if eval_result.missing_elements:
                justification_parts.append(
                    f"Missing key elements: {', '.join(eval_result.missing_elements[:2])}"
                )
            if eval_result.relevant_excerpts:
                justification_parts.append(f"Found: {', '.join(eval_result.relevant_excerpts[:2])}")

            allocation = MarkAllocation(
                criterion_id=eval_result.criterion_id,
                criterion_description=criterion.description[:200],
                marks_earned=marks_earned,
                marks_possible=criterion.max_marks,
                justification="; ".join(justification_parts)
                if justification_parts
                else "Partial credit awarded",
                evidence_from_answer=", ".join(eval_result.relevant_excerpts[:2]),
            )
            allocations.append(allocation)

        return allocations

    def _compute_breakdown(
        self, allocations: list[MarkAllocation], pass_threshold: float
    ) -> MarkBreakdown:
        """Compute overall mark breakdown."""
        total_marks = sum(a.marks_earned for a in allocations)
        possible_marks = sum(a.marks_possible for a in allocations)
        percentage = (total_marks / possible_marks * 100) if possible_marks > 0 else 0

        grade = self._percentage_to_grade(percentage)
        passed = percentage >= (pass_threshold * 100)

        return MarkBreakdown(
            total_marks=total_marks,
            possible_marks=possible_marks,
            percentage=round(percentage, 1),
            grade=grade,
            passed=passed,
            allocations=allocations,
        )

    def _percentage_to_grade(self, percentage: float) -> str:
        """Convert percentage to grade."""
        if percentage >= 90:
            return "HD"
        elif percentage >= 80:
            return "A"
        elif percentage >= 70:
            return "B"
        elif percentage >= 60:
            return "C"
        elif percentage >= 50:
            return "D"
        else:
            return "F"

    def _generate_summaries(self, allocations: list[MarkAllocation]) -> tuple[str, str]:
        """Generate summaries of earned and lost marks."""
        earned_items = []
        lost_items = []

        for alloc in allocations:
            if alloc.marks_earned >= alloc.marks_possible * 0.8:
                earned_items.append(
                    f"{alloc.criterion_id}: {alloc.marks_earned}/{alloc.marks_possible}"
                )
            elif alloc.marks_earned < alloc.marks_possible * 0.3:
                lost_marks = round(alloc.marks_possible - alloc.marks_earned, 1)
                if lost_marks > 0:
                    lost_items.append(f"{alloc.criterion_id}: -{lost_marks} marks")

        earned_summary = "; ".join(earned_items) if earned_items else "No full marks awarded"
        lost_summary = "; ".join(lost_items) if lost_items else "No significant marks lost"

        return earned_summary, lost_summary

    def _extract_command_term(self, question: str) -> Optional[str]:
        """Extract the command term from a question."""
        if not question:
            return None

        question_lower = question.lower()

        known_terms = [
            "analyse",
            "analyze",
            "evaluate",
            "explain",
            "discuss",
            "compare",
            "contrast",
            "describe",
            "define",
            "identify",
            "outline",
            "summarise",
            "summarize",
            "calculate",
            "assess",
            "critique",
            "examine",
            "interpret",
            "justify",
            "predict",
            "propose",
            "apply",
            "clarify",
            "distinguish",
            "explore",
            "formulate",
            "state",
        ]

        for term in known_terms:
            if f" {term} " in question_lower or question_lower.endswith(f" {term}"):
                return resolve_command_term(term)

        return None

    def _generate_fix_list(
        self,
        allocations: list[MarkAllocation],
        answer: str,
        error_traps: Optional[list[str]] = None,
        command_term: Optional[str] = None,
        subject: str = "",
    ) -> list[FixItem]:
        """Generate ranked list of fixes needed."""
        fixes = []
        rank = 1

        error_trap_messages = {
            "sign_error": "Check your signs - look for positive/negative errors",
            "unit_error": "Verify units and conversions",
            "calculation_error": "Recheck arithmetic calculations",
            "formula_misuse": "Ensure correct formula selection and rearrangement",
            "concept_misunderstanding": "Review fundamental concept understanding",
            "direction_error": "Check direction conventions (vectors/forces)",
            "approximation_error": "Avoid early or excessive rounding",
            "verbal_error": "Use precise terminology (e.g., 'affect' vs 'effect')",
            "definition_incomplete": "Provide complete definitions with key components",
            "structure_error": "Follow required structure (PEEL for essays)",
            "diagram_label_error": "Verify diagram labels and interpretations",
            "data_misread": "Double-check graph/table readings",
            "unnecessary_assumption": "Avoid unwarranted assumptions",
            "context_ignored": "Address all specific context in the question",
            "explanation_too_brief": "Expand answer to match mark allocation",
            "logic_gap": "Include all logical steps in reasoning",
            "counter_example_missed": "Provide required counter examples",
            "mark_alloc_ignored": "Match detail level to marks available",
        }

        for alloc in allocations:
            lost_percentage = (
                1 - (alloc.marks_earned / alloc.marks_possible) if alloc.marks_possible > 0 else 0
            )

            if lost_percentage >= 0.7:
                severity = "critical"
            elif lost_percentage >= 0.4:
                severity = "major"
            elif lost_percentage > 0:
                severity = "minor"
            else:
                continue

            issue = f"Only earned {alloc.marks_earned}/{alloc.marks_possible} marks"
            fix = f"Address missing elements in {alloc.criterion_id}: {alloc.justification}"

            if error_traps:
                relevant_traps = []
                for trap in error_traps:
                    if trap in error_trap_messages:
                        relevant_traps.append(error_trap_messages[trap])
                if relevant_traps:
                    fix += f". Also: {'; '.join(relevant_traps[:2])}"

            fix = FixItem(
                rank=rank,
                severity=severity,
                criterion_id=alloc.criterion_id,
                issue=issue,
                fix=fix,
                marks_impact=round(alloc.marks_possible - alloc.marks_earned, 1),
            )
            fixes.append(fix)
            rank += 1

        if command_term and subject:
            structure_feedback = evaluate_answer_structure(answer, command_term, subject)
            if not structure_feedback.get("complete", True):
                structure_fix = FixItem(
                    rank=rank,
                    severity="major",
                    criterion_id="structure",
                    issue=f"Response doesn't follow required structure for '{command_term}'",
                    fix=f"Follow the required structure: {structure_feedback.get('scaffold', '')[:300]}...",
                    marks_impact=2.0,
                )
                fixes.append(structure_fix)
                rank += 1

        return fixes[:10]

    def _generate_rewrite_target(
        self,
        allocations: list[MarkAllocation],
        answer: str,
        subject: str,
        topic: str,
        criteria: list[MarkingCriterion],
        archetype_tags: Optional[list[str]] = None,
        command_term: Optional[str] = None,
    ) -> RewriteTarget:
        """Generate rewrite target with focus areas."""
        focus_areas = []
        key_improvements = []
        structural_suggestions = []

        archetype_guidance = {
            QuestionArchetype.CALCULATION: [
                "Show all working steps clearly",
                "State the formula before substituting values",
                "Include units in all calculations",
                "Verify your final answer makes sense",
            ],
            QuestionArchetype.DEFINITION: [
                "Start with a clear statement of the term",
                "Include all key components of the definition",
                "Use precise technical terminology",
            ],
            QuestionArchetype.EXTENDED_RESPONSE: [
                "Use PEEL structure (Point, Evidence, Explain, Link)",
                "Address both sides of the argument if required",
                "Include specific examples or evidence",
                "Conclude by directly answering the question",
            ],
            QuestionArchetype.DATA_INTERPRETATION: [
                "Reference specific data points from the source",
                "Identify trends and patterns in the data",
                "Explain what the data demonstrates",
            ],
            QuestionArchetype.COMPARE_CONTRAST: [
                "Address both items being compared",
                "Use comparative language (similarly, whereas, unlike)",
                "Focus on key similarities and differences",
            ],
        }

        critical_count = sum(1 for f in allocations if f.marks_earned < f.marks_possible * 0.5)
        major_count = sum(1 for f in allocations if 0.5 <= f.marks_earned < f.marks_possible * 0.8)

        if critical_count > 0:
            focus_areas.append(f"Address {critical_count} critical gaps in understanding")
        if major_count > 0:
            focus_areas.append(f"Improve {major_count} partially answered criteria")

        if archetype_tags:
            for tag in archetype_tags:
                try:
                    archetype = QuestionArchetype(tag)
                    if archetype in archetype_guidance:
                        focus_areas.extend(archetype_guidance[archetype][:2])
                except ValueError:
                    pass

        for alloc in allocations:
            if alloc.marks_earned < alloc.marks_possible:
                improvement = f"Expand response for {alloc.criterion_id} to cover missing points"
                if improvement not in key_improvements:
                    key_improvements.append(improvement)

        word_count = len(answer.split())
        if word_count < 100:
            structural_suggestions.append("Expand answer with more detail and explanation")
        if word_count > 500:
            structural_suggestions.append("Focus on conciseness while maintaining key points")

        structural_suggestions.append("Use subject-specific terminology accurately")
        structural_suggestions.append("Structure response with clear topic sentences")

        answer_words = answer.split()
        avg_word_length = (
            sum(len(w) for w in answer_words[:50]) / min(len(answer_words), 50)
            if answer_words
            else 0
        )
        if avg_word_length > 7:
            key_improvements.append("Use clearer, more accessible language")

        template = self._get_rewrite_template(subject, topic)

        if command_term and subject:
            scaffold = generate_scaffold(subject, command_term, marks=10)
            if scaffold:
                template = f"{template}\n\n**Required Structure for '{command_term}':**\n{scaffold}"

        return RewriteTarget(
            focus_areas=focus_areas[:5],
            key_improvements=key_improvements[:5],
            template=template,
            word_count_guidance=f"Current: {word_count} words. Target: 150-300 words for concise response",
            structural_suggestions=structural_suggestions[:5],
        )

    def _get_rewrite_template(self, subject: str, topic: str) -> str:
        """Get a rewrite template based on subject."""
        templates = {
            "mathematics": """Rewrite template:
1. State the key concept/theorem used
2. Show working steps clearly
3. State final answer with units

Example structure:
"Using [METHOD], we first [STEP 1]. Then [STEP 2]. Therefore, the solution is [ANSWER].""",
            "chemistry": """Rewrite template:
1. Identify the reaction/process
2. State conditions/reagents
3. Explain the mechanism briefly
4. State the product/outcome

Example structure:
"This reaction involves [TYPE]. Under [CONDITIONS], [REAGENTS] leads to [PRODUCT] via [MECHANISM].""",
            "physics": """Rewrite template:
1. Identify the physical principle
2. State the relevant formula
3. Substitute values with units
4. Calculate and state answer

Example structure:
"Applying [PRINCIPLE], the formula is [FORMULA]. Substituting [VALUES] gives [RESULT].""",
            "english": """Rewrite template:
1. Make a clear thesis statement
2. Provide textual evidence
3. Explain how evidence supports argument
4. Conclude with reference to thesis

Example structure:
"[THESIS]. As demonstrated in [TEXT], [EVIDENCE] shows that [ANALYSIS]. Therefore, [CONCLUSION].""",
        }

        subject_lower = subject.lower() if subject else ""
        for key in templates:
            if key in subject_lower:
                return templates[key]

        return """Rewrite template:
1. Make a clear opening statement
2. Support with evidence/reasoning
3. Explain your reasoning
4. Conclude clearly

Example structure:
"[STATEMENT]. This is shown by [EVIDENCE] because [REASONING]. Therefore, [CONCLUSION]."""

    def _generate_overall_feedback(
        self,
        breakdown: MarkBreakdown,
        subject: str,
        archetype_tags: Optional[list[str]] = None,
        error_traps: Optional[list[str]] = None,
        command_term: Optional[str] = None,
        student_answer: str = "",
    ) -> str:
        """Generate overall feedback message."""
        percentage = breakdown.percentage
        grade = breakdown.grade

        if percentage >= 90:
            feedback = f"Excellent work! Demonstrates strong understanding of {subject} concepts."
        elif percentage >= 80:
            feedback = f"Good performance. Minor improvements needed in detailed responses."
        elif percentage >= 70:
            feedback = f"Satisfactory. Focus on addressing missing key elements."
        elif percentage >= 60:
            feedback = f"Below standard. Significant gaps in understanding require attention."
        else:
            feedback = f"Insufficient. Major concepts need review before proceeding."

        feedback += f" Grade: {grade} ({percentage:.1f}%)"

        if archetype_tags:
            archetype_names = [a.replace("_", " ").title() for a in archetype_tags[:2]]
            if archetype_names:
                feedback += f" Question type: {', '.join(archetype_names)}."

        if error_traps:
            trap_count = len(error_traps)
            if trap_count > 0:
                feedback += (
                    f" Watch out for {trap_count} common trap{'s' if trap_count > 1 else ''}."
                )

        if command_term and student_answer:
            structure_feedback = evaluate_answer_structure(student_answer, command_term, subject)
            if not structure_feedback.get("complete", True):
                missing = structure_feedback.get("missing_components", [])
                if missing:
                    feedback += f" Structure check: Missing {', '.join(missing[:3])}."

        return feedback

    def _compute_quality_score(self, allocations: list[MarkAllocation], answer: str) -> float:
        """Compute an overall quality score (0.0-1.0)."""
        if not allocations:
            return 0.0

        completeness_score = (
            sum(a.marks_earned for a in allocations) / sum(a.marks_possible for a in allocations)
            if allocations
            else 0
        )

        word_count = len(answer.split())
        length_score = min(1.0, word_count / 200) if word_count > 0 else 0

        combined = (completeness_score * 0.7) + (length_score * 0.3)
        return round(min(1.0, combined), 2)


def should_trigger_examiner(
    intent: IntentType | str | None,
    policy_id: str | None,
    evaluation_strictness: float = 0.5,
) -> bool:
    """
    Convenience function to check if examiner should be triggered.

    Args:
        intent: The detected user intent
        policy_id: Current policy mode ID
        evaluation_strictness: Policy evaluation strictness

    Returns:
        True if examiner should evaluate the answer
    """
    examiner = Examiner()
    return examiner.should_examine(intent, policy_id, evaluation_strictness)


def evaluate_answer(
    student_answer: str,
    marking_guidelines: list[dict[str, Any]],
    subject: str = "",
    topic: str = "",
    policy_id: str = "exam",
    strictness: float = 0.95,
) -> MarkingResult:
    """
    Convenience function to evaluate an answer.

    Args:
        student_answer: The student's written answer
        marking_guidelines: Retrieved marking guidelines
        subject: Subject name
        topic: Topic name
        policy_id: Policy mode ID
        strictness: Evaluation strictness

    Returns:
        Complete MarkingResult
    """
    examiner = Examiner()
    return examiner.evaluate(
        student_answer=student_answer,
        marking_guidelines=marking_guidelines,
        subject=subject,
        topic=topic,
        policy_id=policy_id,
        evaluation_strictness=strictness,
    )
