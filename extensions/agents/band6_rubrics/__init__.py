"""
Band 6 Rubric Engine
====================

Provides HSC Band 6 rubric evaluation for student responses across all subjects.
Loads YAML rubric files and evaluates responses against criteria.

Usage:
    from extensions.agents.band6_rubrics import (
        load_rubric,
        evaluate_response,
        Band6RubricResult,
    )

    rubric = load_rubric("english_advanced")
    result = evaluate_response(
        rubric=rubric,
        response=student_answer,
        subject="english_advanced",
    )
    print(result.band_estimate)
    print(result.rubric_checklist)
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.logging import get_logger

logger = get_logger("Band6Rubrics")

RUBRICS_DIR = Path(__file__).parent

SUBJECT_ALIASES = {
    "english": "english_advanced",
    "english_adv": "english_advanced",
    "legal": "legal_studies",
    "legal_studies": "legal_studies",
    "business": "business_studies",
    "business_studies": "business_studies",
    "maths": "mathematics_advanced",
    "math": "mathematics_advanced",
    "mathematics": "mathematics_advanced",
    "bio": "biology",
    "physics": "physics",
    "chem": "chemistry",
    "chemistry": "chemistry",
}


@dataclass
class RubricCriterion:
    """A single criterion from a rubric dimension."""

    id: str
    description: str
    indicators: list[str]
    pass_threshold: float


@dataclass
class RubricDimension:
    """A dimension of the rubric (e.g., Clarity, Accuracy)."""

    name: str
    description: str
    weight: float
    criteria: list[RubricCriterion]


@dataclass
class Band6Rubric:
    """Complete Band 6 rubric for a subject."""

    subject: str
    display_name: str
    hsc_band: int
    description: str
    dimensions: list[RubricDimension]
    band_thresholds: dict[str, int]
    improvement_tips: dict[str, list[str]]

    def get_dimension(self, name: str) -> RubricDimension | None:
        """Get a dimension by name (case-insensitive)."""
        for dim in self.dimensions:
            if dim.name.lower() == name.lower():
                return dim
        return None


@dataclass
class CriterionResult:
    """Result of evaluating a single criterion."""

    criterion_id: str
    description: str
    score: float
    passed: bool
    feedback: str
    indicators_checked: list[str]
    missing_indicators: list[str]


@dataclass
class DimensionResult:
    """Result of evaluating a rubric dimension."""

    dimension_name: str
    score: float
    weight: float
    weighted_score: float
    criteria_results: list[CriterionResult]
    passed: bool


@dataclass
class Band6RubricResult:
    """Complete result of rubric evaluation."""

    band_estimate: str
    band_score: float
    overall_score: float
    overall_passed: bool
    dimension_results: list[DimensionResult]
    rubric_checklist: list[dict[str, Any]]
    top_fixes: list[dict[str, str]]
    improvement_tips: list[str]
    feedback_summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "band_estimate": self.band_estimate,
            "band_score": self.band_score,
            "overall_score": self.overall_score,
            "overall_passed": self.overall_passed,
            "rubric_checklist": [
                {
                    "dimension": d.dimension_name,
                    "score": d.score,
                    "passed": d.passed,
                    "criteria": [
                        {
                            "id": c.criterion_id,
                            "description": c.description,
                            "score": c.score,
                            "passed": c.passed,
                            "feedback": c.feedback,
                        }
                        for c in d.criteria_results
                    ],
                }
                for d in self.dimension_results
            ],
            "top_fixes": self.top_fixes,
            "improvement_tips": self.improvement_tips,
            "feedback_summary": self.feedback_summary,
        }


def load_rubric_file(filepath: Path) -> dict[str, Any]:
    """Load a YAML rubric file."""
    import yaml

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load rubric {filepath}: {e}")
        raise


def parse_rubric(data: dict[str, Any]) -> Band6Rubric:
    """Parse rubric data into Band6Rubric object."""
    dimensions = []
    for dim_data in data.get("dimensions", []):
        criteria = []
        for crit_data in dim_data.get("criteria", []):
            criteria.append(
                RubricCriterion(
                    id=crit_data["id"],
                    description=crit_data["description"],
                    indicators=crit_data.get("indicators", []),
                    pass_threshold=crit_data.get("pass_threshold", 0.7),
                )
            )
        dimensions.append(
            RubricDimension(
                name=dim_data["name"],
                description=dim_data.get("description", ""),
                weight=dim_data["weight"],
                criteria=criteria,
            )
        )

    return Band6Rubric(
        subject=data["subject"],
        display_name=data.get("display_name", data["subject"]),
        hsc_band=data.get("hsc_band", 6),
        description=data.get("description", ""),
        dimensions=dimensions,
        band_thresholds=data.get("band_thresholds", {}),
        improvement_tips=data.get("improvement_tips", {}),
    )


def load_rubric(subject: str) -> Band6Rubric | None:
    """
    Load a rubric for the given subject.

    Args:
        subject: Subject name (e.g., "english_advanced", "legal_studies")

    Returns:
        Band6Rubric object or None if not found
    """
    subject_lower = subject.lower().replace(" ", "_").replace("-", "_")
    alias = SUBJECT_ALIASES.get(subject_lower, subject_lower)

    rubrics_dir = Path(__file__).parent
    possible_names = [f"{alias}.yaml", f"{subject_lower}.yaml"]

    for name in possible_names:
        filepath = rubrics_dir / name
        if filepath.exists():
            try:
                data = load_rubric_file(filepath)
                return parse_rubric(data)
            except Exception as e:
                logger.error(f"Error parsing rubric {filepath}: {e}")
                continue

    logger.warning(f"No rubric found for subject: {subject}")
    return None


def evaluate_criterion(criterion: RubricCriterion, response: str, dim_name: str) -> CriterionResult:
    """Evaluate a single criterion against a response."""
    response_lower = response.lower()

    indicators_checked = []
    missing_indicators = []
    indicator_scores = []

    for indicator in criterion.indicators:
        indicator_lower = indicator.lower()
        if len(indicator_lower) > 10:
            key_phrases = indicator_lower.split(",")[0:2]
            matched = any(phrase.strip() in response_lower for phrase in key_phrases)
            if matched:
                indicators_checked.append(indicator)
                indicator_scores.append(1.0)
            else:
                missing_indicators.append(indicator)
                indicator_scores.append(0.3)
        else:
            if indicator_lower in response_lower:
                indicators_checked.append(indicator)
                indicator_scores.append(1.0)
            else:
                missing_indicators.append(indicator)
                indicator_scores.append(0.2)

    score = sum(indicator_scores) / len(indicator_scores) if indicator_scores else 0.5
    passed = score >= criterion.pass_threshold

    if passed:
        feedback = f"✓ {criterion.description}: Demonstrated effectively"
    elif len(indicators_checked) == 0:
        feedback = f"✗ {criterion.description}: Not addressed in response"
    else:
        feedback = f"△ {criterion.description}: Partially addressed ({len(indicators_checked)}/{len(criterion.indicators)} indicators)"

    return CriterionResult(
        criterion_id=criterion.id,
        description=criterion.description,
        score=score,
        passed=passed,
        feedback=feedback,
        indicators_checked=indicators_checked,
        missing_indicators=missing_indicators,
    )


def evaluate_dimension(dimension: RubricDimension, response: str) -> DimensionResult:
    """Evaluate a rubric dimension against a response."""
    criteria_results = [evaluate_criterion(c, response, dimension.name) for c in dimension.criteria]

    dim_score = sum(c.score for c in criteria_results) / len(criteria_results)
    weighted_score = dim_score * dimension.weight
    passed = dim_score >= 0.7

    return DimensionResult(
        dimension_name=dimension.name,
        score=dim_score,
        weight=dimension.weight,
        weighted_score=weighted_score,
        criteria_results=criteria_results,
        passed=passed,
    )


def calculate_band(score: float, rubric: Band6Rubric) -> tuple[str, float]:
    """Calculate the HSC band from the overall score."""
    thresholds = rubric.band_thresholds

    if score >= thresholds.get("band_6", 90):
        return "Band 6", 6.0
    elif score >= thresholds.get("band_5", 80):
        return "Band 5", 5.0
    elif score >= thresholds.get("band_4", 70):
        return "Band 4", 4.0
    elif score >= thresholds.get("band_3", 60):
        return "Band 3", 3.0
    elif score >= thresholds.get("band_2", 50):
        return "Band 2", 2.0
    else:
        return "Band 1", 1.0


def generate_top_fixes(
    dimension_results: list[DimensionResult], rubric: Band6Rubric, limit: int = 3
) -> list[dict[str, str]]:
    """Generate the top 3 fixes based on failed criteria."""
    fixes = []

    for dim_result in dimension_results:
        if not dim_result.passed:
            for crit in dim_result.criteria_results:
                if not crit.passed and crit.missing_indicators:
                    dim_tips = rubric.improvement_tips.get(
                        dim_result.dimension_name.lower().replace(" ", "_"), []
                    )
                    tip = dim_tips[0] if dim_tips else f"Review {dim_result.dimension_name}"

                    fixes.append(
                        {
                            "dimension": dim_result.dimension_name,
                            "criterion": crit.description,
                            "issue": f"Missing: {', '.join(crit.missing_indicators[:2])}",
                            "fix": tip,
                            "priority": "high" if dim_result.score < 0.5 else "medium",
                        }
                    )

    fixes.sort(key=lambda x: (x["priority"] == "high", len(x["issue"])), reverse=True)
    return fixes[:limit]


def evaluate_response(
    rubric: Band6Rubric,
    response: str,
    subject: str = "",
) -> Band6RubricResult:
    """
    Evaluate a student response against a Band 6 rubric.

    Args:
        rubric: The Band6Rubric to evaluate against
        response: The student's response
        subject: Subject name (for improvement tips)

    Returns:
        Band6RubricResult with band estimate, checklist, and fixes
    """
    dimension_results = [evaluate_dimension(dim, response) for dim in rubric.dimensions]

    overall_score = sum(d.weighted_score for d in dimension_results) * 100
    overall_passed = overall_score >= 70

    band_label, band_value = calculate_band(overall_score, rubric)

    rubric_checklist = []
    for dim_result in dimension_results:
        rubric_checklist.append(
            {
                "dimension": dim_result.dimension_name,
                "score": round(dim_result.score * 100, 1),
                "passed": dim_result.passed,
                "criteria": [
                    {
                        "id": c.criterion_id,
                        "description": c.description,
                        "score": round(c.score * 100, 1),
                        "passed": c.passed,
                        "feedback": c.feedback,
                    }
                    for c in dim_result.criteria_results
                ],
            }
        )

    top_fixes = generate_top_fixes(dimension_results, rubric)

    improvement_tips = []
    for dim_result in dimension_results:
        if not dim_result.passed:
            dim_key = dim_result.dimension_name.lower().replace(" ", "_")
            tips = rubric.improvement_tips.get(dim_key, [])
            improvement_tips.extend(tips[:2])

    feedback_summary_parts = []
    if overall_passed:
        feedback_summary_parts.append(
            f"Strong performance at {band_label} level with {overall_score:.1f}% overall."
        )
    else:
        feedback_summary_parts.append(
            f"Currently performing below {band_label} threshold at {overall_score:.1f}%."
        )

    failed_dims = [d for d in dimension_results if not d.passed]
    if failed_dims:
        feedback_summary_parts.append(
            f"Primary areas for improvement: {', '.join(d.dimension_name for d in failed_dims[:2])}."
        )

    if top_fixes:
        feedback_summary_parts.append(
            f"Top priority: {top_fixes[0]['issue']} - {top_fixes[0]['fix']}"
        )

    return Band6RubricResult(
        band_estimate=band_label,
        band_score=band_value,
        overall_score=overall_score,
        overall_passed=overall_passed,
        dimension_results=dimension_results,
        rubric_checklist=rubric_checklist,
        top_fixes=top_fixes,
        improvement_tips=improvement_tips[:6],
        feedback_summary=" ".join(feedback_summary_parts),
    )


def evaluate_answer(
    response: str,
    subject: str,
) -> Band6RubricResult | None:
    """
    Convenience function to evaluate a response against a subject's rubric.

    Args:
        response: The student's response
        subject: Subject name

    Returns:
        Band6RubricResult or None if rubric not found
    """
    rubric = load_rubric(subject)
    if rubric is None:
        return None
    return evaluate_response(rubric, response, subject)


def get_available_subjects() -> list[dict[str, str]]:
    """Get list of available rubric subjects."""
    rubrics_dir = Path(__file__).parent / "band6_rubrics"
    subjects = []

    for f in rubrics_dir.glob("*.yaml"):
        try:
            data = load_rubric_file(f)
            subjects.append(
                {
                    "subject": data.get("subject", f.stem),
                    "display_name": data.get("display_name", f.stem.replace("_", " ").title()),
                    "hsc_band": data.get("hsc_band", 6),
                }
            )
        except Exception:
            continue

    return subjects
