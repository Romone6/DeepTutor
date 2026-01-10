"""
Maths Advanced Working Validator
=================================

Validates HSC Mathematics Advanced workings for compliance and common errors.
Provides step-by-step compliance checking and trap detection.

Usage:
    from extensions.agents.maths_working_validator import (
        validate_working,
        WorkingValidationResult,
        COMMON_TRAPS,
    )

    result = validate_working(
        question="Solve ln(x) + ln(x+2) = 3",
        working="x^2 + 2x = e^3, x = [-2 ± sqrt(4 + 4e^3)]/2, x = -1 ± sqrt(1 + e^3)",
        answer="x = -1 + sqrt(1 + e^3)",
        topic="calculus",
    )
    print(result.traps_found)
    print(result.compliance_issues)
"""

import re
from dataclasses import dataclass, field
from typing import Any

from src.logging import get_logger

logger = get_logger("MathsWorkingValidator")


@dataclass
class WorkingIssue:
    """A detected issue in the working."""

    type: str
    severity: str  # "error", "warning", "info"
    description: str
    location: str = ""
    suggestion: str = ""
    mark_impact: float = 0.0


@dataclass
class WorkingValidationResult:
    """Complete result of validating math working."""

    is_compliant: bool
    compliance_score: float
    issues: list[WorkingIssue]
    traps_found: list[dict[str, Any]]
    compliance_issues: list[str]
    marks_deducted: float
    feedback_summary: str
    rewrite_suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_compliant": self.is_compliant,
            "compliance_score": self.compliance_score,
            "issues": [i.__dict__ for i in self.issues],
            "traps_found": self.traps_found,
            "compliance_issues": self.compliance_issues,
            "marks_deducted": self.marks_deducted,
            "feedback_summary": self.feedback_summary,
            "rewrite_suggestions": self.rewrite_suggestions,
        }


COMMON_TRAPS = {
    "log_rules": {
        "name": "Logarithm Rules",
        "patterns": [
            r"ln\s*\([^)]+\)\s*\+\s*ln\s*\([^)]+\)\s*=\s*ln\s*\([^)]*\+\s*[^)]*\)",
            r"log\s*\([^)]+\)\s*\+\s*log\s*\([^)]+\)\s*=\s*log\s*\([^)]*\+\s*[^)]*\)",
            r"ln\s*\([^)]+\)\s*\+\s*ln\s*\([^)]+\)\s*=\s*ln\s*\(\s*\w+\s*\+\s*\w+\s*\)",
        ],
        "trap_description": "Incorrect logarithm rule application",
        "common_error": "Misapplying log addition rules",
        "example": "ln(a) + ln(b) ≠ ln(a + b), should be ln(a) + ln(b) = ln(ab)",
    },
    "trig_identities": {
        "name": "Trigonometric Identities",
        "patterns": [
            r"1\s*\+\s*tan\s*2?\s*\(\s*\w+\s*\)\s*=\s*sec\s*\(\s*\w+\s*\)",
            r"1\s*\+\s*cot\s*2?\s*\(\s*\w+\s*\)\s*=\s*csc\s*\(\s*\w+\s*\)",
        ],
        "trap_description": "Incorrect trigonometric identity",
        "common_error": "Forgetting squares in identities",
        "example": "1 + tan²(x) = sec²(x), not sec(x)",
    },
    "sign_errors": {
        "name": "Sign Errors",
        "patterns": [
            r"-\s*\(\s*\w+\s*[-+]\s*\w+\s*\)\s*=\s*-\w+\s*[-+]\s*\w+",
            r"\(\s*\w+\s*[-+]\s*\w+\s*\)\s*\^2\s*=\s*\w+\s*\^2\s*[-+]\s*2?\w+\w+\s*\+\s*\w+\s*\^2",
        ],
        "trap_description": "Sign manipulation errors",
        "common_error": "Losing track of negative signs",
        "example": "-(x - 3) = -x + 3, not -x - 3",
    },
    "extraneous_solutions": {
        "name": "Extraneous Solutions",
        "patterns": [
            r"=\s*[^)]*\s*[,;]\s*(check|verify|since|because)",
            r"=\s*[±√]\s*\w+\s*[,;]\s*(must|check|verify)",
        ],
        "trap_description": "Solutions that don't satisfy original equation",
        "common_error": "Not checking domain or extraneous roots",
        "example": "When squaring both sides, check solutions in original equation",
    },
    "domain_restrictions": {
        "name": "Domain Restrictions",
        "patterns": [
            r"x\s*>\s*0",
            r"x\s*>=",
            r"domain",
            r"restriction",
        ],
        "trap_description": "Missing or incorrect domain",
        "common_error": "Forgetting domain restrictions",
        "example": "ln(x) requires x > 0, √x requires x ≥ 0",
    },
    "calculus_steps": {
        "name": "Calculus Steps",
        "patterns": [
            r"d\s*\/\s*dx\s*\[\s*\w+\s*\([^)]*\)\s*\]\s*=\s*\w+\s*\([^)]*\)",
            r"d\s*\/\s*dx\s*\[\s*\w+\s*\^\s*2\s*\]\s*=\s*\w+",
            r"d\s*\/\s*dx\s*\[\s*\w+\s*\]\s*=\s*\w+\s*\^\s*2?",
        ],
        "trap_description": "Incorrect calculus application",
        "common_error": "Forgetting chain rule",
        "example": "d/dx[sin(x²)] = cos(x²) · 2x, not cos(x²)",
    },
    "algebra_slips": {
        "name": "Basic Algebra",
        "patterns": [
            r"\(\s*\w+\s*[+-]\s*\w+\s*\)\s*\^2\s*=\s*\w+\s*\^2\s*[+-]\s*\w+\s*\^2",
            r"\w+\s*\(\s*\w+\s*[+-]\s*\w+\s*\)\s*=\s*\w+\s*\^2\s*[+-]\s*\w+\s*",
        ],
        "trap_description": "Basic algebraic errors",
        "common_error": "Expanding or simplifying incorrectly",
        "example": "(a + b)² = a² + 2ab + b², not a² + b²",
    },
    "rounding_errors": {
        "name": "Rounding and Units",
        "patterns": [
            r"\d+\.\d{4,}",
            r"units?",
            r"\bcm\b",
            r"\bm\b",
            r"\bs\b",
        ],
        "trap_description": "Incorrect rounding or missing units",
        "common_error": "Rounding too early or not at all",
        "example": "Keep full precision until final answer, then round to 3 significant figures",
    },
}


@dataclass
class TopicPattern:
    """Pattern for detecting math topics."""

    name: str
    keywords: list[str]
    required_patterns: list[str]
    domain_restriction: str = ""
    common_traps: list[str] = field(default_factory=list)


TOPIC_PATTERNS = {
    "general": TopicPattern(
        name="General Mathematics",
        keywords=[],
        required_patterns=["show working", "state answer"],
        common_traps=["algebra_slips", "sign_errors"],
    ),
    "calculus": TopicPattern(
        name="Calculus",
        keywords=["derivative", "integral", "differentiation", "integration", "d/dx", "∫"],
        required_patterns=["show working", "state formula", "substitute"],
        common_traps=["calculus_steps", "sign_errors", "domain_restrictions"],
    ),
    "exponential_logarithmic": TopicPattern(
        name="Exponential and Logarithmic Functions",
        keywords=["ln", "log", "e^", "exponential"],
        required_patterns=["state domain", "apply log rules", "check solutions"],
        domain_restriction="x > 0 for ln(x)",
        common_traps=["log_rules", "extraneous_solutions", "domain_restrictions"],
    ),
    "trigonometric": TopicPattern(
        name="Trigonometric Functions",
        keywords=["sin", "cos", "tan", "trig", "radians", "degrees"],
        required_patterns=["state identity", "apply correctly", "check domain"],
        domain_restriction="tan(x) undefined at π/2 + nπ",
        common_traps=["trig_identities", "sign_errors", "domain_restrictions"],
    ),
    "algebraic_techniques": TopicPattern(
        name="Algebraic Techniques",
        keywords=["simplify", "expand", "factor", "solve"],
        required_patterns=["show each step", "check solution"],
        common_traps=["algebra_slips", "sign_errors", "extraneous_solutions"],
    ),
    "financial_math": TopicPattern(
        name="Financial Mathematics",
        keywords=["interest", "investment", "loan", "nper", "pmt", "fv", "pv"],
        required_patterns=["identify variables", "state formula", "substitute correctly"],
        common_traps=["sign_errors", "rounding_errors"],
    ),
    "statistical_analysis": TopicPattern(
        name="Statistical Analysis",
        keywords=["mean", "median", "variance", "sd", "probability", "normal"],
        required_patterns=["state formula", "show calculation", "interpret result"],
        common_traps=["rounding_errors"],
    ),
}


def detect_topic(question: str) -> str:
    """Detect the math topic from the question."""
    question_lower = question.lower()

    for topic_id, pattern in TOPIC_PATTERNS.items():
        for keyword in pattern.keywords:
            if keyword.lower() in question_lower:
                return topic_id

    return "general"


def check_compliance(
    question: str,
    working: str,
    answer: str,
    topic: str,
) -> list[WorkingIssue]:
    """Check working for HSC compliance."""
    issues = []

    if not working or len(working.strip()) < 10:
        issues.append(
            WorkingIssue(
                type="missing_working",
                severity="error",
                description="No or insufficient working shown",
                suggestion="Show all steps including formulas, substitutions, and calculations",
                mark_impact=3.0,
            )
        )
        return issues

    topic_pattern = TOPIC_PATTERNS.get(topic, TOPIC_PATTERNS["general"])

    # Check for required elements based on topic
    working_lower = working.lower()

    # Check if formula/method is stated
    formula_patterns = [r"using", r"formula", r"method", r"d\/dx", r"∫", r"by"]
    if not any(re.search(p, working_lower) for p in formula_patterns):
        issues.append(
            WorkingIssue(
                type="missing_formula",
                severity="warning",
                description="Formula or method not explicitly stated",
                suggestion="Start by stating the formula or method you'll use",
                mark_impact=1.0,
            )
        )

    # Check for substitution
    sub_patterns = [r"substitut", r"replac", r"putting", r"x\s*=", r"\d+\s*=", r"=\s*\d"]
    if not any(re.search(p, working_lower) for p in sub_patterns):
        issues.append(
            WorkingIssue(
                type="missing_substitution",
                severity="warning",
                description="Values not shown being substituted",
                suggestion="Show the substitution step clearly",
                mark_impact=1.0,
            )
        )

    # Check for final statement
    answer_patterns = [r"therefore", r"so", r"hence", r"thus", r"final answer"]
    if not any(re.search(p, working_lower) for p in answer_patterns):
        issues.append(
            WorkingIssue(
                type="missing_final_statement",
                severity="warning",
                description="No final statement before answer",
                suggestion="Add 'Therefore, the answer is...' before stating the final answer",
                mark_impact=0.5,
            )
        )

    # Check for domain restrictions if applicable
    if topic_pattern.domain_restriction:
        domain_patterns = [r"x\s*>\s*0", r"x\s*>=", r"domain", r"restriction", r"x\s*\\in"]
        if not any(re.search(p, working_lower) for p in domain_patterns):
            issues.append(
                WorkingIssue(
                    type="missing_domain",
                    severity="warning",
                    description=f"Domain restriction not mentioned: {topic_pattern.domain_restriction}",
                    suggestion=f"State the domain restriction: {topic_pattern.domain_restriction}",
                    mark_impact=1.0,
                )
            )

    # Check for units if relevant
    unit_patterns = [r"m\/s", r"cm", r"kg", r"degrees?", r"radians", r"m", r"s"]
    if any(re.search(p, working_lower) for p in unit_patterns):
        if "units" not in working_lower and "unit" not in working_lower:
            issues.append(
                WorkingIssue(
                    type="missing_units",
                    severity="warning",
                    description="Units mentioned but not explicitly stated in answer",
                    suggestion="Include units in your final answer",
                    mark_impact=0.5,
                )
            )

    # Check for rounding
    decimal_patterns = [r"\d+\.\d{4,}", r"\d+\.\d{3}"]
    if re.search(r"\d+\.\d{3,}", working):
        issues.append(
            WorkingIssue(
                type="excessive_decimals",
                severity="info",
                description="Intermediate calculations show excessive decimals",
                suggestion="Round to 3-4 significant figures for intermediate steps",
                mark_impact=0.0,
            )
        )

    return issues


def check_traps(
    question: str,
    working: str,
    answer: str,
    topic: str,
) -> list[dict[str, Any]]:
    """Check for common HSC math traps."""
    traps_found = []

    topic_pattern = TOPIC_PATTERNS.get(topic, TOPIC_PATTERNS["general"])
    relevant_traps = topic_pattern.common_traps + ["sign_errors", "algebra_slips"]

    # Exclude compliance-related traps from trap detection
    # Domain restrictions and rounding errors are checked in check_compliance
    trap_exclusions = ["domain_restrictions", "rounding_errors", "extraneous_solutions"]

    for trap_key in relevant_traps:
        if trap_key in trap_exclusions:
            continue
        if trap_key not in COMMON_TRAPS:
            continue

        trap = COMMON_TRAPS[trap_key]

        for pattern in trap["patterns"]:
            matches = re.findall(pattern, working, re.IGNORECASE)
            if matches:
                for match in matches[:2]:  # Limit to first 2 matches per trap
                    traps_found.append(
                        {
                            "type": trap_key,
                            "name": trap["name"],
                            "description": trap["trap_description"],
                            "matched": match if isinstance(match, str) else str(match),
                            "common_error": trap["common_error"],
                            "example": trap["example"],
                            "severity": "high"
                            if trap_key in ["log_rules", "trig_identities", "sign_errors"]
                            else "medium",
                        }
                    )

    return traps_found


def validate_working(
    question: str,
    working: str,
    answer: str,
    topic: str = "",
) -> WorkingValidationResult:
    """
    Validate HSC Mathematics Advanced working.

    Args:
        question: The original math question
        working: The student's working steps
        answer: The student's final answer
        topic: Optional topic hint (auto-detected if not provided)

    Returns:
        WorkingValidationResult with compliance issues and traps found
    """
    if not topic:
        topic = detect_topic(question)

    compliance_issues = check_compliance(question, working, answer, topic)
    traps_found = check_traps(question, working, answer, topic)

    # Calculate scores
    max_marks = 7.0  # Typical HSC question worth

    marks_deducted = sum(issue.mark_impact for issue in compliance_issues)
    for trap in traps_found:
        if trap["severity"] == "high":
            marks_deducted += 1.5
        else:
            marks_deducted += 0.5

    compliance_score = max(0, 100 - (marks_deducted / max_marks * 100))

    is_compliant = compliance_score >= 70 and len(traps_found) == 0

    # Generate feedback
    feedback_parts = []
    if compliance_issues:
        high_severity = [i for i in compliance_issues if i.severity == "error"]
        if high_severity:
            feedback_parts.append(f"Critical: {high_severity[0].description}")
        warning_count = len([i for i in compliance_issues if i.severity == "warning"])
        if warning_count:
            feedback_parts.append(f"{warning_count} compliance warning(s)")

    if traps_found:
        high_traps = [t for t in traps_found if t["severity"] == "high"]
        if high_traps:
            feedback_parts.append(f"⚠️ {len(high_traps)} potential trap(s) detected")

    if not feedback_parts:
        feedback_parts.append("Working appears complete and correct")

    rewrite_suggestions = []
    for issue in compliance_issues[:3]:
        rewrite_suggestions.append(f"• {issue.suggestion}")
    for trap in traps_found[:2]:
        rewrite_suggestions.append(f"• Check: {trap['example']}")

    return WorkingValidationResult(
        is_compliant=is_compliant,
        compliance_score=round(compliance_score, 1),
        issues=compliance_issues,
        traps_found=traps_found,
        compliance_issues=[i.description for i in compliance_issues],
        marks_deducted=round(marks_deducted, 1),
        feedback_summary=". ".join(feedback_parts),
        rewrite_suggestions=rewrite_suggestions,
    )


def validate_answer_only(
    answer: str,
    correct_answer: str,
    tolerance: float = 0.01,
) -> dict[str, Any]:
    """Simple answer validation with tolerance checking."""
    try:
        # Try to extract numeric values
        answer_nums = [float(s) for s in re.findall(r"-?\d+\.?\d*", answer)]
        correct_nums = [float(s) for s in re.findall(r"-?\d+\.?\d*", correct_answer)]

        if len(answer_nums) == len(correct_nums):
            all_close = all(
                abs(a - c) < tolerance * max(abs(c), 1) for a, c in zip(answer_nums, correct_nums)
            )
            return {
                "is_correct": all_close,
                "answer_matches": all_close,
                "note": "Numerical comparison within tolerance" if all_close else "Values differ",
            }

        # String comparison for exact answers
        return {
            "is_correct": answer.strip().lower() == correct_answer.strip().lower(),
            "answer_matches": answer.strip().lower() == correct_answer.strip().lower(),
            "note": "Exact string comparison",
        }
    except Exception as e:
        return {
            "is_correct": False,
            "answer_matches": False,
            "note": f"Could not compare: {e}",
        }


def generate_mistake_check_section(
    question: str,
    working: str,
    answer: str,
    topic: str = "",
) -> str:
    """
    Generate a 'Mistake Check' section for the solve output.

    Returns a formatted markdown section to append to solve output.
    """
    result = validate_working(question, working, answer, topic)

    sections = ["### Mistake Check\n"]

    # Summary
    if result.is_compliant:
        sections.append("✅ **Compliance: PASS**\n")
    else:
        sections.append("⚠️ **Compliance: REVIEW NEEDED**\n")

    sections.append(
        f"**Score:** {result.compliance_score:.0f}%  |  **Issues:** {len(result.compliance_issues)}  |  **Traps Detected:** {len(result.traps_found)}\n"
    )

    # Compliance issues
    if result.compliance_issues:
        sections.append("**Compliance Issues:**\n")
        for issue in result.compliance_issues:
            sections.append(f"- {issue}\n")
        sections.append("\n")

    # Traps found
    if result.traps_found:
        sections.append("**⚠️ Potential Trap Alert:**\n")
        for trap in result.traps_found[:3]:
            sections.append(f"- **{trap['name']}**: {trap['matched']}\n")
            sections.append(f"  - Check: {trap['example']}\n")
        sections.append("\n")

    # Suggestions
    if result.rewrite_suggestions:
        sections.append("**Improvement Suggestions:**\n")
        for suggestion in result.rewrite_suggestions[:5]:
            sections.append(f"{suggestion}\n")
        sections.append("\n")

    # Final verdict
    if result.is_compliant:
        sections.append("**Verdict:** Your working is complete and appears correct. ✅\n")
    else:
        sections.append(
            f"**Verdict:** Review the issues above. Approximately {result.marks_deducted:.1f} marks may be affected.\n"
        )

    return "".join(sections)


def get_common_traps_for_topic(topic: str) -> list[dict[str, str]]:
    """Get list of common traps for a specific topic."""
    topic_pattern = TOPIC_PATTERNS.get(topic, TOPIC_PATTERNS["general"])
    relevant_traps = topic_pattern.common_traps + ["sign_errors", "algebra_slips"]

    return [
        {
            "type": trap_key,
            "name": COMMON_TRAPS[trap_key]["name"],
            "example": COMMON_TRAPS[trap_key]["example"],
            "common_error": COMMON_TRAPS[trap_key]["common_error"],
        }
        for trap_key in relevant_traps
        if trap_key in COMMON_TRAPS
    ]
