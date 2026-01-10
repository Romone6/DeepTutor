"""
Business Studies Answer Quality Guardrails
==========================================

Enforces HSC Business Studies answer quality: concept definition, case evidence,
analysis, judgement, and explicit syllabus term links.

Usage:
    from extensions.agents.business_answer_guardrails import (
        BusinessGuardrailResult,
        validate_business_answer,
        check_judgement,
        check_case_evidence,
        generate_exemplar_paragraph,
    )

    result = validate_business_answer(
        question="Evaluate the effectiveness of a business strategy",
        answer="The business did well because they had good leadership.",
        case_context="Case: Apple Inc. - Tim Cook's supply chain optimization",
        topic="operations",
    )
    print(result.missing_elements)
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any

from src.logging import get_logger

logger = get_logger("BusinessGuardrails")


@dataclass
class BusinessIssue:
    """A detected issue in Business Studies answer."""

    type: str
    severity: str  # "error", "warning", "info"
    description: str
    original_text: str = ""
    suggestion: str = ""
    missing_element: str = ""
    mark_impact: float = 0.0


@dataclass
class BusinessGuardrailResult:
    """Complete result of validating Business Studies answer."""

    is_acceptable: bool
    score: float
    issues: list[BusinessIssue]
    missing_concepts: list[str]
    missing_evidence: list[str]
    missing_judgement: list[str]
    missing_syllabus_links: list[str]
    structure_issues: list[str]
    rewrite_suggestions: list[str] = field(default_factory=list)
    concepts_found: list[str] = field(default_factory=list)
    evidence_found: list[str] = field(default_factory=list)
    exemplar_paragraph: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_acceptable": self.is_acceptable,
            "score": self.score,
            "issues": [i.__dict__ for i in self.issues],
            "missing_concepts": self.missing_concepts,
            "missing_evidence": self.missing_evidence,
            "missing_judgement": self.missing_judgement,
            "missing_syllabus_links": self.missing_syllabus_links,
            "structure_issues": self.structure_issues,
            "rewrite_suggestions": self.rewrite_suggestions,
            "concepts_found": self.concepts_found,
            "evidence_found": self.evidence_found,
            "exemplar_paragraph": self.exemplar_paragraph,
        }


SYLLABUS_CONCEPTS = {
    "operations": {
        "required": [
            "operations",
            "production",
            "efficiency",
            "quality control",
            "supply chain",
            "inventory management",
            "production planning",
        ],
        "recommended": [
            "just-in-time",
            "total quality management",
            "lean manufacturing",
            "outsourcing",
            "capacity utilization",
            "BCG matrix",
            "economies of scale",
        ],
        "frameworks": ["SWOT", "PESTEL", "Porter's Five Forces", "Value Chain"],
    },
    "marketing": {
        "required": [
            "marketing",
            "target market",
            "marketing mix",
            "product",
            "price",
            "place",
            "promotion",
            "market research",
        ],
        "recommended": [
            "branding",
            "consumer behaviour",
            "market segmentation",
            "product life cycle",
            "pricing strategies",
            "distribution channels",
            "promotional mix",
        ],
        "frameworks": ["Ansoff Matrix", "BCG Matrix", "STP Model"],
    },
    "finance": {
        "required": [
            "financial planning",
            "budgeting",
            "cash flow",
            "revenue",
            "expenses",
            "profit",
            "financial statements",
            "ratio analysis",
        ],
        "recommended": [
            "break-even analysis",
            "return on investment",
            "working capital",
            "sources of finance",
            "cost of capital",
            "investment appraisal",
            "financial performance",
        ],
        "frameworks": ["DuPont Analysis", "Cash Flow Statement", "Ratio Analysis"],
    },
    "human_resources": {
        "required": [
            "human resources",
            "recruitment",
            "training",
            "performance appraisal",
            "industrial relations",
            "workforce planning",
            "organisational culture",
        ],
        "recommended": [
            "motivation theories",
            "leadership styles",
            "team dynamics",
            "job design",
            "remuneration",
            "OH&S",
            "diversity",
        ],
        "frameworks": ["Maslow's Hierarchy", "Herzberg's Two-Factor Theory", "McGregor"],
    },
    "strategic_management": {
        "required": [
            "strategy",
            "competitive advantage",
            "business objectives",
            "stakeholders",
            "corporate social responsibility",
            "globalisation",
            "technology",
        ],
        "recommended": [
            "mergers and acquisitions",
            "diversification",
            "market entry",
            "risk management",
            "innovation",
            "sustainability",
            "ethical considerations",
        ],
        "frameworks": ["SWOT", "PESTEL", "Porter's Generic Strategies", "Balanced Scorecard"],
    },
    "entrepreneurship": {
        "required": [
            "entrepreneurship",
            "small business",
            "business plan",
            "start-up",
            "innovation",
            "risk-taking",
            "opportunity recognition",
        ],
        "recommended": [
            "bootstrapping",
            "venture capital",
            "angel investors",
            "growth strategies",
            "exit strategies",
            "intellectual property",
            "business incubator",
        ],
        "frameworks": ["Business Model Canvas", "Lean Canvas", "SWOT"],
    },
}

CASE_EVIDENCE_MARKERS = {
    "strong": [
        r"\b(company|firm|business)\s+(?:name\s+)?(?:is|was|has|had)\s+[\w\s]+",
        r"\b(?:in|for)\s+\d{4}\s+(?:the\s+)?(?:company|firm|business)",
        r"\baccording to\s+(?:the\s+)?(?:case|annual report|financial statements)",
        r"\bdata (?:from|shows|indicates)",
        r"\bfinancial (?:results|performance|data|ratios)",
        r"\b(?:sales|revenue|profit)\s+(?:increased|decreased|grew|fell|by|of)",
        r"\bmarket share",
        r"\b(?:case study|case)\s+(?:of|about)",
    ],
    "weak": [
        r"\bmany businesses",
        r"\bmost companies",
        r"\bsome businesses",
        r"\bthey (?:often|sometimes|usually)",
        r"\bin business",
        r"\bbusinesses (?:tend|usually|often)",
    ],
}

JUDGEMENT_MARKERS = {
    "strong": [
        r"\b(therefore|thus|consequently)\s+",
        r"\bshould\s+(?:be|consider|implement)",
        r"\bis (?:recommended|suggested|advised)",
        r"\b(most|least)\s+effective",
        r"\bbest (?:option|approach|strategy)",
        r"\bhighly\s+(?:effective|successful|recommended)",
        r"\bsignificantly\s+(?:improved|affected|impacted)",
        r"\bcritical(?:ly)?\s+(?:factor|to)",
        r"\bkey(?:ly)?\s+(?:to|for)",
        r"\bin order to (?:maximise|minimise|achieve)",
    ],
    "weak": [
        r"\bmight\s+(?:be|consider)",
        r"\bcould\s+(?:be|consider)",
        r"\bmaybe",
        r"\bperhaps",
        r"\bsometimes",
        r"\bit depends",
        r"\bmight work",
    ],
}

STRUCTURE_MARKERS = {
    "definition": [
        r"\b(is defined as|refers to|means|is a)",
        r"\bthe concept of",
        r"\bby definition",
    ],
    "analysis": [
        r"\bbecause|due to|this means|as a result|consequently",
        r"\bhowever|although|despite",
    ],
    "evidence": [
        r"\bfor example|for instance|in the case of|such as",
        r"\bspecifically|particularly",
    ],
    "judgement": [r"\btherefore|thus|in conclusion|overall", r"\bthis suggests|this indicates"],
    "linkage": [
        r"\blinked to|related to|connected with|associated with",
        r"\baccording to the syllabus",
    ],
}

GENERIC_PHRASES = {
    "to_avoid": [
        r"\bgood management",
        r"\bbetter decision making",
        r"\bstrong leadership",
        r"\befficient operations",
        r"\bcompetitive advantage",
        r"\bsuccessful business",
        r"\bworks well",
        r"\bdoes well",
        r"\bdoes not work",
        r"\bhas problems",
        r"\bhelpful to the business",
        r"\bimportant for business",
        r"\bgood for the company",
        r"\bmakes profit",
        r"\bincreases sales",
    ],
    "preferred_alternatives": {
        "good management": "specific leadership style or management practice",
        "better decision making": "evidence-based or data-informed decisions",
        "strong leadership": "specific leadership approach (e.g., transformational, servant leadership)",
        "efficient operations": "specific operational efficiency measure (e.g., lean manufacturing, JIT)",
        "competitive advantage": "specific source of advantage (e.g., cost leadership, differentiation)",
        "successful business": "specific success metric (e.g., market share growth, ROI)",
        "works well": "specific outcome with evidence",
        "does well": "specific performance indicator",
        "does not work": "specific failure with causal factors",
        "has problems": "specific issue with root cause",
        "helpful to the business": "specific benefit with impact",
        "important for business": "specific relevance to business objectives",
        "good for the company": "specific positive outcome",
        "makes profit": "profit margin, revenue growth, or financial metric",
        "increases sales": "sales data, market share, or growth percentage",
    },
}


def extract_concepts_from_text(text: str, topic: str) -> list[str]:
    """Extract business concepts from text."""
    text_lower = text.lower()
    concepts_found = []

    topic_data = SYLLABUS_CONCEPTS.get(topic, {})
    required = topic_data.get("required", [])
    recommended = topic_data.get("recommended", [])

    for concept in required + recommended:
        if concept.lower() in text_lower:
            concepts_found.append(concept)

    return list(set(concepts_found))


def find_missing_concepts(text: str, topic: str, required_count: int = 2) -> list[str]:
    """Find required concepts that are missing from the answer."""
    text_lower = text.lower()
    topic_data = SYLLABUS_CONCEPTS.get(topic, {})
    required = topic_data.get("required", [])

    missing = []
    for concept in required:
        if concept.lower() not in text_lower:
            missing.append(concept)

    return missing[:required_count]


def check_case_evidence(answer: str, case_context: str = "") -> list[dict[str, Any]]:
    """Check for case study evidence in answer."""
    issues = []

    answer_lower = answer.lower()

    has_strong_evidence = any(
        re.search(pattern, answer_lower) for pattern in CASE_EVIDENCE_MARKERS["strong"]
    )

    has_weak_evidence = any(
        re.search(pattern, answer_lower) for pattern in CASE_EVIDENCE_MARKERS["weak"]
    )

    if case_context and not has_strong_evidence:
        issues.append(
            {
                "type": "missing_case_evidence",
                "severity": "error",
                "description": "Answer does not incorporate case context provided",
                "original_text": "",
                "suggestion": f"Reference specific facts from the case: {case_context[:50]}...",
            }
        )
    elif not case_context and not has_strong_evidence and len(answer) > 100:
        issues.append(
            {
                "type": "generic_evidence",
                "severity": "warning",
                "description": "Answer lacks specific case study evidence",
                "original_text": "",
                "suggestion": "Include specific business names, dates, data, or facts from case studies",
            }
        )

    if has_weak_evidence:
        weak_matches = []
        for pattern in CASE_EVIDENCE_MARKERS["weak"]:
            matches = re.findall(pattern, answer_lower)
            weak_matches.extend(matches)
        if weak_matches:
            issues.append(
                {
                    "type": "weak_evidence",
                    "severity": "warning",
                    "description": "Evidence is too generic",
                    "original_text": weak_matches[0],
                    "suggestion": "Replace generic terms with specific case facts",
                }
            )

    return issues


def check_judgement(answer: str) -> list[dict[str, Any]]:
    """Check for judgement and conclusion in answer."""
    issues = []

    answer_lower = answer.lower()
    sentences = re.split(r"[.!?]+", answer)
    sentences = [s.strip() for s in sentences if s.strip()]

    has_strong_judgement = any(
        re.search(pattern, answer_lower) for pattern in JUDGEMENT_MARKERS["strong"]
    )

    has_weak_judgement = any(
        re.search(pattern, answer_lower) for pattern in JUDGEMENT_MARKERS["weak"]
    )

    last_sentence = sentences[-1] if sentences else ""
    has_conclusion_words = any(
        word in last_sentence.lower()
        for word in ["therefore", "thus", "in conclusion", "overall", "finally"]
    )

    if not has_strong_judgement and not has_conclusion_words and len(answer) > 150:
        issues.append(
            {
                "type": "missing_judgement",
                "severity": "error",
                "description": "Answer lacks clear judgement or conclusion",
                "original_text": "",
                "suggestion": "Add a concluding statement with clear recommendation or evaluation",
            }
        )

    if has_weak_judgement:
        weak_matches = []
        for pattern in JUDGEMENT_MARKERS["weak"]:
            matches = re.findall(pattern, answer_lower)
            weak_matches.extend(matches)
        if weak_matches:
            issues.append(
                {
                    "type": "weak_judgement",
                    "severity": "warning",
                    "description": "Judgement is tentative or vague",
                    "original_text": weak_matches[0],
                    "suggestion": "Use definitive language: 'should', 'therefore', 'is recommended'",
                }
            )

    return issues


def check_syllabus_links(answer: str, topic: str) -> list[dict[str, Any]]:
    """Check for explicit syllabus concept links."""
    issues = []

    topic_data = SYLLABUS_CONCEPTS.get(topic, {})
    frameworks = topic_data.get("frameworks", [])

    answer_lower = answer.lower()

    has_framework = any(fw.lower() in answer_lower for fw in frameworks)
    has_syllabus_link = any(
        re.search(pattern, answer_lower)
        for pattern in [
            r"\baccording to (?:the )?syllabus",
            r"\bsyllabus (?:requires|states|defines)",
        ]
    )

    if not has_framework and not has_syllabus_link and len(answer) > 100:
        issues.append(
            {
                "type": "missing_syllabus_link",
                "severity": "warning",
                "description": "Answer does not explicitly link to syllabus concepts or frameworks",
                "original_text": "",
                "suggestion": f"Reference relevant frameworks: {', '.join(frameworks[:3])}",
            }
        )

    return issues


def check_structure(answer: str) -> list[str]:
    """Check answer structure completeness."""
    issues = []

    sentences = re.split(r"[.!?]+", answer)
    sentences = [s.strip() for s in sentences if s.strip()]

    has_definition = any(
        re.search(pattern, sentence.lower())
        for sentence in sentences[:3]
        for pattern in STRUCTURE_MARKERS["definition"]
    )

    has_analysis = any(
        re.search(pattern, answer.lower()) for pattern in STRUCTURE_MARKERS["analysis"]
    )

    has_evidence = any(
        re.search(pattern, answer.lower()) for pattern in STRUCTURE_MARKERS["evidence"]
    )

    has_judgement = any(
        re.search(pattern, answer.lower()) for pattern in STRUCTURE_MARKERS["judgement"]
    )

    if not has_definition and len(sentences) >= 2:
        issues.append("Missing definition or concept introduction")

    if not has_analysis and len(sentences) >= 3:
        issues.append("Missing analytical content (causes, effects, relationships)")

    if not has_evidence and len(sentences) >= 3:
        issues.append("Missing specific examples or case evidence")

    if not has_judgement and len(sentences) >= 4:
        issues.append("Missing clear conclusion or judgement")

    return issues


def check_generic_phrases(answer: str) -> list[dict[str, Any]]:
    """Check for overly generic business phrases."""
    issues = []

    for pattern in GENERIC_PHRASES["to_avoid"]:
        matches = re.findall(pattern, answer.lower())
        if matches:
            for match in matches[:2]:
                preferred = ""
                for avoid, prefer in GENERIC_PHRASES["preferred_alternatives"].items():
                    if avoid in match or re.search(avoid, match):
                        preferred = prefer
                        break

                issues.append(
                    {
                        "type": "generic_phrase",
                        "severity": "warning",
                        "description": f"Generic phrase: '{match}'",
                        "original_text": match,
                        "suggestion": f"Replace with more specific language: '{preferred}'"
                        if preferred
                        else "Be more specific",
                    }
                )

    return issues


def validate_business_answer(
    question: str,
    answer: str,
    topic: str = "operations",
    case_context: str = "",
    required_concepts: int = 2,
) -> BusinessGuardrailResult:
    """
    Validate Business Studies answer quality.

    Args:
        question: The Business Studies question
        answer: Student's answer
        topic: Topic area (operations, marketing, finance, etc.)
        case_context: Optional case study context provided
        required_concepts: Minimum required syllabus concepts

    Returns:
        BusinessGuardrailResult with issues and suggestions
    """
    issues = []

    concepts_found = extract_concepts_from_text(answer, topic)
    missing_concepts = find_missing_concepts(answer, topic, required_concepts)

    evidence_issues = check_case_evidence(answer, case_context)
    issues.extend([BusinessIssue(**i) for i in evidence_issues])

    judgement_issues = check_judgement(answer)
    issues.extend([BusinessIssue(**i) for i in judgement_issues])

    syllabus_issues = check_syllabus_links(answer, topic)
    issues.extend([BusinessIssue(**i) for i in syllabus_issues])

    generic_issues = check_generic_phrases(answer)
    issues.extend([BusinessIssue(**i) for i in generic_issues])

    structure_issues = check_structure(answer)

    missing_evidence = [
        i["description"]
        for i in evidence_issues
        if i["type"] in ["missing_case_evidence", "generic_evidence"]
    ]
    missing_judgement = [
        i["description"]
        for i in judgement_issues
        if i["type"] in ["missing_judgement", "weak_judgement"]
    ]
    missing_syllabus_links = [i["description"] for i in syllabus_issues]

    rewrite_suggestions = []
    for issue in issues[:5]:
        if issue.suggestion:
            rewrite_suggestions.append(f"• {issue.description}: {issue.suggestion}")

    score = 100.0
    for issue in issues:
        if issue.severity == "error":
            score -= 8.0
        elif issue.severity == "warning":
            score -= 3.0
        else:
            score -= 1.0

    score -= len(missing_concepts) * 5.0
    score -= len(structure_issues) * 5.0
    score = max(0, min(100, score))

    is_acceptable = score >= 70 and len(missing_judgement) == 0 and len(missing_evidence) == 0

    evidence_found = []
    for pattern in CASE_EVIDENCE_MARKERS["strong"]:
        matches = re.findall(pattern, answer.lower())
        evidence_found.extend(matches[:2])

    exemplar = ""
    if not is_acceptable:
        exemplar = generate_exemplar_paragraph(question, topic, case_context)

    return BusinessGuardrailResult(
        is_acceptable=is_acceptable,
        score=round(score, 1),
        issues=issues,
        missing_concepts=missing_concepts,
        missing_evidence=missing_evidence,
        missing_judgement=missing_judgement,
        missing_syllabus_links=missing_syllabus_links,
        structure_issues=structure_issues,
        rewrite_suggestions=rewrite_suggestions,
        concepts_found=concepts_found,
        evidence_found=evidence_found[:5],
        exemplar_paragraph=exemplar,
    )


def generate_exemplar_paragraph(
    question: str,
    topic: str,
    case_context: str = "",
) -> str:
    """Generate an exemplar paragraph for Business Studies answer."""
    topic_data = SYLLABUS_CONCEPTS.get(topic, {})
    frameworks = topic_data.get("frameworks", [])
    required = topic_data.get("required", [])[:3]

    framework_str = f" using frameworks such as {', '.join(frameworks[:2])}" if frameworks else ""
    case_str = (
        f" In the case of {case_context.split('.')[0] if case_context else 'a relevant business'},"
        if case_context
        else ""
    )

    exemplar = f"""**Exemplar Structure:**

{case_str} The concept of {required[0] if required else "business operations"} refers to{"" if not required else " the systematic process of managing resources and processes to achieve organisational objectives"}. This is significant because{framework_str and f" it allows businesses to {frameworks[0].lower() if frameworks else 'achieve competitive advantage'} effectively"}.

Analysis shows that{"" if not case_context else f" based on the case evidence, {case_context[:100]}..."} factors such as market conditions and internal capabilities influence outcomes. For example,{case_context and f" the case demonstrates" or " research indicates"} effective {required[0] if required else "planning"} leads to improved performance.

Therefore, it is recommended that businesses{"" if not required else f" prioritise {required[1] if len(required) > 1 else required[0]} and"} implement evidence-based strategies to achieve their objectives."""

    return exemplar


def get_syllabus_concepts(topic: str) -> dict[str, list[str]]:
    """Get required and recommended concepts for a topic."""
    return SYLLABUS_CONCEPTS.get(topic, {"required": [], "recommended": [], "frameworks": []})


def generate_case_pack_metadata(
    case_name: str,
    file_path: str,
    topic: str = "general",
) -> dict[str, Any]:
    """Generate metadata for case pack ingestion."""
    return {
        "subject": "business_studies",
        "source_type": "case_notes",
        "topic": topic,
        "module": "hsc",
        "year": 2024,
        "case_name": case_name,
        "source_file": file_path,
        "tags": ["case_pack", f"case_{case_name.lower().replace(' ', '_')}", topic],
    }
