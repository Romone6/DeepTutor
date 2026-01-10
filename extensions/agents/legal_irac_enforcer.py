"""
Legal Studies IRAC/ILAC Enforcer
=================================

Enforces HSC Legal Studies answer structure using IRAC (Issue, Rule, Application, Conclusion)
or ILAC (Issue, Law, Application, Conclusion) framework.

Usage:
    from extensions.agents.legal_irac_enforcer import (
        LegalIRACResult,
        validate_irac_structure,
        check_citation_format,
        suggest_irac_improvements,
    )

    result = validate_irac_structure(
        question="Discuss the effectiveness of mirror self-incrimination protections",
        answer="The answer text...",
        topic="crime",
    )
    print(result.structure_issues)
"""

import re
from dataclasses import dataclass, field
from typing import Any

from src.logging import get_logger

logger = get_logger("LegalIRACEnforcer")


@dataclass
class LegalIssue:
    """A detected issue in Legal Studies answer."""

    type: str
    severity: str  # "error", "warning", "info"
    description: str
    location: str = ""  # Which IRAC section
    suggestion: str = ""
    mark_impact: float = 0.0


@dataclass
class LegalIRACResult:
    """Complete result of validating Legal Studies IRAC/ILAC structure."""

    is_acceptable: bool
    score: float
    issues: list[LegalIssue]
    has_issue: bool
    has_rule_law: bool
    has_application: bool
    has_conclusion: bool
    structure_score: float
    citation_count: int
    missing_citations: list[str]
    structure_issues: list[str] = field(default_factory=list)
    rewrite_suggestions: list[str] = field(default_factory=list)
    irac_scaffold: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_acceptable": self.is_acceptable,
            "score": self.score,
            "issues": [i.__dict__ for i in self.issues],
            "has_issue": self.has_issue,
            "has_rule_law": self.has_rule_law,
            "has_application": self.has_application,
            "has_conclusion": self.has_conclusion,
            "structure_score": self.structure_score,
            "citation_count": self.citation_count,
            "missing_citations": self.missing_citations,
            "structure_issues": self.structure_issues,
            "rewrite_suggestions": self.rewrite_suggestions,
            "irac_scaffold": self.irac_scaffold,
        }


IRAC_MARKERS = {
    "issue": [
        r"^\s*\*?\s*[Ii]ssue\s*[:]*\s*$",
        r"^\s*\*?\s*[Ll]egal\s+[Ii]ssue\s*[:]*\s*$",
        r"^\s*\*?\s*[Tt]he\s+[Ii]ssue\s*(?:is|relates|concerns)\b",
        r"\bthe issue\s+(?:is|relates to|concerns|arises from)\b",
    ],
    "rule_law": [
        r"^\s*\*?\s*[Rr]ule\s*[:]*\s*$",
        r"^\s*\*?\s*[Ll]aw\s*[:]*\s*$",
        r"^\s*\*?\s*[Rr]elevant\s+[Ll]aw\s*[:]*\s*$",
        r"^\s*\*?\s*[Ll]egal\s+[Pp]rinciples?\s*[:]*\s*$",
        r"\bpursuant to\b",
        r"\bunder (?:section|regulation)\b",
        r"\bSection\s+\d+",
        r"\bCrimes\s+Act\s+\d{4}",
    ],
    "application": [
        r"^\s*\*?\s*[Aa]pplication\s*[:]*\s*$",
        r"^\s*\*?\s*[Aa]pplication\s+to\s+[Ff]acts\s*[:]*\s*$",
        r"^\s*\*?\s*[Aa]nalysis\s*[:]*\s*$",
        r"\bIn this case,\s+(?:the )?[A-Z]",
        r"\bApplying\s+(?:this|the)\b",
        r"\btherefore,\s+(?:the )?[A-Z][a-z]+\s+",
    ],
    "conclusion": [
        r"^\s*\*?\s*[Cc]onclusion\s*[:]*\s*$",
        r"^\s*\*?\s*[Cc]onclusion\s+and\s+[Ff]indings?\s*[:]*\s*$",
        r"\bin\s+conclusion\b",
        r"\bTherefore,\s+(?:the )?[A-Z][a-z]+\s+(?:is|was|has|should)",
    ],
}

CITATION_PATTERNS = {
    "cases": [
        r"[A-Z][a-z]+\s+v\s+[A-Z][a-z]+",
        r"\([A-Z][a-z]+\s+\d{4}\)",
        r"\d+\s+(?:C\.L\.R\.|S\.R\.|A\.L\.R\.|N\.S\.W\.R\.|V\.L\.R\.)",
    ],
    "legislation": [
        r"(?:Criminal|Civil|Evidence|Trade Practices|Competition and Consumer|Bills of Exchange)\s+(?:Code\s+)?Act\s+(?:\d{4}|\d{3,4})",
        r"section\s+\d+(?:\(\d+\))?(?:\s*\(?\w+\)?)?",
        r"s\.\s*\d+(?:\(\d+\))?",
    ],
    "reports": [
        r"(?:ALRC|LRC|ALJP|HREOC)\s+(?:Report\s+)?No\.?\s*\d+",
        r"\d+\s+CLR",
    ],
}

VALID_AUTHORITY_TYPES = [
    "case",
    "legislation",
    "treaty",
    "international agreement",
    "law reform report",
    "statistics",
    "contemporary media",
]

SECTION_TITLES = [
    ("issue", ["ISSUE", "LEGAL ISSUE", "THE ISSUE"]),
    ("rule", ["RULE", "LAW", "RELEVANT LAW", "THE LAW", "LEGAL PRINCIPLES"]),
    ("application", ["APPLICATION", "APPLICATION TO FACTS", "ANALYSIS"]),
    ("conclusion", ["CONCLUSION", "CONCLUSION AND FINDINGS"]),
]

HEADING_PATTERN = re.compile(
    r"^\s*(?:\*?\s*)?(?:###?\s*)?(?:ISSUE|LEGAL ISSUE|THE ISSUE|RULE|LAW|RELEVANT LAW|THE LAW|LEGAL PRINCIPLES|APPLICATION|APPLICATION TO FACTS|ANALYSIS|CONCLUSION|CONCLUSION AND FINDINGS)\s*[:*#]*\s*$",
    re.IGNORECASE,
)


def detect_structure_sections(text: str) -> dict[str, list[tuple[str, int, int]]]:
    """Detect IRAC sections in the text."""
    sections = {"issue": [], "rule_law": [], "application": [], "conclusion": []}

    lines = text.split("\n")
    current_section = None
    current_start = 0

    for i, line in enumerate(lines):
        heading_match = HEADING_PATTERN.match(line)
        if heading_match:
            if current_section is not None:
                sections[current_section].append((text[current_start:i], current_start, i))
            current_section = None
            for section_name, titles in SECTION_TITLES:
                for title in titles:
                    if title in line.upper():
                        current_section = section_name if section_name != "rule" else "rule_law"
                        current_start = i + 1
                        break
                if current_section:
                    break

    if current_section is not None:
        sections[current_section].append((text[current_start:], current_start, len(lines)))

    for section, markers in IRAC_MARKERS.items():
        if section == "rule_law":
            section = "rule"
        for marker in markers:
            if section in sections:
                for content, start, end in sections[section]:
                    if re.search(marker, content, re.IGNORECASE):
                        break
                else:
                    continue
                break

    return sections


def has_section_content(text: str, section_type: str) -> bool:
    """Check if a section has substantive content."""
    markers = IRAC_MARKERS.get(section_type, [])
    if section_type == "rule_law":
        markers = IRAC_MARKERS.get("rule", []) + IRAC_MARKERS.get("law", [])

    text_lower = text.lower()
    for marker in markers:
        if re.search(marker, text_lower):
            return True

    if section_type == "issue":
        if len(text.strip()) > 50:
            return True

    if section_type == "rule_law":
        if len(text.strip()) > 80:
            return True

    if section_type == "application":
        if len(text.strip()) > 100:
            return True

    if section_type == "conclusion":
        if len(text.strip()) > 30:
            return True

    return False


def check_irac_structure(text: str, min_section_lengths: dict[str, int] = None) -> dict[str, Any]:
    """Check IRAC/ILAC structure completeness."""
    if min_section_lengths is None:
        min_section_lengths = {
            "issue": 30,
            "rule_law": 60,
            "application": 80,
            "conclusion": 30,
        }

    lines = text.split("\n")
    section_found = {k: False for k in IRAC_MARKERS.keys()}
    section_content = {k: "" for k in IRAC_MARKERS.keys()}
    section_start = {k: -1 for k in IRAC_MARKERS.keys()}

    current_section = None
    for i, line in enumerate(lines):
        heading_match = HEADING_PATTERN.match(line)
        if heading_match:
            current_section = None
            for section_name, titles in SECTION_TITLES:
                for title in titles:
                    if title in line.upper():
                        current_section = section_name
                        if section_name == "rule":
                            current_section = "rule_law"
                        section_start[current_section] = i
                        break
                if current_section:
                    break

    for section_type in IRAC_MARKERS.keys():
        section_key = section_type if section_type != "rule_law" else "rule_law"
        content = section_content.get(section_key, "")

        for marker in IRAC_MARKERS.get(section_type, []):
            if re.search(marker, text, re.IGNORECASE):
                section_found[section_type] = True
                break

    has_issue = section_found.get("issue", False)
    has_rule_law = section_found.get("rule_law", False)
    has_application = section_found.get("application", False)
    has_conclusion = section_found.get("conclusion", False)

    structure_score = 0.0
    if has_issue:
        structure_score += 25
    if has_rule_law:
        structure_score += 25
    if has_application:
        structure_score += 25
    if has_conclusion:
        structure_score += 25

    return {
        "has_issue": has_issue,
        "has_rule_law": has_rule_law,
        "has_application": has_application,
        "has_conclusion": has_conclusion,
        "structure_score": structure_score,
        "section_found": section_found,
    }


def check_citation_format(text: str) -> tuple[int, list[str], list[str]]:
    """
    Check citation format in text.

    Returns:
        citation_count, valid_citations, invalid_citations
    """
    valid_citations = []
    invalid_citations = []

    case_pattern = r"[A-Z][a-zA-Z\s&']+\s+v\s+[A-Z][a-zA-Z\s&']+"
    case_matches = re.findall(case_pattern, text)
    for match in case_matches:
        if len(match.strip()) > 5:
            valid_citations.append(f"Case: {match.strip()}")

    leg_pattern = r"(?:Criminal|Civil|Evidence|Trade Practices|Competition and Consumer|Bills of Exchange)\s+(?:Code\s+)?Act\s+(?:\d{4}|\d{3,4})"
    leg_matches = re.findall(leg_pattern, text)
    for match in leg_matches:
        valid_citations.append(f"Legislation: {match.strip()}")

    section_pattern = r"(?:section|s\.|s\.)?\s*\d+(?:\(\d+\))?(?:\s*\(?\w+\)?)?"
    section_matches = re.findall(section_pattern, text, re.IGNORECASE)
    for match in section_matches:
        if len(match.strip()) > 2:
            valid_citations.append(f"Section: {match.strip()}")

    citation_count = len(valid_citations)

    vague_references = [
        r"\bthe\s+act\b(?!\s+\d{4})",
        r"\bthe\s+case\b(?!\s+[A-Z])",
        r"\bthis\s+law\b(?!\s+\d+)",
        r"\bsome\s+legislation\b",
        r"\b(a )?court\s+(?:case|ruling|decision)\b",
    ]
    for pattern in vague_references:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0] + match[1] if match[0] else match[1]
            invalid_citations.append(f"Vague reference: '{match.strip()}'")

    return citation_count, valid_citations, invalid_citations


def validate_irac_structure(
    question: str,
    answer: str,
    topic: str = "crime",
    retrieved_authorities: list[dict] = None,
    require_citations: bool = True,
    min_citations: int = 2,
) -> LegalIRACResult:
    """
    Validate Legal Studies answer IRAC/ILAC structure.

    Args:
        question: The Legal Studies question
        answer: Student's answer
        topic: Topic area (crime, human_rights, family, consumers, workplace)
        retrieved_authorities: List of authority snippets from KB
        require_citations: Whether citations are required
        min_citations: Minimum number of citations required

    Returns:
        LegalIRACResult with structure analysis
    """
    issues = []

    structure_result = check_irac_structure(answer)

    has_issue = structure_result["has_issue"]
    has_rule_law = structure_result["has_rule_law"]
    has_application = structure_result["has_application"]
    has_conclusion = structure_result["has_conclusion"]

    if not has_issue:
        issues.append(
            LegalIssue(
                type="missing_issue",
                severity="error",
                description="IRAC structure: Issue not clearly identified",
                suggestion="Start by identifying the key legal issue or question. Use: 'The issue is...' or 'This case concerns...'",
                mark_impact=1.0,
            )
        )

    if not has_rule_law:
        issues.append(
            LegalIssue(
                type="missing_rule_law",
                severity="error",
                description="IRAC structure: Rule/Law section missing",
                suggestion="State the relevant legal principles, legislation, or case law. Use: 'The law states...' or cite section numbers.",
                mark_impact=1.5,
            )
        )

    if not has_application:
        issues.append(
            LegalIssue(
                type="missing_application",
                severity="error",
                description="IRAC structure: Application to facts missing",
                suggestion="Apply the law to the specific facts. Use: 'In this case...' or 'Therefore...' to link law to facts.",
                mark_impact=1.5,
            )
        )

    if not has_conclusion:
        issues.append(
            LegalIssue(
                type="missing_conclusion",
                severity="warning",
                description="IRAC structure: Conclusion missing or weak",
                suggestion="Provide a clear conclusion based on your analysis. Use: 'Therefore...' or 'In conclusion...'",
                mark_impact=0.5,
            )
        )

    citation_count, valid_citations, invalid_citations = check_citation_format(answer)
    missing_citations = []

    if require_citations and retrieved_authorities:
        authority_types = set()
        for auth in retrieved_authorities:
            doc_type = auth.get("metadata", {}).get("source_type", "")
            if doc_type:
                authority_types.add(doc_type)

        required_types = {"case", "legislation"}
        missing_types = required_types - authority_types

        if missing_types:
            missing_citations.append(f"Missing {', '.join(missing_types)} citations")

        if citation_count < min_citations:
            issues.append(
                LegalIssue(
                    type="insufficient_citations",
                    severity="warning",
                    description=f"Insufficient citations: {citation_count} found, {min_citations} required",
                    suggestion="Include more case law and legislation citations from the knowledge base",
                    mark_impact=0.5,
                )
            )

    for invalid in invalid_citations[:3]:
        issues.append(
            LegalIssue(
                type="vague_citation",
                severity="warning",
                description=invalid,
                suggestion="Replace with specific case name, citation, or legislation reference",
                mark_impact=0.25,
            )
        )

    structure_issues = []
    if not has_issue:
        structure_issues.append("Missing Issue section")
    if not has_rule_law:
        structure_issues.append("Missing Rule/Law section")
    if not has_application:
        structure_issues.append("Missing Application section")
    if not has_conclusion:
        structure_issues.append("Missing or weak Conclusion")

    rewrite_suggestions = []
    for issue in issues[:5]:
        if issue.suggestion:
            rewrite_suggestions.append(f"• {issue.description}: {issue.suggestion}")

    score = structure_result["structure_score"]
    score -= len([i for i in issues if i.severity == "error"]) * 10
    score -= len([i for i in issues if i.severity == "warning"]) * 5
    score = max(0, min(100, score))

    irac_scaffold = {}
    if not has_issue:
        irac_scaffold["issue"] = (
            "**ISSUE:** What is the legal question or problem presented in this scenario?"
        )
    if not has_rule_law:
        irac_scaffold["rule_law"] = (
            "**RULE/LAW:** What are the relevant legal principles, legislation, or case law?"
        )
    if not has_application:
        irac_scaffold["application"] = (
            "**APPLICATION:** How do the facts apply to the law? What conclusions follow?"
        )
    if not has_conclusion:
        irac_scaffold["conclusion"] = (
            "**CONCLUSION:** Based on the analysis, what is the final determination?"
        )

    is_acceptable = score >= 70 and has_issue and has_rule_law and has_application

    return LegalIRACResult(
        is_acceptable=is_acceptable,
        score=round(score, 1),
        issues=issues,
        has_issue=has_issue,
        has_rule_law=has_rule_law,
        has_application=has_application,
        has_conclusion=has_conclusion,
        structure_score=structure_result["structure_score"],
        citation_count=citation_count,
        missing_citations=missing_citations,
        structure_issues=structure_issues,
        rewrite_suggestions=rewrite_suggestions,
        irac_scaffold=irac_scaffold,
    )


def suggest_irac_improvements(result: LegalIRACResult, topic: str = "crime") -> list[str]:
    """Generate specific IRAC improvement suggestions."""
    suggestions = []

    if not result.has_issue:
        suggestions.append(
            "**ISSUE**: Clearly identify the legal question. E.g., 'The issue is whether [party] committed [offence] because...'"
        )

    if not result.has_rule_law:
        suggestions.append(
            "**RULE/LAW**: State the relevant law. Cite specific sections or cases. E.g., 'Section 61 of the Crimes Act 1900 (NSW) provides that...'"
        )

    if not result.has_application:
        suggestions.append(
            "**APPLICATION**: Apply the law to facts. E.g., 'Here, the accused [actions], which satisfies the elements of [offence] because...'"
        )

    if not result.has_conclusion:
        suggestions.append(
            "**CONCLUSION**: Provide a clear answer. E.g., 'Therefore, the accused is guilty/not guilty of [offence]'"
        )

    if result.citation_count < 2:
        suggestions.append(
            "**CITATIONS**: Include at least 2 relevant case or legislative citations from your notes or the syllabus"
        )

    return suggestions


def generate_irac_scaffold(question: str, topic: str = "crime") -> dict[str, str]:
    """Generate a complete IRAC scaffold for a question."""
    return {
        "issue": f"**ISSUE:** What is the legal question raised by '{question}'?",
        "rule_law": f"**RULE/LAW:** What relevant legislation, case law, or legal principles apply to this {topic} issue?",
        "application": f"**APPLICATION:** How do the facts of this scenario satisfy or fail to satisfy the legal elements?",
        "conclusion": f"**CONCLUSION:** Based on the application of the law to the facts, what is the legal outcome?",
    }
