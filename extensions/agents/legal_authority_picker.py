"""
Legal Authority Picker
======================

Ensures Legal Studies answers only cite authorities from retrieved knowledge base snippets.
Prevents hallucination of cases, legislation, or other legal authorities.

Usage:
    from extensions.agents.legal_authority_picker import (
        AuthorityCandidate,
        pick_authorities,
        validate_citation,
        format_citation,
        extract_authorities_from_kb,
    )

    candidates = pick_authorities(
        question="Discuss the defence of mental illness",
        retrieved_snippets=kb_results,
        max_authorities=3,
    )
    print(candidates[0].formatted_citation)
"""

import re
from dataclasses import dataclass, field
from typing import Any

from src.logging import get_logger

logger = get_logger("LegalAuthorityPicker")


@dataclass
class AuthorityCandidate:
    """A candidate legal authority from KB."""

    doc_id: str
    source_type: str  # "case", "legislation", "syllabus", "marking_guide"
    title: str
    citation: str
    content: str
    relevance_score: float
    year: int = 0
    jurisdiction: str = ""
    page_ref: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def formatted_citation(self) -> str:
        """Format citation for use in answer."""
        if self.source_type == "case":
            return f"{self.title} ({self.year})" if self.year else self.title
        elif self.source_type == "legislation":
            return self.title
        elif self.source_type == "syllabus":
            return f"Syllabus: {self.title}"
        else:
            return self.title

    @property
    def reference(self) -> str:
        """Get short reference for KB source display."""
        if self.source_type == "case":
            return f"[{self.doc_id}] {self.title}"
        elif self.source_type == "legislation":
            return f"[{self.doc_id}] {self.title}"
        elif self.source_type == "syllabus":
            return f"[{self.doc_id}]"
        else:
            return f"[{self.doc_id}]"

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source_type": self.source_type,
            "title": self.title,
            "citation": self.citation,
            "content_preview": self.content[:200] + "..."
            if len(self.content) > 200
            else self.content,
            "relevance_score": self.relevance_score,
            "year": self.year,
            "jurisdiction": self.jurisdiction,
            "formatted_citation": self.formatted_citation,
            "reference": self.reference,
            "metadata": self.metadata,
        }


@dataclass
class AuthorityValidationResult:
    """Result of validating a citation against KB."""

    is_valid: bool
    citation_text: str
    matched_authority: AuthorityCandidate = None
    suggestion: str = ""
    hallucinated: bool = False


CASE_NAME_PATTERN = re.compile(r"^[A-Z][a-zA-Z\s&']+\s+v\s+[A-Z][a-zA-Z\s&']+(?:\s+\(\d{4}\))?$")

LEGAL_CITATION_PATTERN = re.compile(
    r"(?:\d+\s+)?(?:CLR|S\.R\.|N\.S\.W\.R\.|V\.L\.R\.|A\.L\.R\.)\s+\d+"
)

SECTION_REFERENCE_PATTERN = re.compile(r"(?:section|s\.|s\s+)\s*(\d+(?:\(\d+\))?)", re.IGNORECASE)

ACT_REFERENCE_PATTERN = re.compile(
    r"(?:Criminal|Civil|Evidence|Trade Practices|Competition and Consumer|Bills of Exchange|Law Enforcement)\s+(?:Code\s+)?Act\s+(?:\d{4}|\d{3,4})",
    re.IGNORECASE,
)


def extract_authorities_from_kb(snippets: list[dict]) -> list[AuthorityCandidate]:
    """
    Extract authority candidates from KB retrieval results.

    Args:
        snippets: List of dicts with 'content', 'metadata', 'score'

    Returns:
        List of AuthorityCandidate objects
    """
    authorities = []

    for snippet in snippets:
        metadata = snippet.get("metadata", {})
        content = snippet.get("content", "")
        score = snippet.get("score", 0.5)

        doc_id = metadata.get("doc_id", "unknown")
        source_type = metadata.get("source_type", "general")

        title = ""
        if "question" in metadata:
            title = metadata["question"]
        elif "topic" in metadata:
            title = metadata["topic"]
        elif "case_name" in metadata:
            title = metadata["case_name"]
        else:
            title = doc_id

        authority = AuthorityCandidate(
            doc_id=doc_id,
            source_type=source_type,
            title=title,
            citation="",
            content=content,
            relevance_score=score,
            year=metadata.get("year", 0),
            jurisdiction=metadata.get("jurisdiction", ""),
            metadata=metadata,
        )

        if source_type == "case":
            case_match = re.search(CASE_NAME_PATTERN, content[:500])
            if case_match:
                authority.citation = case_match.group().strip()

        elif source_type == "legislation":
            act_match = re.search(ACT_REFERENCE_PATTERN, content[:500])
            if act_match:
                authority.citation = act_match.group().strip()

        authorities.append(authority)

    return authorities


def pick_authorities(
    question: str,
    retrieved_snippets: list[dict],
    topic: str = "crime",
    max_authorities: int = 5,
    prefer_cases: bool = True,
) -> list[AuthorityCandidate]:
    """
    Select appropriate authorities from KB for a question.

    Args:
        question: The legal question
        retrieved_snippets: KB retrieval results
        topic: Topic area
        max_authorities: Maximum authorities to return
        prefer_cases: Whether to prioritize case law

    Returns:
        List of AuthorityCandidate sorted by relevance
    """
    authorities = extract_authorities_from_kb(retrieved_snippets)

    if not authorities:
        return []

    filtered = []
    case_count = 0
    legislation_count = 0

    for auth in authorities:
        if auth.source_type == "case" and case_count < 3:
            filtered.append(auth)
            case_count += 1
        elif auth.source_type == "legislation" and legislation_count < 2:
            filtered.append(auth)
            legislation_count += 1
        elif auth.source_type in ["syllabus", "marking_guide"]:
            filtered.append(auth)

    filtered.sort(key=lambda x: x.relevance_score, reverse=True)

    return filtered[:max_authorities]


def validate_citation(
    citation: str,
    valid_authorities: list[AuthorityCandidate],
) -> AuthorityValidationResult:
    """
    Validate if a citation matches an authority in the KB.

    Args:
        citation: The citation text from the answer
        valid_authorities: List of valid authority candidates

    Returns:
        AuthorityValidationResult with validation status
    """
    citation_clean = citation.strip()

    for authority in valid_authorities:
        authority_citation = authority.citation.lower().strip()
        citation_lower = citation_clean.lower()

        if authority_citation and authority_citation in citation_lower:
            return AuthorityValidationResult(
                is_valid=True,
                citation_text=citation_clean,
                matched_authority=authority,
                suggestion="",
                hallucinated=False,
            )

        if authority.title.lower() in citation_lower:
            return AuthorityValidationResult(
                is_valid=True,
                citation_text=citation_clean,
                matched_authority=authority,
                suggestion="",
                hallucinated=False,
            )

    return AuthorityValidationResult(
        is_valid=False,
        citation_text=citation_clean,
        matched_authority=None,
        suggestion="Citation not found in knowledge base. Use a case or legislation from your notes.",
        hallucinated=True,
    )


def validate_all_citations(
    answer: str,
    valid_authorities: list[AuthorityCandidate],
) -> list[AuthorityValidationResult]:
    """
    Validate all citations in an answer against KB authorities.

    Args:
        answer: The student's answer
        valid_authorities: List of valid authority candidates

    Returns:
        List of AuthorityValidationResult for each citation
    """
    results = []

    case_pattern = r"[A-Z][a-zA-Z\s&']+\s+v\s+[A-Z][a-zA-Z\s&']+"
    case_matches = re.findall(case_pattern, answer)

    for case in case_matches:
        result = validate_citation(case, valid_authorities)
        results.append(result)

    act_pattern = ACT_REFERENCE_PATTERN
    act_matches = re.findall(act_pattern, answer)

    for act in act_matches:
        result = validate_citation(act, valid_authorities)
        if not any(r.citation_text == act and r.is_valid for r in results):
            results.append(result)

    section_pattern = r"(?:section|s\.?)\s*\d+(?:\(\d+\))?"
    section_matches = re.findall(section_pattern, answer, re.IGNORECASE)

    for section in section_matches:
        result = validate_citation(section, valid_authorities)
        if not any(r.citation_text == section and r.is_valid for r in results):
            results.append(result)

    return results


def format_authority_for_answer(
    authority: AuthorityCandidate,
    integration_point: str = "rule",
) -> str:
    """
    Format an authority for integration into an answer.

    Args:
        authority: The authority to format
        integration_point: Where to use it (rule, application, conclusion)

    Returns:
        Formatted citation text
    """
    if authority.source_type == "case":
        if integration_point == "rule":
            return f"In {authority.formatted_citation}, the court held that..."
        elif integration_point == "application":
            return f"Applying {authority.formatted_citation} to these facts..."
        else:
            return authority.formatted_citation

    elif authority.source_type == "legislation":
        if integration_point == "rule":
            return f"Section {authority.citation} of {authority.title} provides that..."
        else:
            return authority.title

    elif authority.source_type == "syllabus":
        return f"According to the {authority.title} syllabus area..."

    return authority.formatted_citation


def get_authorities_panel_data(authorities: list[AuthorityCandidate]) -> dict[str, Any]:
    """
    Generate data for the UI authorities panel.

    Args:
        authorities: List of selected authorities

    Returns:
        Dict with panel data
    """
    cases = [a for a in authorities if a.source_type == "case"]
    legislation = [a for a in authorities if a.source_type == "legislation"]
    other = [a for a in authorities if a.source_type not in ["case", "legislation"]]

    return {
        "total_count": len(authorities),
        "cases": {
            "count": len(cases),
            "authorities": [a.to_dict() for a in cases],
        },
        "legislation": {
            "count": len(legislation),
            "authorities": [a.to_dict() for a in legislation],
        },
        "other": {
            "count": len(other),
            "authorities": [a.to_dict() for a in other],
        },
        "display_list": [
            {
                "type": a.source_type,
                "title": a.title,
                "reference": a.reference,
                "relevance": f"{a.relevance_score:.0%}",
            }
            for a in authorities
        ],
    }


def suggest_authorities_for_topic(
    topic: str,
    question: str,
    retrieved_snippets: list[dict],
    missing_elements: list[str] = None,
) -> list[dict[str, str]]:
    """
    Suggest specific authorities to fill gaps in an answer.

    Args:
        topic: Legal topic area
        question: The question
        retrieved_snippets: KB results
        missing_elements: What parts of IRAC are missing

    Returns:
        List of suggestions with authority details and placement
    """
    suggestions = []
    authorities = pick_authorities(question, retrieved_snippets, topic, max_authorities=5)

    for auth in authorities:
        suggestion = {
            "authority": auth.formatted_citation,
            "doc_id": auth.doc_id,
            "type": auth.source_type,
            "placement": "rule",
            "reason": "Relevant authority for this topic",
        }

        if auth.source_type == "case":
            suggestion["reason"] = (
                f"Key case for {topic} - use in RULE section to establish legal principle"
            )
            suggestion["placement"] = "rule"

        elif auth.source_type == "legislation":
            suggestion["reason"] = f"Relevant legislation for {topic} - cite in RULE section"
            suggestion["placement"] = "rule"

        if missing_elements:
            if "application" in missing_elements:
                suggestion["placement"] = "application"
                suggestion["reason"] += ", apply to facts in APPLICATION section"

        suggestions.append(suggestion)

    return suggestions
