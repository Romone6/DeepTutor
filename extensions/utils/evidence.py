"""
Evidence Map System
===================

Provides citation and evidence tracking for RAG outputs.

Features:
- EvidenceSnippet format with source metadata
- EvidenceMap container for response annotations
- Citation formatting utilities
- Source type categorization (syllabus, exam, marking_guide)

Usage:
    from extensions.utils.evidence import EvidenceSnippet, EvidenceMap, create_evidence_map

    evidence = create_evidence_map(snippets, query)
    response_with_citations = format_response_with_citations(answer, evidence)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SourceType(str, Enum):
    """Types of source documents for evidence."""

    SYLLABUS = "syllabus"
    EXAM_PAPER = "exam_paper"
    MARKING_GUIDE = "marking_guide"
    TEXTBOOK = "textbook"
    NOTES = "notes"
    UNKNOWN = "unknown"


@dataclass
class EvidenceSnippet:
    """
    A snippet with full citation metadata for evidence tracking.

    Fields:
        text: The actual content snippet
        doc_id: Unique document identifier
        page: Page number (if applicable)
        source_type: Type of source (syllabus/exam/marking_guide/etc)
        year: Publication year (for exams)
        score: Retrieval relevance score (0-1)
        topic: Topic label from metadata
        citation_key: Unique citation identifier (e.g., "[S1]", "[E2]")
    """

    text: str
    doc_id: str
    page: Optional[int] = None
    source_type: SourceType = SourceType.UNKNOWN
    year: Optional[int] = None
    score: float = 0.0
    topic: str = ""
    citation_key: str = ""

    def to_dict(self) -> dict:
        return {
            "text": self.text[:500] + "..." if len(self.text) > 500 else self.text,
            "doc_id": self.doc_id,
            "page": self.page,
            "source_type": self.source_type.value,
            "year": self.year,
            "score": self.score,
            "topic": self.topic,
            "citation_key": self.citation_key,
        }

    @classmethod
    def from_snippet(cls, snippet: "Snippet", index: int = 0) -> "EvidenceSnippet":
        """Create EvidenceSnippet from retriever Snippet."""
        metadata = snippet.metadata

        source_type = SourceType.UNKNOWN
        source_str = metadata.get("source_type", "").lower()
        if "syllabus" in source_str:
            source_type = SourceType.SYLLABUS
        elif "exam" in source_str:
            source_type = SourceType.EXAM_PAPER
        elif "marking" in source_str or "guide" in source_str:
            source_type = SourceType.MARKING_GUIDE
        elif "textbook" in source_str:
            source_type = SourceType.TEXTBOOK
        elif "notes" in source_str:
            source_type = SourceType.NOTES

        page = metadata.get("page")
        if page is not None:
            try:
                page = int(page)
            except (ValueError, TypeError):
                page = None

        year = metadata.get("year")
        if year is not None:
            try:
                year = int(year)
            except (ValueError, TypeError):
                year = None

        topic = metadata.get("topic", "")

        doc_id = metadata.get("doc_id", f"doc_{index}")

        return cls(
            text=snippet.text,
            doc_id=doc_id,
            page=page,
            source_type=source_type,
            year=year,
            score=snippet.score,
            topic=topic,
            citation_key=_generate_citation_key(source_type, index),
        )


@dataclass
class EvidenceMap:
    """
    Container for all evidence used in a response.

    Provides structured access to sources and formatted citations.
    """

    query: str
    snippets: list[EvidenceSnippet] = field(default_factory=list)
    kb_name: str = ""
    total_retrieved: int = 0
    used_for_answer: int = 0
    kb_empty: bool = False
    generation_timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "kb_name": self.kb_name,
            "total_retrieved": self.total_retrieved,
            "used_for_answer": self.used_for_answer,
            "kb_empty": self.kb_empty,
            "timestamp": self.generation_timestamp,
            "snippets": [s.to_dict() for s in self.snippets],
        }

    def get_citations(self) -> list[str]:
        """Get list of citation keys for used snippets."""
        return [s.citation_key for s in self.snippets]

    def get_sources_by_type(self, source_type: SourceType) -> list[EvidenceSnippet]:
        """Get all snippets of a specific source type."""
        return [s for s in self.snippets if s.source_type == source_type]

    def format_citation_list(self) -> str:
        """Format citations as a reference list."""
        if not self.snippets:
            return "No sources available."

        parts = ["## References"]
        parts.append("")

        for snippet in self.snippets:
            citation = self._format_single_citation(snippet)
            parts.append(f"{snippet.citation_key} {citation}")

        return "\n".join(parts)

    def _format_single_citation(self, snippet: EvidenceSnippet) -> str:
        """Format a single citation."""
        source_labels = {
            SourceType.SYLLABUS: "Syllabus",
            SourceType.EXAM_PAPER: f"Exam Paper {snippet.year}" if snippet.year else "Exam Paper",
            SourceType.MARKING_GUIDE: "Marking Guide",
            SourceType.TEXTBOOK: "Textbook",
            SourceType.NOTES: "Study Notes",
            SourceType.UNKNOWN: "Source",
        }

        label = source_labels.get(snippet.source_type, "Source")

        parts = [label]
        if snippet.doc_id and snippet.doc_id != "unknown":
            parts.append(f"({snippet.doc_id})")
        if snippet.page:
            parts.append(f"p.{snippet.page}")
        if snippet.topic:
            parts.append(f"- {snippet.topic}")

        return " ".join(parts)


def _generate_citation_key(source_type: SourceType, index: int) -> str:
    """Generate a unique citation key like [S1], [E2], [M1]."""
    prefixes = {
        SourceType.SYLLABUS: "S",
        SourceType.EXAM_PAPER: "E",
        SourceType.MARKING_GUIDE: "M",
        SourceType.TEXTBOOK: "T",
        SourceType.NOTES: "N",
        SourceType.UNKNOWN: "X",
    }
    prefix = prefixes.get(source_type, "X")
    return f"[{prefix}{index + 1}]"


def create_evidence_map(
    snippets: list["Snippet"],
    query: str,
    kb_name: str = "",
    max_snippets: int = 6,
) -> EvidenceMap:
    """
    Create an EvidenceMap from retrieved snippets.

    Args:
        snippets: List of retrieved Snippets
        query: The search query
        kb_name: Knowledge base name
        max_snippets: Maximum snippets to include in evidence map

    Returns:
        EvidenceMap with formatted citations
    """
    import datetime

    evidence_snippets = [
        EvidenceSnippet.from_snippet(s, i) for i, s in enumerate(snippets[:max_snippets])
    ]

    return EvidenceMap(
        query=query,
        snippets=evidence_snippets,
        kb_name=kb_name,
        total_retrieved=len(snippets),
        used_for_answer=len(evidence_snippets),
        kb_empty=len(snippets) == 0,
        generation_timestamp=datetime.datetime.now().isoformat(),
    )


def format_response_with_citations(
    response: str,
    evidence_map: EvidenceMap,
    citation_style: str = "inline",
) -> tuple[str, EvidenceMap]:
    """
    Format a response with inline citations.

    Args:
        response: The LLM-generated response
        evidence_map: EvidenceMap with sources
        citation_style: "inline" or "bracketed"

    Returns:
        Tuple of (formatted_response, evidence_map)
    """
    if not evidence_map.snippets:
        return response, evidence_map

    citations = evidence_map.get_citations()

    if citation_style == "inline":
        citation_str = " ".join(citations)
        if response.rstrip().endswith("."):
            response = response.rstrip(".") + f" {citation_str}."
        else:
            response = response + f" {citation_str}"

    return response, evidence_map


def get_citation_text_for_ui(evidence_map: EvidenceMap) -> list[dict]:
    """
    Get formatted citations for UI display.

    Returns list of dicts with:
        - key: citation_key
        - label: formatted source label
        - preview: short text preview
        - full_text: complete snippet text
    """
    if not evidence_map.snippets:
        return []

    citations = []
    for snippet in evidence_map.snippets:
        source_labels = {
            SourceType.SYLLABUS: "Syllabus",
            SourceType.EXAM_PAPER: f"Exam {snippet.year}" if snippet.year else "Exam Paper",
            SourceType.MARKING_GUIDE: "Marking Guide",
            SourceType.TEXTBOOK: "Textbook",
            SourceType.NOTES: "Notes",
            SourceType.UNKNOWN: "Source",
        }

        preview = snippet.text[:150] + "..." if len(snippet.text) > 150 else snippet.text

        citations.append(
            {
                "key": snippet.citation_key,
                "label": source_labels.get(snippet.source_type, "Source"),
                "doc_id": snippet.doc_id,
                "page": snippet.page,
                "topic": snippet.topic,
                "preview": preview,
                "full_text": snippet.text,
                "score": snippet.score,
                "source_type": snippet.source_type.value,
            }
        )

    return citations


class CitationStyle(str, Enum):
    """Citation style options."""

    INLINE = "inline"
    BRACKETED = "bracketed"
    FOOTNOTE = "footnote"
    APA = "apa"
    MLA = "mla"


def format_citation_reference(
    snippet: EvidenceSnippet,
    style: CitationStyle = CitationStyle.BRACKETED,
) -> str:
    """
    Format a single citation reference.

    Args:
        snippet: EvidenceSnippet to format
        style: Citation style to use

    Returns:
        Formatted citation string
    """
    if style == CitationStyle.INLINE:
        return f"[{snippet.citation_key}]"

    elif style == CitationStyle.BRACKETED:
        return f"({snippet.citation_key})"

    elif style == CitationStyle.FOOTNOTE:
        return f"^{snippet.citation_key.strip('[]')}"

    elif style == CitationStyle.APA:
        source = snippet.source_type.value.capitalize()
        year = f" ({snippet.year})" if snippet.year else ""
        return f"{source}{year}"

    elif style == CitationStyle.MLA:
        source = snippet.source_type.value.capitalize()
        page = f", p. {snippet.page}" if snippet.page else ""
        return f"{source}{page}"

    return f"[{snippet.citation_key}]"


__all__ = [
    "EvidenceSnippet",
    "EvidenceMap",
    "SourceType",
    "CitationStyle",
    "create_evidence_map",
    "format_response_with_citations",
    "get_citation_text_for_ui",
    "format_citation_reference",
]
