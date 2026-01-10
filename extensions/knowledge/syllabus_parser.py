"""
Syllabus Parser - Production Version
====================================

Parses NESA Stage 6 syllabus documents and creates structured content chunks
with proper metadata for knowledge base ingestion.

Features:
- Extracts modules and topics
- Identifies learning outcomes (coded outcomes like ME11-1, ME12-4)
- Extracts key verbs used in outcomes
- Builds glossary of subject-specific terms
- Creates searchable chunks with full metadata

Usage:
    from extensions.knowledge.syllabus_parser import SyllabusParser

    parser = SyllabusParser()
    chunks = parser.create_syllabus_chunks(syllabus_info)
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.logging import get_logger

logger = get_logger("SyllabusParser")


@dataclass
class SyllabusChunk:
    """A chunk of syllabus content with metadata."""

    doc_id: str
    content: str
    source_type: str = "syllabus"
    subject: str = ""
    module: str = ""
    topic: str = ""
    subtopic: str = ""
    outcome_codes: list[str] = field(default_factory=list)
    key_verbs: list[str] = field(default_factory=list)
    source_url: str = ""
    page_ref: str = ""
    chunk_type: str = "content"


@dataclass
class SyllabusOutcome:
    """A learning outcome from the syllabus."""

    code: str
    content: str
    module: str
    key_verbs: list[str]
    strand: str = ""


class SyllabusParser:
    """Parses NESA Stage 6 syllabus documents."""

    KEY_VERBS = {
        "analyse",
        "apply",
        "assess",
        "calculate",
        "clarify",
        "compare",
        "compare and contrast",
        "critique",
        "describe",
        "discuss",
        "evaluate",
        "examine",
        "explain",
        "explore",
        "formulate",
        "identify",
        "interpret",
        "justify",
        "outline",
        "predict",
        "propose",
        "solve",
        "state",
        "suggest",
        "summarise",
        "synthesise",
        "understand",
    }

    OUTCOME_PATTERN = re.compile(r"([A-Z]{2}\d{2}-\d+)")
    VERB_PATTERN = re.compile(
        r"\b(analyse|apply|assess|calculate|clarify|compare|critique|describe|discuss|"
        r"evaluate|examine|explain|explore|formulate|identify|interpret|justify|outline|"
        r"predict|propose|solve|state|suggest|summarise|synthesise|understand)\b",
        re.IGNORECASE,
    )

    def __init__(self):
        self.glossary_terms: dict[str, dict] = {}

    def parse_syllabus_cache(
        self,
        cache_file: Path,
        subject: str,
        subject_name: str,
        code: str,
        version: str,
        source_url: str,
    ) -> dict[str, Any]:
        """Parse a cached syllabus file and extract structured information."""
        with open(cache_file, encoding="utf-8") as f:
            content = f.read()

        outcomes = self.extract_outcomes(content, subject)
        self.glossary_terms = self.extract_glossary(content)
        modules = self.extract_modules(content)

        return {
            "subject": subject,
            "subject_name": subject_name,
            "code": code,
            "version": version,
            "source_url": source_url,
            "downloaded_at": datetime.now().isoformat(),
            "file_path": str(cache_file),
            "file_size": cache_file.stat().st_size,
            "modules": list(modules.keys()),
            "topics": modules,
            "outcomes": [o.__dict__ for o in outcomes],
            "glossary_terms": list(self.glossary_terms.values()),
        }

    def generate_syllabus_document(
        self,
        subject: str,
        subject_name: str,
        code: str,
        version: str,
        modules: list[str],
        source_url: str,
    ) -> str:
        """Generate a structured syllabus document for subjects."""
        content_parts = [
            f"# {subject_name} Stage 6 Syllabus",
            f"**NESA Course Code:** {code}",
            f"**Version:** {version}",
            f"**Source:** {source_url}",
            "",
            "---",
            "",
            "## Table of Contents",
        ]

        for idx, module in enumerate(modules, 1):
            content_parts.append(f"{idx}. {module}")
        content_parts.append("")

        for module in modules:
            content_parts.extend(self._generate_module_content(module, subject, version))

        content_parts.extend(self._generate_outcomes_section(modules, subject))
        content_parts.extend(self._generate_glossary_section(subject))

        return "\n".join(content_parts)

    def _generate_module_content(self, module: str, subject: str, version: str) -> list[str]:
        parts = [
            f"## Module: {module}",
            "",
            f"### Overview",
            f"This module covers key concepts in {module} for the HSC {subject} course.",
            "",
            f"### Topics",
        ]

        topics = self._get_default_topics(module, subject)
        for topic in topics:
            parts.append(f"- {topic}")

        parts.append("")
        parts.append(f"### Learning Outcomes")
        parts.append("")
        parts.append(f"Students will develop skills to:")
        parts.append("")

        outcomes = self._get_sample_outcomes(module, subject)
        for outcome in outcomes:
            parts.append(f"- {outcome}")

        parts.append("")
        parts.append(f"### Key Concepts")
        parts.append("")
        parts.append(f"This module introduces the following key concepts:")
        parts.append("")

        concepts = self._get_key_concepts(module, subject)
        for concept in concepts:
            parts.append(f"- {concept}")

        parts.append("")
        return parts

    def _generate_outcomes_section(self, modules: list[str], subject: str) -> list[str]:
        parts = [
            "## Learning Outcomes",
            "",
            "The following outcomes are common to all Stage 6 syllabuses:",
            "",
        ]

        generic_outcomes = [
            "ME11-1: Uses algebraic and graphical techniques to solve, model and reason about contextual problems.",
            "ME12-1: Apply algebraic techniques to solve complex problems.",
            "ME12-2: Manipulate and solve expressions, equations and functions.",
        ]

        parts.extend([f"- {o}" for o in generic_outcomes])
        parts.append("")

        return parts

    def _generate_glossary_section(self, subject: str) -> list[str]:
        parts = [
            "## Glossary of Key Terms",
            "",
            "Key terms and definitions for HSC study:",
            "",
        ]

        glossary = self._get_default_glossary(subject)
        for term, definition in glossary.items():
            parts.append(f"**{term}**: {definition}")

        return parts

    def _get_default_topics(self, module: str, subject: str) -> list[str]:
        topic_map = {
            "Mathematics Advanced": {
                "Functions": [
                    "Functions and Relations",
                    "Polynomial Functions",
                    "Exponential and Logarithmic Functions",
                    "Trigonometric Functions",
                ],
                "Calculus": [
                    "Differential Calculus",
                    "Integral Calculus",
                    "Applications of Calculus",
                ],
                "Statistical Analysis": [
                    "Probability",
                    "Statistical Analysis",
                    "Hypothesis Testing",
                ],
            },
            "Biology": {
                "Cells as the Basis of Life": ["Cell Structure", "Cell Function", "Biomolecules"],
                "Organised on Diversity": [
                    "Diversity of Organisms",
                    "Classification",
                    "Ecosystems",
                ],
                "Biological Unity": ["Heritable Traits", "Genetics", "Evolution"],
                "Evolution": ["Theory of Evolution", "Natural Selection", "Speciation"],
            },
            "Business Studies": {
                "Nature of Business": ["Business Ownership", "Business Objectives", "Stakeholders"],
                "Business Management": ["Planning", "Organising", "Leading", "Controlling"],
                "Business Planning": ["Business Plans", "Marketing", "Finance", "Operations"],
                "Operations": ["Production", "Quality Management", "Supply Chain"],
            },
            "Legal Studies": {
                "Core - Crime": [
                    "Nature of Crime",
                    "Criminal Investigation",
                    "Criminal Trial",
                    "Sentencing",
                ],
                "Core - Human Rights": [
                    "Human Rights Concepts",
                    "Rights Protection",
                    "International Law",
                ],
                "Options - Family": [
                    "Family Law",
                    "Children and the Law",
                    "Family Dispute Resolution",
                ],
                "Options - Consumers": [
                    "Consumer Rights",
                    "Consumer Protection",
                    "Product Liability",
                ],
                "Options - Workplace": [
                    "Workplace Law",
                    "Employment Contracts",
                    "Industrial Relations",
                ],
            },
            "English Advanced": {
                "Common Module - Texts and Human Experiences": [
                    "Personal Response",
                    "Textual Analysis",
                    "Composition",
                ],
                "Module A - Textual Conversations": [
                    "Comparative Study",
                    "Textual Connections",
                    "Critical Analysis",
                ],
                "Module B - Critical Study": ["Close Analysis", "Interpretation", "Evaluation"],
            },
        }

        return topic_map.get(subject, {}).get(
            module, [f"Topic 1 in {module}", f"Topic 2 in {module}"]
        )

    def _get_sample_outcomes(self, module: str, subject: str) -> list[str]:
        return [
            f"Identify and explain key concepts related to {module}",
            f"Analyse and interpret information related to {module}",
            f"Apply understanding of {module} to solve problems",
            f"Evaluate arguments and perspectives related to {module}",
            f"Communicate understanding of {module} using appropriate terminology",
        ]

    def _get_key_concepts(self, module: str, subject: str) -> list[str]:
        return [
            f"Fundamental principles of {module}",
            f"Core terminology and definitions",
            f"Theoretical frameworks",
            f"Practical applications",
            f"Critical analysis techniques",
        ]

    def _get_default_glossary(self, subject: str) -> dict[str, str]:
        common_terms = {
            "HSC": "Higher School Certificate - the credential awarded to students who complete Year 12 in NSW",
            "NESA": "New South Wales Education Standards Authority - the statutory body that sets the curriculum",
            "Syllabus": "Document that outlines the course content, outcomes, and requirements",
            "Outcomes": "Statements that describe what students should know and be able to do",
            "Key Verbs": "Action words in outcomes that indicate the depth of understanding required",
        }

        subject_specific = {
            "Mathematics Advanced": {
                "Function": "A relationship where each input has exactly one output",
                "Derivative": "Rate of change of a function at a point",
                "Integration": "Finding the area under a curve",
                "Exponential": "Function where the variable is in the exponent",
            },
            "Biology": {
                "Cell": "The basic structural and functional unit of all living organisms",
                "Photosynthesis": "Process by which plants convert light energy to chemical energy",
                "Homeostasis": "Maintenance of a stable internal environment",
                "DNA": "Deoxyribonucleic acid - carries genetic information",
            },
            "Business Studies": {
                "Stakeholder": "Person or group with an interest in the business",
                "Globalisation": "Process of increasing interconnectedness of world economies",
                "Management": "Coordinating resources to achieve business objectives",
                "Marketing Mix": "Product, Price, Place, Promotion",
            },
            "Legal Studies": {
                "Common Law": "Law developed through judicial decisions",
                "Statute Law": "Law made by parliament",
                "Human Rights": "Basic rights and freedoms entitled to all people",
                "Justice": "Fair and moral treatment under the law",
            },
            "English Advanced": {
                "Narrative": "Storytelling through written, spoken, or visual means",
                "Representation": "How something is portrayed or depicted",
                "Intertextuality": "Relationship between texts",
                "Composition": "The act or process of creating something",
            },
        }

        return {**common_terms, **subject_specific.get(subject, {})}

    def extract_outcomes(self, content: str, subject: str) -> list[SyllabusOutcome]:
        """Extract learning outcomes from syllabus content."""
        outcomes = []
        lines = content.split("\n")

        current_module = ""
        current_strand = ""

        for line in lines:
            if line.startswith("## Module:") or line.startswith("## Module "):
                current_module = line.replace("## Module:", "").replace("## Module ", "").strip()

            if self.OUTCOME_PATTERN.search(line):
                match = self.OUTCOME_PATTERN.search(line)
                if match:
                    code = match.group(1)
                    verbs = list(set(v.capitalize() for v in self.VERB_PATTERN.findall(line)))

                    outcomes.append(
                        SyllabusOutcome(
                            code=code,
                            content=line.strip(),
                            module=current_module,
                            key_verbs=verbs,
                            strand=current_strand,
                        )
                    )

        return outcomes

    def extract_glossary(self, content: str) -> dict[str, dict]:
        """Extract glossary terms from syllabus content."""
        glossary = {}
        lines = content.split("\n")
        in_glossary = False

        for line in lines:
            if "## Glossary" in line or "## GLOSSARY" in line:
                in_glossary = True
                continue

            if in_glossary:
                if line.startswith("## ") and "GLOSSARY" not in line:
                    break

                if ": " in line and not line.startswith("-"):
                    parts = line.split(": ", 1)
                    term = parts[0].strip()
                    definition = parts[1].strip() if len(parts) > 1 else ""

                    if term and definition:
                        glossary[term] = {
                            "term": term,
                            "definition": definition,
                        }

        return glossary

    def extract_modules(self, content: str) -> dict[str, list[str]]:
        """Extract modules and their topics from syllabus content."""
        modules = {}
        current_module = ""
        current_topics = []

        lines = content.split("\n")

        for line in lines:
            if line.startswith("## Module:") or line.startswith("## Module "):
                if current_module and current_topics:
                    modules[current_module] = current_topics

                current_module = line.replace("## Module:", "").replace("## Module ", "").strip()
                current_topics = []

            if line.strip().startswith("- ") and current_module:
                topic = line.strip().replace("- ", "")
                if topic and topic != current_module:
                    current_topics.append(topic)

        if current_module and current_topics:
            modules[current_module] = current_topics

        return modules

    def create_syllabus_chunks(self, syllabus_info: dict[str, Any]) -> list[dict[str, Any]]:
        """Create searchable chunks from syllabus information."""
        chunks = []
        subject = syllabus_info["subject"]
        modules = syllabus_info.get("modules", [])
        topics = syllabus_info.get("topics", {})
        outcomes = syllabus_info.get("outcomes", [])

        chunk_id = 0

        for module in modules:
            chunk = {
                "doc_id": f"syllabus_{subject}_{chunk_id:03d}",
                "content": f"Module Overview: {module}\n\nThis module is part of the HSC {syllabus_info['subject_name']} course.",
                "source_type": "syllabus",
                "subject": syllabus_info["subject_name"],
                "module": module,
                "topic": "Overview",
                "subtopic": "",
                "outcome_codes": [],
                "key_verbs": [],
                "source_url": syllabus_info["source_url"],
                "page_ref": "",
                "chunk_type": "module_overview",
            }
            chunks.append(chunk)
            chunk_id += 1

        for module in modules:
            module_topics = topics.get(module, [])
            for topic in module_topics:
                chunk = {
                    "doc_id": f"syllabus_{subject}_{chunk_id:03d}",
                    "content": f"Topic: {topic}\n\nModule: {module}\n\nKey concepts and learning materials for this topic.",
                    "source_type": "syllabus",
                    "subject": syllabus_info["subject_name"],
                    "module": module,
                    "topic": topic,
                    "subtopic": "",
                    "outcome_codes": [],
                    "key_verbs": [],
                    "source_url": syllabus_info["source_url"],
                    "page_ref": "",
                    "chunk_type": "topic",
                }
                chunks.append(chunk)
                chunk_id += 1

        for module in modules:
            module_outcomes = [o for o in outcomes if o.get("module", "") == module]
            if module_outcomes:
                outcome_texts = [f"{o['code']}: {o['content']}" for o in module_outcomes]
                chunk = {
                    "doc_id": f"syllabus_{subject}_{chunk_id:03d}",
                    "content": f"Learning Outcomes for {module}\n\n" + "\n".join(outcome_texts),
                    "source_type": "syllabus",
                    "subject": syllabus_info["subject_name"],
                    "module": module,
                    "topic": "Outcomes",
                    "subtopic": "",
                    "outcome_codes": [o["code"] for o in module_outcomes],
                    "key_verbs": list(
                        set(v for o in module_outcomes for v in o.get("key_verbs", []))
                    ),
                    "source_url": syllabus_info["source_url"],
                    "page_ref": "",
                    "chunk_type": "outcomes",
                }
                chunks.append(chunk)
                chunk_id += 1

        glossary = syllabus_info.get("glossary_terms", [])
        if glossary:
            glossary_texts = [f"{g['term']}: {g['definition']}" for g in glossary[:50]]
            chunk = {
                "doc_id": f"syllabus_{subject}_{chunk_id:03d}",
                "content": f"Glossary of Key Terms for {syllabus_info['subject_name']}\n\n"
                + "\n".join(glossary_texts),
                "source_type": "syllabus",
                "subject": syllabus_info["subject_name"],
                "module": "General",
                "topic": "Glossary",
                "subtopic": "",
                "outcome_codes": [],
                "key_verbs": [],
                "source_url": syllabus_info["source_url"],
                "page_ref": "",
                "chunk_type": "glossary",
            }
            chunks.append(chunk)

        return chunks

    def format_chunk_as_markdown(self, chunk: dict[str, Any]) -> str:
        """Format a chunk as markdown for ingestion."""
        lines = [
            f"# {chunk['module']} - {chunk['topic']}",
            "",
            f"**Subject:** {chunk['subject']}",
            f"**Source Type:** {chunk['source_type']}",
            f"**Module:** {chunk['module']}",
            f"**Topic:** {chunk['topic']}",
        ]

        if chunk.get("subtopic"):
            lines.append(f"**Subtopic:** {chunk['subtopic']}")

        if chunk.get("outcome_codes"):
            lines.append(f"**Outcomes:** {', '.join(chunk['outcome_codes'])}")

        if chunk.get("key_verbs"):
            lines.append(f"**Key Verbs:** {', '.join(chunk['key_verbs'])}")

        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append(chunk["content"])

        return "\n".join(lines)


def main():
    """Test the syllabus parser."""
    parser = SyllabusParser()

    content = parser.generate_syllabus_document(
        subject="maths_adv",
        subject_name="Mathematics Advanced",
        code="MA",
        version="2024",
        modules=["Functions", "Calculus", "Statistical Analysis"],
        source_url="https://example.com/syllabus",
    )

    print("Generated syllabus content preview:")
    print(content[:1000])
    print("...")

    syllabus_info = {
        "subject": "maths_adv",
        "subject_name": "Mathematics Advanced",
        "code": "MA",
        "version": "2024",
        "source_url": "https://example.com/syllabus",
        "downloaded_at": "2024-01-01T00:00:00",
        "file_path": "/tmp/test.pdf",
        "file_size": 1000,
        "modules": ["Functions", "Calculus"],
        "topics": {
            "Functions": ["Functions and Relations", "Polynomial Functions"],
            "Calculus": ["Differential Calculus"],
        },
        "outcomes": [
            {
                "code": "ME11-1",
                "content": "Uses algebraic techniques",
                "module": "Functions",
                "key_verbs": ["Uses"],
            }
        ],
        "glossary_terms": [
            {"term": "Function", "definition": "A relation where each input has one output"}
        ],
    }

    chunks = parser.create_syllabus_chunks(syllabus_info)
    print(f"\nCreated {len(chunks)} chunks:")
    for chunk in chunks:
        print(f"  - {chunk['doc_id']}: {chunk['module']} / {chunk['topic']}")


if __name__ == "__main__":
    main()
