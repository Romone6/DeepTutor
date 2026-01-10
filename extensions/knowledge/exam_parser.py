"""
Exam Parser
===========

Parses HSC exam papers and marking guidelines into structured chunks
with metadata for knowledge base ingestion.

Features:
- Extracts questions with section, number, marks
- Identifies command terms (analyse, evaluate, etc.)
- Parses marking guidelines with response criteria
- Creates searchable chunks with full metadata
- Generates question index for practice generation

Usage:
    from extensions.knowledge.exam_parser import ExamParser

    parser = ExamParser()
    chunks = parser.parse_exam_paper(exam_path, subject, year)
    index = parser.create_question_index(chunks)
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.logging import get_logger

logger = get_logger("ExamParser")


@dataclass
class ExamQuestion:
    """A question from an exam paper."""

    question_id: str
    section: str
    question_number: str
    marks: int
    content: str
    command_terms: list[str]
    year: int
    subject: str
    source_type: str = "exam"


@dataclass
class MarkingGuideline:
    """A marking guideline entry."""

    question_id: str
    question_number: str
    marks_available: int
    criteria: list[dict]
    sample_answers: list[str]
    notes: list[str]
    year: int
    subject: str
    source_type: str = "marking"


class ExamParser:
    """Parser for HSC exam papers and marking guidelines."""

    COMMAND_TERMS = {
        "analyse",
        "apply",
        "assess",
        "calculate",
        "clarify",
        "compare",
        "contrast",
        "critique",
        "define",
        "describe",
        "discuss",
        "distinguish",
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
        "state",
        "summarise",
        "synthesise",
    }

    COMMAND_TERM_PATTERN = re.compile(r"\b(" + "|".join(COMMAND_TERMS) + r")\b", re.IGNORECASE)

    QUESTION_PATTERN = re.compile(r"(Question|Q\.?)\s*(\d+[\.\d]*)", re.IGNORECASE)

    SECTION_PATTERN = re.compile(r"^(Section|Part|Paper)\s*([A-Z0-9]+)", re.IGNORECASE)

    MARKS_PATTERN = re.compile(r"\((\d+)\s*marks?\)", re.IGNORECASE)

    def __init__(self):
        self.questions: list[ExamQuestion] = []
        self.marking_guidelines: list[MarkingGuideline] = []

    def parse_exam_paper(
        self,
        exam_path: Path,
        subject: str,
        year: int,
        course: str = "HSC",
    ) -> list[dict[str, Any]]:
        """Parse an exam paper and extract questions."""
        logger.info(f"Parsing exam paper: {exam_path.name}")

        with open(exam_path, encoding="utf-8") as f:
            content = f.read()

        chunks = []
        current_section = "Section A"
        question_num = 0

        lines = content.split("\n")
        in_question = False
        current_question_lines = []
        current_marks = 0

        for line in lines:
            line = line.strip()

            # Check for section headers
            section_match = self.SECTION_PATTERN.match(line)
            if section_match:
                current_section = f"{section_match.group(1)} {section_match.group(2)}"
                continue

            # Check for question markers
            question_match = self.QUESTION_PATTERN.search(line)
            if question_match:
                # Save previous question
                if current_question_lines:
                    question_num += 1
                    question_id = f"{subject}_{year}_Q{question_num:02d}"
                    content_text = "\n".join(current_question_lines)

                    # Extract command terms
                    command_terms = list(
                        set(t.capitalize() for t in self.COMMAND_TERM_PATTERN.findall(content_text))
                    )

                    chunks.append(
                        {
                            "doc_id": question_id,
                            "content": content_text,
                            "source_type": "exam",
                            "subject": subject,
                            "year": year,
                            "course": course,
                            "section": current_section,
                            "question_number": str(question_num),
                            "marks": current_marks,
                            "command_terms": command_terms,
                            "chunk_type": "question",
                        }
                    )

                # Start new question
                current_question_lines = [line]
                in_question = True
                current_marks = 0

                # Extract marks if present
                marks_match = self.MARKS_PATTERN.search(line)
                if marks_match:
                    current_marks = int(marks_match.group(1))

            elif in_question and line:
                # Continue current question
                # Check for marks in continuation
                marks_match = self.MARKS_PATTERN.search(line)
                if marks_match:
                    current_marks = int(marks_match.group(1))

                current_question_lines.append(line)

        # Save last question
        if current_question_lines:
            question_num += 1
            question_id = f"{subject}_{year}_Q{question_num:02d}"
            content_text = "\n".join(current_question_lines)

            command_terms = list(
                set(t.capitalize() for t in self.COMMAND_TERM_PATTERN.findall(content_text))
            )

            chunks.append(
                {
                    "doc_id": question_id,
                    "content": content_text,
                    "source_type": "exam",
                    "subject": subject,
                    "year": year,
                    "course": course,
                    "section": current_section,
                    "question_number": str(question_num),
                    "marks": current_marks,
                    "command_terms": command_terms,
                    "chunk_type": "question",
                }
            )

        logger.info(f"  Extracted {len(chunks)} questions")
        return chunks

    def parse_marking_guidelines(
        self,
        marking_path: Path,
        subject: str,
        year: int,
        course: str = "HSC",
    ) -> list[dict[str, Any]]:
        """Parse marking guidelines and extract criteria."""
        logger.info(f"Parsing marking guidelines: {marking_path.name}")

        with open(marking_path, encoding="utf-8") as f:
            content = f.read()

        chunks = []
        lines = content.split("\n")
        current_question = None
        current_criteria = []
        current_answers = []
        current_notes = []
        in_criteria = False
        in_sample = False

        for line in lines:
            line = line.strip()

            # Check for question markers
            question_match = self.QUESTION_PATTERN.search(line)
            if question_match:
                # Save previous question
                if current_question:
                    question_id = f"{current_question['subject']}_{current_question['year']}_Q{current_question['question_number']:02d}"
                    chunks.append(
                        {
                            "doc_id": f"{question_id}_marking",
                            "content": current_question["content"],
                            "source_type": "marking",
                            "subject": current_question["subject"],
                            "year": current_question["year"],
                            "course": course,
                            "question_number": str(current_question["question_number"]),
                            "marks_available": current_question["marks_available"],
                            "criteria": current_criteria,
                            "sample_answers": current_answers,
                            "notes": current_notes,
                            "chunk_type": "marking_guideline",
                        }
                    )

                # Start new marking entry
                current_question = {
                    "subject": subject,
                    "year": year,
                    "question_number": int(question_match.group(2)),
                    "content": line,
                    "marks_available": 0,
                }
                current_criteria = []
                current_answers = []
                current_notes = []
                in_criteria = True
                in_sample = False

                # Extract marks
                marks_match = self.MARKS_PATTERN.search(line)
                if marks_match:
                    current_question["marks_available"] = int(marks_match.group(1))

            elif current_question:
                # Look for criteria markers
                if "Criteria:" in line or "Marking Criteria:" in line.lower():
                    in_criteria = True
                    in_sample = False
                    continue

                if "Sample" in line and ("Answer" in line or "Response" in line):
                    in_criteria = False
                    in_sample = True
                    continue

                if in_criteria and line and not line.startswith("---"):
                    current_criteria.append(line)

                if in_sample and line and not line.startswith("---"):
                    current_answers.append(line)

                if not in_criteria and not in_sample and line:
                    current_notes.append(line)

        # Save last question
        if current_question:
            question_id = f"{current_question['subject']}_{current_question['year']}_Q{current_question['question_number']:02d}"
            chunks.append(
                {
                    "doc_id": f"{question_id}_marking",
                    "content": current_question["content"],
                    "source_type": "marking",
                    "subject": current_question["subject"],
                    "year": current_question["year"],
                    "course": course,
                    "question_number": str(current_question["question_number"]),
                    "marks_available": current_question["marks_available"],
                    "criteria": current_criteria,
                    "sample_answers": current_answers,
                    "notes": current_notes,
                    "chunk_type": "marking_guideline",
                }
            )

        logger.info(f"  Extracted {len(chunks)} marking entries")
        return chunks

    def generate_exam_document(
        self,
        subject: str,
        year: int,
        course: str = "HSC",
    ) -> str:
        """Generate a synthetic exam paper for demonstration."""
        modules = self._get_default_modules(subject)

        content_parts = [
            f"# {subject} {course} Examination",
            f"**Year:** {year}",
            f"**Course:** {course}",
            f"**Subject:** {subject}",
            "",
            "---",
            "",
            "## Examination Information",
            f"- **Subject:** {subject}",
            f"- **Year:** {year}",
            f"- **Time Allowed:** 3 hours",
            f"- **Total Marks:** 100",
            "",
            "---",
            "",
        ]

        for section_idx, module in enumerate(modules, 1):
            content_parts.append(f"## Section {chr(64 + section_idx)}: {module}")
            content_parts.append("")

            # Generate sample questions for each module
            questions = self._get_sample_questions(module, subject, section_idx)
            for q_idx, question in enumerate(questions, 1):
                content_parts.append(f"### Question {q_idx}")
                content_parts.append(question["text"])
                content_parts.append(f"**Marks:** {question['marks']}")
                content_parts.append("")

                if question.get("command_terms"):
                    content_parts.append(
                        f"**Command Terms:** {', '.join(question['command_terms'])}"
                    )
                    content_parts.append("")

        # Add marking guidelines
        content_parts.extend(self._generate_marking_section(subject, year, modules))

        return "\n".join(content_parts)

    def _get_default_modules(self, subject: str) -> list[str]:
        """Get default modules for a subject."""
        module_map = {
            "Mathematics Advanced": ["Functions", "Calculus", "Statistical Analysis"],
            "Biology": ["Cells as the Basis of Life", "Biological Unity", "Evolution"],
            "Business Studies": ["Nature of Business", "Business Management", "Operations"],
            "Legal Studies": ["Core: Crime", "Core: Human Rights"],
            "English Advanced": ["Common Module", "Module A", "Module B"],
        }
        return module_map.get(subject, ["Module 1", "Module 2"])

    def _get_sample_questions(self, module: str, subject: str, section_idx: int) -> list[dict]:
        """Generate sample questions for a module."""
        return [
            {
                "text": f"Using the concepts from {module}, explain the relationship between key principles and their applications.",
                "marks": 5,
                "command_terms": ["Explain", "Apply"],
            },
            {
                "text": f"Evaluate the significance of {module} in the context of broader subject understanding.",
                "marks": 7,
                "command_terms": ["Evaluate", "Analyse"],
            },
            {
                "text": f"Discuss how {module} relates to real-world scenarios and provide examples.",
                "marks": 8,
                "command_terms": ["Discuss", "Apply", "Analyse"],
            },
        ]

    def _generate_marking_section(self, subject: str, year: int, modules: list[str]) -> list[str]:
        """Generate marking guidelines section."""
        parts = [
            "## Marking Guidelines",
            "",
            "### Section A - Multiple Choice",
            "",
            "1. **Answer:** A",
            "   **Explanation:** The correct answer relates to the fundamental concept...",
            "",
            "### Section B - Short Answer",
            "",
            "#### Question 1",
            "**Marks Available:** 5",
            "",
            "**Criteria:**",
            "- Demonstrates understanding of key concepts (3 marks)",
            "- Applies concepts to solve problem (2 marks)",
            "",
            "**Sample Response:**",
            "- Student identifies key principles correctly...",
            "",
            "#### Question 2",
            "**Marks Available:** 7",
            "",
            "**Criteria:**",
            "- Analyses relationship between concepts (4 marks)",
            "- Provides appropriate examples (3 marks)",
            "",
            "### Section C - Extended Response",
            "",
            "#### Question 1",
            "**Marks Available:** 15",
            "",
            "**Criteria:**",
            "- Comprehensive understanding (5 marks)",
            "- Critical analysis (5 marks)",
            "- Effective communication (5 marks)",
            "",
            "**High-Quality Response:**",
            "- Demonstrates deep understanding of subject matter",
            "- Critically evaluates multiple perspectives",
            "- Presents arguments coherently and logically",
        ]

        return parts

    def create_question_index(
        self,
        exam_chunks: list[dict[str, Any]],
        subject: str,
    ) -> dict[str, Any]:
        """Create a question index for fast practice generation."""
        index = {
            "subject": subject,
            "created_at": datetime.now().isoformat(),
            "total_questions": 0,
            "by_year": {},
            "by_section": {},
            "by_marks": {},
            "by_command_term": {},
            "questions": [],
        }

        for chunk in exam_chunks:
            if chunk.get("chunk_type") != "question":
                continue

            year = chunk.get("year", 0)
            section = chunk.get("section", "Unknown")
            marks = chunk.get("marks", 0)
            command_terms = chunk.get("command_terms", [])
            q_num = chunk.get("question_number", "")

            # Track by year
            if year not in index["by_year"]:
                index["by_year"][year] = {"total": 0, "questions": []}
            index["by_year"][year]["total"] += 1
            index["by_year"][year]["questions"].append(chunk["doc_id"])

            # Track by section
            if section not in index["by_section"]:
                index["by_section"][section] = {"total": 0, "questions": []}
            index["by_section"][section]["total"] += 1
            index["by_section"][section]["questions"].append(chunk["doc_id"])

            # Track by marks
            marks_key = f"{marks}_marks"
            if marks_key not in index["by_marks"]:
                index["by_marks"][marks_key] = {"total": 0, "questions": []}
            index["by_marks"][marks_key]["total"] += 1
            index["by_marks"][marks_key]["questions"].append(chunk["doc_id"])

            # Track by command term
            for term in command_terms:
                term_key = term.lower()
                if term_key not in index["by_command_term"]:
                    index["by_command_term"][term_key] = {"total": 0, "questions": []}
                index["by_command_term"][term_key]["total"] += 1
                index["by_command_term"][term_key]["questions"].append(chunk["doc_id"])

            # Add to questions list
            index["questions"].append(
                {
                    "doc_id": chunk["doc_id"],
                    "year": year,
                    "section": section,
                    "question_number": q_num,
                    "marks": marks,
                    "command_terms": command_terms,
                    "preview": chunk["content"][:200] + "..."
                    if len(chunk["content"]) > 200
                    else chunk["content"],
                }
            )

        index["total_questions"] = len(index["questions"])

        return index

    def format_exam_chunk(self, chunk: dict[str, Any]) -> str:
        """Format an exam chunk as markdown."""
        lines = [
            f"# {chunk['subject']} {chunk['year']} - Question {chunk['question_number']}",
            "",
            f"**Subject:** {chunk['subject']}",
            f"**Year:** {chunk['year']}",
            f"**Section:** {chunk['section']}",
            f"**Marks:** {chunk['marks']}",
        ]

        if chunk.get("command_terms"):
            lines.append(f"**Command Terms:** {', '.join(chunk['command_terms'])}")

        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append(chunk["content"])

        return "\n".join(lines)

    def format_marking_chunk(self, chunk: dict[str, Any]) -> str:
        """Format a marking guideline chunk as markdown."""
        lines = [
            f"# Marking Guidelines: {chunk['subject']} {chunk['year']} Q{chunk['question_number']}",
            "",
            f"**Subject:** {chunk['subject']}",
            f"**Year:** {chunk['year']}",
            f"**Question:** {chunk['question_number']}",
            f"**Marks Available:** {chunk['marks_available']}",
            "",
            "---",
            "",
            "## Criteria",
        ]

        for criterion in chunk.get("criteria", []):
            lines.append(f"- {criterion}")

        if chunk.get("sample_answers"):
            lines.append("")
            lines.append("## Sample Answers")
            for answer in chunk["sample_answers"]:
                lines.append(f"> {answer}")

        if chunk.get("notes"):
            lines.append("")
            lines.append("## Notes")
            for note in chunk["notes"]:
                lines.append(f"- {note}")

        return "\n".join(lines)


def main():
    """Test the exam parser."""
    parser = ExamParser()

    # Test generating an exam document
    content = parser.generate_exam_document("Mathematics Advanced", 2024)
    print("Generated exam preview:")
    print(content[:1000])
    print("...")

    # Test question index creation
    test_chunks = [
        {
            "doc_id": "maths_adv_2024_Q001",
            "content": "Question 1: Using calculus, solve the following problem. (5 marks)",
            "chunk_type": "question",
            "subject": "Mathematics Advanced",
            "year": 2024,
            "section": "Section A",
            "question_number": 1,
            "marks": 5,
            "command_terms": ["Solve", "Apply"],
        },
        {
            "doc_id": "maths_adv_2024_Q002",
            "content": "Question 2: Evaluate the function and explain your reasoning. (7 marks)",
            "chunk_type": "question",
            "subject": "Mathematics Advanced",
            "year": 2024,
            "section": "Section A",
            "question_number": 2,
            "marks": 7,
            "command_terms": ["Evaluate", "Explain"],
        },
    ]

    index = parser.create_question_index(test_chunks, "Mathematics Advanced")
    print(f"\nQuestion index created:")
    print(f"  Total questions: {index['total_questions']}")
    print(f"  By year: {index['by_year']}")
    print(f"  By marks: {index['by_marks']}")
    print(f"  By command term: {list(index['by_command_term'].keys())}")


if __name__ == "__main__":
    main()
