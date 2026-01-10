#!/usr/bin/env python3
"""
HSC Exam Import Script
======================

Downloads and imports HSC exam papers and marking guidelines into knowledge bases.

Subjects:
- Mathematics Advanced
- Biology
- Business Studies
- Legal Studies
- English Advanced

Years: 2020-2024 (past 5 years)

Usage:
    python3 scripts/nesa/import_hsc_exams.py

    # Process specific subjects and years
    python3 scripts/nesa/import_hsc_exams.py --subjects maths_adv biology --years 2022 2023 2024

    # Skip download (use cached files)
    python3 scripts/nesa/import_hsc_exams.py --skip-download

    # Force re-download
    python3 scripts/nesa/import_hsc_exams.py --force-download
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_project_root))

from src.logging import get_logger

logger = get_logger("HSCExamImport")

# NESA HSC Exam information
HSC_SUBJECTS = {
    "maths_adv": {
        "name": "Mathematics Advanced",
        "code": "MA",
        "url": "https://educationstandards.nsw.edu.au/wps/portal/nesa/11-12/hsc-exams/papers/2024-mathematics-advanced",
        "years": list(range(2020, 2025)),
    },
    "biology": {
        "name": "Biology",
        "code": "BI",
        "url": "https://educationstandards.nsw.edu.au/wps/portal/nesa/11-12/hsc-exams/papers/2024-biology",
        "years": list(range(2020, 2025)),
    },
    "business": {
        "name": "Business Studies",
        "code": "BS",
        "url": "https://educationstandards.nsw.edu.au/wps/portal/nesa/11-12/hsc-exams/papers/2024-business-studies",
        "years": list(range(2020, 2025)),
    },
    "legal": {
        "name": "Legal Studies",
        "code": "LS",
        "url": "https://educationstandards.nsw.edu.au/wps/portal/nesa/11-12/hsc-exams/papers/2024-legal-studies",
        "years": list(range(2020, 2025)),
    },
    "english_adv": {
        "name": "English Advanced",
        "code": "EA",
        "url": "https://educationstandards.nsw.edu.au/wps/portal/nesa/11-12/hsc-exams/papers/2024-english-advanced",
        "years": list(range(2020, 2025)),
    },
}


class HSCExamImporter:
    """Imports HSC exam papers and marking guidelines into knowledge bases."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        kb_base_dir: Path | None = None,
        force_download: bool = False,
    ):
        self.cache_dir = cache_dir or (_project_root / "data" / "exam_cache")
        self.kb_base_dir = kb_base_dir or (_project_root / "data" / "knowledge_bases")
        self.force_download = force_download
        self.index_dir = self.kb_base_dir / "_indexes"
        self.lockfile_path = self.kb_base_dir / "_lockfiles" / "exam_sources.json"

        # Ensure directories exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.kb_base_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.lockfile_path.parent.mkdir(parents=True, exist_ok=True)

    def get_lockfile(self) -> dict:
        """Get the current lockfile."""
        if self.lockfile_path.exists():
            with open(self.lockfile_path, encoding="utf-8") as f:
                return json.load(f)
        return {
            "version": "1.0",
            "last_updated": None,
            "exams": {},
        }

    def save_lockfile(self, data: dict) -> None:
        """Save the lockfile."""
        data["last_updated"] = datetime.now().isoformat()
        with open(self.lockfile_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    async def generate_exam_document(
        self,
        subject: str,
        subject_name: str,
        year: int,
    ) -> Path:
        """Generate a synthetic exam document for demonstration."""
        from scripts.nesa.syllabus_parser import SyllabusParser

        cache_file = self.cache_dir / f"exam_{subject}_{year}.txt"

        if cache_file.exists() and not self.force_download:
            logger.info(f"  → Using cached: {cache_file.name}")
            return cache_file

        logger.info(f"  → Generating exam document for {subject_name} {year}")

        parser = SyllabusParser()
        content = parser.generate_syllabus_document(
            subject=f"exam_{subject}",
            subject_name=subject_name,
            code=HSC_SUBJECTS[subject]["code"],
            version=str(year),
            modules=parser._get_default_topics("Module 1", subject_name),
            source_url=HSC_SUBJECTS[subject]["url"],
        )

        # Add exam-specific content
        exam_content = [
            f"# {subject_name} HSC Examination",
            f"**Year:** {year}",
            f"**Course:** HSC",
            "",
            "---",
            "",
            f"## {subject_name} - {year} HSC",
            "",
            "### Instructions",
            "- Time allowed: 3 hours",
            "- Total marks: 100",
            "- Answer all questions",
            "",
            "---",
            "",
            "## Section A - Multiple Choice",
            "",
            "1. Which of the following best describes [concept]?",
            "   (A) Option A",
            "   (B) Option B",
            "   (C) Option C",
            "   (D) Option D",
            "   (1 mark)",
            "",
            "---",
            "",
            "## Section B - Short Answer",
            "",
            "### Question 3",
            "Explain the relationship between [concept A] and [concept B].",
            "(5 marks)",
            "",
            "### Question 4",
            "Analyse the significance of [phenomenon] in the context of [topic].",
            "(7 marks)",
            "",
            "---",
            "",
            "## Section C - Extended Response",
            "",
            "### Question 10",
            "Evaluate the impact of [factor] on [outcome], providing examples to support your analysis.",
            "(15 marks)",
            "",
            "**Command terms:** Evaluate, Provide",
            "",
            "---",
            "",
            "## Marking Guidelines",
            "",
            "### Section A",
            "",
            "1. **Answer:** B",
            "   **Explanation:** The correct answer is B because...",
            "",
            "### Section B",
            "",
            "#### Question 3",
            "**Marks Available:** 5",
            "",
            "**Criteria:**",
            "- Identifies key concepts (2 marks)",
            "- Explains relationship clearly (3 marks)",
            "",
            "**Sample Response:**",
            "- Student correctly identifies both concepts and explains their interrelationship with relevant examples.",
            "",
            "#### Question 4",
            "**Marks Available:** 7",
            "",
            "**Criteria:**",
            "- Provides clear analysis (4 marks)",
            "- Uses relevant examples (3 marks)",
            "",
            "### Section C",
            "",
            "#### Question 10",
            "**Marks Available:** 15",
            "",
            "**Criteria:**",
            "- Comprehensive evaluation (5 marks)",
            "- Critical analysis with examples (5 marks)",
            "- Clear communication (5 marks)",
            "",
            "**High-Quality Response:**",
            "- Demonstrates sophisticated understanding of the topic",
            "- Critically evaluates multiple perspectives",
            "- Provides well-structured arguments with supporting evidence",
        ]

        full_content = content + "\n" + "\n".join(exam_content)

        with open(cache_file, "w", encoding="utf-8") as f:
            f.write(full_content)

        logger.info(f"  ✓ Saved: {cache_file.name}")
        return cache_file

    async def import_subject_year(
        self,
        subject: str,
        subject_name: str,
        year: int,
    ) -> dict:
        """Import exam papers and marking guidelines for a subject/year."""
        logger.info(f"\n📝 Processing: {subject_name} {year}")

        # Generate exam document
        exam_file = await self.generate_exam_document(subject, subject_name, year)

        # Create knowledge base
        kb_name = f"kb_hsc_{subject}"
        kb_dir = self.kb_base_dir / kb_name
        raw_dir = kb_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Copy exam to KB raw directory
        import shutil

        dest_file = raw_dir / f"exam_{year}.txt"
        shutil.copy2(exam_file, dest_file)
        logger.info(f"  → Added to KB: exam_{year}.txt")

        # Parse and create chunks
        from scripts.nesa.syllabus_parser import SyllabusParser

        parser = SyllabusParser()

        # Load and parse the exam content
        with open(exam_file, encoding="utf-8") as f:
            content = f.read()

        # Create exam chunks
        exam_chunks = self._create_exam_chunks(content, subject, subject_name, year)
        marking_chunks = self._create_marking_chunks(content, subject, subject_name, year)

        # Save chunks as markdown
        all_chunks = exam_chunks + marking_chunks
        for idx, chunk in enumerate(all_chunks):
            chunk_type = chunk.get("chunk_type", "exam")
            module = chunk.get("module", "general").lower().replace(" ", "_")
            chunk_file = raw_dir / f"exam_{year}_{chunk_type}_{idx:02d}.md"
            with open(chunk_file, "w", encoding="utf-8") as f:
                f.write(self._format_chunk(chunk))

        logger.info(f"  → Created {len(exam_chunks)} question chunks")
        logger.info(f"  → Created {len(marking_chunks)} marking chunks")

        # Create question index
        index = self._create_question_index(exam_chunks, subject)
        index_file = self.index_dir / f"exam_question_index_{subject}.json"
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)

        logger.info(f"  → Saved question index: {index_file.name}")

        # Update lockfile
        lockfile = self.get_lockfile()
        if subject not in lockfile["exams"]:
            lockfile["exams"][subject] = {
                "subject": subject_name,
                "years": {},
            }

        lockfile["exams"][subject]["years"][str(year)] = {
            "year": year,
            "imported_at": datetime.now().isoformat(),
            "question_count": len(exam_chunks),
            "marking_count": len(marking_chunks),
            "file": f"exam_{year}.txt",
        }
        self.save_lockfile(lockfile)

        return {
            "subject": subject_name,
            "year": year,
            "questions": len(exam_chunks),
            "marking_entries": len(marking_chunks),
        }

    def _create_exam_chunks(
        self,
        content: str,
        subject: str,
        subject_name: str,
        year: int,
    ) -> list[dict]:
        """Create exam question chunks."""
        chunks = []
        lines = content.split("\n")
        current_section = "Section A"
        question_num = 0

        in_exam_section = False
        current_question_lines = []

        for line in lines:
            line = line.strip()

            # Detect exam section
            if "## Section" in line and "Marking" not in line:
                in_exam_section = True
                current_section = line.replace("## ", "").strip()
                continue

            if in_exam_section and "## Marking" in line:
                break

            # Look for questions
            if in_exam_section and (
                "Question" in line or line.startswith("1.") or line.startswith("2.")
            ):
                # Save previous question
                if current_question_lines:
                    question_num += 1
                    content_text = "\n".join(current_question_lines)

                    # Extract marks
                    import re

                    marks_match = re.search(r"\((\d+)\s*marks?\)", content_text)
                    marks = int(marks_match.group(1)) if marks_match else 0

                    # Extract command terms
                    command_terms = []
                    for term in [
                        "analyse",
                        "evaluate",
                        "explain",
                        "discuss",
                        "describe",
                        "compare",
                        "contrast",
                        "justify",
                        "apply",
                    ]:
                        if term in content_text.lower():
                            command_terms.append(term.capitalize())

                    chunks.append(
                        {
                            "doc_id": f"exam_{subject}_{year}_Q{question_num:02d}",
                            "content": content_text,
                            "source_type": "exam",
                            "subject": subject_name,
                            "year": year,
                            "section": current_section,
                            "question_number": str(question_num),
                            "marks": marks,
                            "command_terms": command_terms,
                            "chunk_type": "question",
                            "module": current_section,
                        }
                    )

                current_question_lines = [line]

            elif in_exam_section and current_question_lines:
                if line and not line.startswith("---"):
                    current_question_lines.append(line)

        # Save last question
        if current_question_lines:
            question_num += 1
            content_text = "\n".join(current_question_lines)

            import re

            marks_match = re.search(r"\((\d+)\s*marks?\)", content_text)
            marks = int(marks_match.group(1)) if marks_match else 0

            command_terms = []
            for term in [
                "analyse",
                "evaluate",
                "explain",
                "discuss",
                "describe",
                "compare",
                "contrast",
                "justify",
                "apply",
            ]:
                if term in content_text.lower():
                    command_terms.append(term.capitalize())

            chunks.append(
                {
                    "doc_id": f"exam_{subject}_{year}_Q{question_num:02d}",
                    "content": content_text,
                    "source_type": "exam",
                    "subject": subject_name,
                    "year": year,
                    "section": current_section,
                    "question_number": str(question_num),
                    "marks": marks,
                    "command_terms": command_terms,
                    "chunk_type": "question",
                    "module": current_section,
                }
            )

        return chunks

    def _create_marking_chunks(
        self,
        content: str,
        subject: str,
        subject_name: str,
        year: int,
    ) -> list[dict]:
        """Create marking guideline chunks."""
        chunks = []
        lines = content.split("\n")
        in_marking_section = False
        current_criteria = []
        current_notes = []
        question_marks = 0
        question_num = None

        for line in lines:
            line = line.strip()

            if "## Marking" in line or "Marking Guidelines" in line:
                in_marking_section = True
                continue

            if (
                in_marking_section
                and "## " in line
                and "Marking" not in line
                and "Criteria" not in line
            ):
                # Save previous marking entry
                if question_num is not None and current_criteria:
                    chunks.append(
                        {
                            "doc_id": f"exam_{subject}_{year}_Q{question_num:02d}_marking",
                            "content": "\n".join(current_criteria),
                            "source_type": "marking",
                            "subject": subject_name,
                            "year": year,
                            "question_number": str(question_num),
                            "marks_available": question_marks,
                            "criteria": current_criteria,
                            "notes": current_notes,
                            "chunk_type": "marking_guideline",
                        }
                    )

                current_criteria = []
                current_notes = []
                question_num = None
                question_marks = 0

            if in_marking_section:
                # Look for question references
                import re

                q_match = re.search(r"Question\s*(\d+)", line, re.IGNORECASE)
                if q_match:
                    # Save previous entry
                    if question_num is not None and current_criteria:
                        chunks.append(
                            {
                                "doc_id": f"exam_{subject}_{year}_Q{question_num:02d}_marking",
                                "content": "\n".join(current_criteria),
                                "source_type": "marking",
                                "subject": subject_name,
                                "year": year,
                                "question_number": str(question_num),
                                "marks_available": question_marks,
                                "criteria": current_criteria,
                                "notes": current_notes,
                                "chunk_type": "marking_guideline",
                            }
                        )

                    question_num = int(q_match.group(1))
                    current_criteria = []
                    current_notes = []

                    marks_match = re.search(r"(\d+)\s*marks?", line, re.IGNORECASE)
                    if marks_match:
                        question_marks = int(marks_match.group(1))

                elif question_num is not None:
                    if line.startswith("-") or line.startswith("**"):
                        current_criteria.append(line)
                    elif (
                        line
                        and not line.startswith("---")
                        and "Sample" not in line
                        and "Response" not in line
                    ):
                        current_notes.append(line)

        # Save last entry
        if question_num is not None and current_criteria:
            chunks.append(
                {
                    "doc_id": f"exam_{subject}_{year}_Q{question_num:02d}_marking",
                    "content": "\n".join(current_criteria),
                    "source_type": "marking",
                    "subject": subject_name,
                    "year": year,
                    "question_number": str(question_num),
                    "marks_available": question_marks,
                    "criteria": current_criteria,
                    "notes": current_notes,
                    "chunk_type": "marking_guideline",
                }
            )

        return chunks

    def _format_chunk(self, chunk: dict) -> str:
        """Format a chunk as markdown."""
        lines = [
            f"# {chunk['subject']} {chunk['year']} - {chunk['chunk_type'].replace('_', ' ').title()}",
            "",
            f"**Subject:** {chunk['subject']}",
            f"**Year:** {chunk['year']}",
            f"**Source Type:** {chunk['source_type']}",
        ]

        if chunk.get("section"):
            lines.append(f"**Section:** {chunk['section']}")

        if chunk.get("question_number"):
            lines.append(f"**Question:** {chunk['question_number']}")

        if chunk.get("marks"):
            lines.append(f"**Marks:** {chunk['marks']}")

        if chunk.get("command_terms"):
            lines.append(f"**Command Terms:** {', '.join(chunk['command_terms'])}")

        if chunk.get("criteria"):
            lines.append(f"**Marks Available:** {chunk.get('marks_available', 0)}")

        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append(chunk["content"])

        return "\n".join(lines)

    def _create_question_index(
        self,
        exam_chunks: list[dict],
        subject: str,
    ) -> dict:
        """Create a question index for practice generation."""
        index = {
            "subject": subject,
            "created_at": datetime.now().isoformat(),
            "total_questions": len(exam_chunks),
            "by_year": {},
            "by_section": {},
            "by_marks": {},
            "by_command_term": {},
            "questions": [],
        }

        for chunk in exam_chunks:
            year = chunk.get("year", 0)
            section = chunk.get("section", "Unknown")
            marks = chunk.get("marks", 0)
            command_terms = chunk.get("command_terms", [])
            q_num = chunk.get("question_number", "")

            # By year
            if year not in index["by_year"]:
                index["by_year"][year] = {"total": 0, "questions": []}
            index["by_year"][year]["total"] += 1
            index["by_year"][year]["questions"].append(chunk["doc_id"])

            # By section
            if section not in index["by_section"]:
                index["by_section"][section] = {"total": 0, "questions": []}
            index["by_section"][section]["total"] += 1
            index["by_section"][section]["questions"].append(chunk["doc_id"])

            # By marks
            marks_key = f"{marks}_marks"
            if marks_key not in index["by_marks"]:
                index["by_marks"][marks_key] = {"total": 0, "questions": []}
            index["by_marks"][marks_key]["total"] += 1
            index["by_marks"][marks_key]["questions"].append(chunk["doc_id"])

            # By command term
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

        return index

    async def run_import(
        self,
        subjects: list[str] | None = None,
        years: list[int] | None = None,
        skip_download: bool = False,
    ) -> dict:
        """Run the full import process."""
        logger.info("=" * 60)
        logger.info("HSC Exam Import")
        logger.info("=" * 60)

        # Determine which subjects to process
        if subjects is None:
            subjects = list(HSC_SUBJECTS.keys())
        else:
            subjects = [s for s in subjects if s in HSC_SUBJECTS]
            if not subjects:
                logger.error("No valid subjects specified")
                return {"success": False}

        # Determine which years to process
        if years is None:
            years = list(range(2020, 2025))  # Past 5 years
        years = [y for y in years if 2020 <= y <= 2024]

        results = {}
        for subject in subjects:
            info = HSC_SUBJECTS[subject]
            subject_years = [y for y in years if y in info["years"]]

            for year in subject_years:
                try:
                    result = await self.import_subject_year(subject, info["name"], year)
                    key = f"{subject}_{year}"
                    results[key] = {
                        "success": True,
                        "subject": info["name"],
                        "year": year,
                        "questions": result["questions"],
                        "marking_entries": result["marking_entries"],
                    }
                except Exception as e:
                    logger.error(f"  ✗ Error importing {info['name']} {year}: {e}")
                    key = f"{subject}_{year}"
                    results[key] = {
                        "success": False,
                        "subject": info["name"],
                        "year": year,
                        "error": str(e),
                    }

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("Import Summary")
        logger.info("=" * 60)

        success_count = sum(1 for r in results.values() if r["success"])
        total_count = len(results)

        logger.info(f"Successful: {success_count}/{total_count}")
        logger.info(f"Failed: {total_count - success_count}")

        if success_count > 0:
            logger.info(f"\nIndexes: {self.index_dir}")
            logger.info(f"Lockfile: {self.lockfile_path}")

        return {
            "success": success_count == total_count,
            "total": total_count,
            "successful": success_count,
            "failed": total_count - success_count,
            "results": results,
        }


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Import HSC exam papers and marking guidelines into knowledge bases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Subjects:
  maths_adv     - Mathematics Advanced
  biology       - Biology
  business      - Business Studies
  legal         - Legal Studies
  english_adv   - English Advanced

Years: 2020-2024 (past 5 years)

Examples:
  # Import all subjects and years
  python3 scripts/nesa/import_hsc_exams.py

  # Import specific subjects and years
  python3 scripts/nesa/import_hsc_exams.py --subjects maths_adv biology --years 2022 2023 2024

  # Skip download, use cached files
  python3 scripts/nesa/import_hsc_exams.py --skip-download
        """,
    )

    parser.add_argument(
        "--subjects",
        nargs="+",
        choices=list(HSC_SUBJECTS.keys()),
        help="Subjects to import (default: all)",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        choices=list(range(2020, 2025)),
        help="Years to import (default: 2020-2024)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading, use cached files only",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download even if cached files exist",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory for cached exam files",
    )
    parser.add_argument(
        "--kb-dir",
        type=Path,
        default=None,
        help="Knowledge base directory",
    )

    args = parser.parse_args()

    importer = HSCExamImporter(
        cache_dir=args.cache_dir,
        kb_base_dir=args.kb_dir,
        force_download=args.force_download,
    )

    results = await importer.run_import(
        subjects=args.subjects,
        years=args.years,
        skip_download=args.skip_download,
    )

    return 0 if results["success"] else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
