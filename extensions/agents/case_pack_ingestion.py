"""
Case Pack Ingestion for Business Studies
=========================================

Utilities for ingesting business case packs (PDF/notes) into the knowledge base
with proper tagging for case-specific retrieval.

Usage:
    from extensions.agents.case_pack_ingestion import (
        ingest_case_pack,
        list_case_packs,
        delete_case_pack,
    )

    # Ingest a case pack
    result = await ingest_case_pack(
        file_path="/path/to/case.pdf",
        case_name="Apple Supply Chain",
        topic="operations",
    )

    # List available case packs
    cases = list_case_packs()

    # Delete a case pack
    delete_case_pack("apple_supply_chain")
"""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data.knowledge_bases.ingest import IngestionConfig, IngestionPipeline
from data.knowledge_bases.loaders import load_document


KB_PATH = Path("/Users/romonedunlop/AI TUTOR/DeepTutor/data/knowledge_bases")
BUSINESS_KB_PATH = KB_PATH / "kb_hsc_business"


@dataclass
class CasePackResult:
    """Result of case pack ingestion."""

    success: bool
    case_name: str
    doc_id: str
    chunks_created: int = 0
    error: str = ""
    metadata: dict[str, Any] = None


def generate_case_doc_id(case_name: str) -> str:
    """Generate a document ID for case pack."""
    import re
    clean = re.sub(r"[^a-zA-Z0-9]", "_", case_name)
    return f"case_{clean.lower()}"


async def ingest_case_pack(
    file_path: str | Path,
    case_name: str,
    topic: str = "general",
    chunk_size: int = 1500,
    chunk_overlap: 300,
) -> CasePackResult:
    """
    Ingest a business case pack into the knowledge base.

    Args:
        file_path: Path to the case file (PDF, TXT, or MD)
        case_name: Name of the case (e.g., "Apple Supply Chain 2023")
        topic: Business topic (operations, marketing, finance, etc.)
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks

    Returns:
        CasePackResult with ingestion status
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return CasePackResult(
            success=False,
            case_name=case_name,
            doc_id="",
            error=f"File not found: {file_path}",
        )

    doc_id = generate_case_doc_id(case_name)

    config = IngestionConfig(
        kb_path=BUSINESS_KB_PATH,
        source=str(file_path),
        source_type="case_notes",
        topic=topic,
        module="hsc",
        year=2024,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        doc_id=doc_id,
        verbose=True,
    )

    pipeline = IngestionPipeline(config)

    try:
        result = await pipeline.run()

        if result.get("success"):
            case_metadata = {
                "case_name": case_name,
                "doc_id": doc_id,
                "topic": topic,
                "source_type": "case_notes",
                "chunks_created": result.get("chunks_created", 0),
                "file_path": str(file_path),
            }

            cases_index_path = BUSINESS_KB_PATH / "_indexes" / "case_packs.json"
            cases_index = {"case_packs": []}
            if cases_index_path.exists():
                try:
                    with open(cases_index_path) as f:
                        cases_index = json.load(f)
                except Exception:
                    pass

            cases_index["case_packs"].append(case_metadata)

            with open(cases_index_path, "w") as f:
                json.dump(cases_index, f, indent=2)

            return CasePackResult(
                success=True,
                case_name=case_name,
                doc_id=doc_id,
                chunks_created=result.get("chunks_created", 0),
                metadata=case_metadata,
            )
        else:
            return CasePackResult(
                success=False,
                case_name=case_name,
                doc_id=doc_id,
                error=result.get("error", "Unknown error"),
            )

    except Exception as e:
        return CasePackResult(
            success=False,
            case_name=case_name,
            doc_id=doc_id,
            error=str(e),
        )


def list_case_packs() -> list[dict[str, Any]]:
    """List all available case packs."""
    cases_index_path = BUSINESS_KB_PATH / "_indexes" / "case_packs.json"

    if not cases_index_path.exists():
        return []

    try:
        with open(cases_index_path) as f:
            data = json.load(f)
        return data.get("case_packs", [])
    except Exception:
        return []


def get_case_by_name(case_name: str) -> dict[str, Any] | None:
    """Get a specific case pack by name."""
    case_name_lower = case_name.lower().replace(" ", "_")

    for case in list_case_packs():
        if case.get("doc_id", "").replace("case_", "").replace("_", " ") == case_name_lower:
            return case
        if case.get("case_name", "").lower() == case_name.lower():
            return case

    return None


def delete_case_pack(case_name: str) -> dict[str, Any]:
    """
    Delete a case pack from the knowledge base.

    Note: This removes the case from the index but does not delete
    the actual chunks from the vector index (FAISS doesn't support
    easy deletion without rebuilding).

    Args:
        case_name: Name or doc_id of the case to delete

    Returns:
        Result with success status
    """
    cases_index_path = BUSINESS_KB_PATH / "_indexes" / "case_packs.json"

    if not cases_index_path.exists():
        return {"success": False, "error": "Case pack index not found"}

    try:
        with open(cases_index_path) as f:
            cases_index = json.load(f)

        original_count = len(cases_index.get("case_packs", []))
        cases_index["case_packs"] = [
            c for c in cases_index.get("case_packs", [])
            if c.get("doc_id") != f"case_{case_name.lower().replace(' ', '_')}"
            and c.get("case_name", "").lower() != case_name.lower()
        ]

        if len(cases_index["case_packs"]) < original_count:
            with open(cases_index_path, "w") as f:
                json.dump(cases_index, f, indent=2)
            return {"success": True, "message": f"Case pack '{case_name}' removed from index"}
        else:
            return {"success": False, "error": f"Case pack '{case_name}' not found"}

    except Exception as e:
        return {"success": False, "error": str(e)}


def get_case_metadata(file_path: str | Path) -> dict[str, Any]:
    """
    Extract metadata from a case pack file.

    Args:
        file_path: Path to the case file

    Returns:
        Dictionary with extracted metadata
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return {"error": "File not found"}

    config = IngestionConfig(
        kb_path=BUSINESS_KB_PATH,
        source=str(file_path),
        source_type="case_notes",
    )

    doc = load_document(file_path, config)

    return {
        "doc_id": doc.doc_id,
        "file_name": file_path.name,
        "file_size": file_path.stat().st_size,
        "content_preview": doc.content[:500] + "..." if len(doc.content) > 500 else doc.content,
        "estimated_chunks": max(1, len(doc.content) // 1000),
    }
