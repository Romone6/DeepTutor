"""
Knowledge Base Ingestion CLI

A simple pipeline to ingest documents into knowledge bases:
    PDF -> chunks -> embeddings -> FAISS index

Usage:
    python scripts/ingest_kb.py --kb hsc_math_adv --source syllabus.pdf --source-type syllabus
    python scripts/ingest_kb.py --kb hsc_biology --source ./docs/ --topic genetics --dry-run
    python scripts/ingest_kb.py --kb hsc_math_adv --source paper_2023.pdf --source-type past_paper --topic calculus --module hsc --year 2023
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.knowledge_bases import get_kb_path, list_subjects
from data.knowledge_bases.ingest import IngestionPipeline, IngestionConfig


def parse_source_type(value: str) -> str:
    valid_types = ["textbook", "syllabus", "past_paper", "lecture_notes", "reference", "exam"]
    if value.lower() not in valid_types:
        raise argparse.ArgumentTypeError(
            f"Invalid source_type: {value}. Must be one of: {', '.join(valid_types)}"
        )
    return value.lower()


def parse_module(value: str) -> str:
    valid_modules = ["preliminary", "hsc"]
    if value.lower() not in valid_modules:
        raise argparse.ArgumentTypeError(
            f"Invalid module: {value}. Must be one of: {', '.join(valid_modules)}"
        )
    return value.lower()


def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into knowledge bases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Ingest a syllabus PDF:
    python scripts/ingest_kb.py --kb hsc_math_adv --source syllabus_2024.pdf --source-type syllabus

  Ingest with topic tagging:
    python scripts/ingest_kb.py --kb hsc_biology --source cell_biology.pdf --topic "cell_structure" --module hsc

  Dry run (validate without embedding):
    python scripts/ingest_kb.py --kb hsc_business --source marketing.pdf --dry-run

  Ingest all PDFs from a directory:
    python scripts/ingest_kb.py --kb hsc_legal --source ./docs/ --source-type textbook

Knowledge Base Names:
  hsc_math_adv    - HSC Mathematics Advanced
  hsc_biology     - HSC Biology
  hsc_business    - HSC Business Studies
  hsc_legal       - HSC Legal Studies
  hsc_english_adv - HSC English Advanced
        """,
    )

    parser.add_argument(
        "--kb",
        required=True,
        choices=list_subjects(),
        help="Knowledge base to ingest into",
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Source file or directory to ingest",
    )

    parser.add_argument(
        "--source-type",
        type=parse_source_type,
        required=True,
        help="Type of source document (textbook, syllabus, past_paper, lecture_notes, reference, exam)",
    )

    parser.add_argument(
        "--topic",
        default="general",
        help="Topic within the subject (default: general)",
    )

    parser.add_argument(
        "--module",
        type=parse_module,
        default="hsc",
        help="HSC module stage (preliminary or hsc, default: hsc)",
    )

    parser.add_argument(
        "--year",
        type=int,
        default=2024,
        help="Publication year (default: 2024)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate documents and metadata without generating embeddings",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum chunk size in characters (default: 1000)",
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between chunks in characters (default: 200)",
    )

    parser.add_argument(
        "--doc-id",
        help="Custom document ID (default: derived from filename)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of documents to process in parallel (default: 10)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress",
    )

    args = parser.parse_args()

    kb_path = get_kb_path(args.kb)
    if kb_path is None:
        print(f"Error: Knowledge base '{args.kb}' not found")
        print(f"Available knowledge bases: {', '.join(list_subjects())}")
        sys.exit(1)

    if not Path(args.source).exists():
        print(f"Error: Source path does not exist: {args.source}")
        sys.exit(1)

    config = IngestionConfig(
        kb_path=kb_path,
        source=args.source,
        source_type=args.source_type,
        topic=args.topic,
        module=args.module,
        year=args.year,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        doc_id=args.doc_id,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    pipeline = IngestionPipeline(config)

    try:
        result = asyncio.run(pipeline.run())
        if result["success"]:
            print("\n" + "=" * 60)
            print("✓ Ingestion completed successfully!")
            print(f"  Documents processed: {result['documents_processed']}")
            print(f"  Chunks created: {result['chunks_created']}")
            if not args.dry_run:
                print(f"  Index saved to: {result['index_path']}")
            print("=" * 60)
            sys.exit(0)
        else:
            print(f"\n✗ Ingestion failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n✗ Ingestion cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Ingestion error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
