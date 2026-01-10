#!/usr/bin/env python3
"""
Question Tagging Tool
=====================

CLI tool for tagging past-paper questions with archetypes and error traps.

Usage:
    python tag_items.py --add
    python tag_items.py --edit q_001
    python tag_items.py --list
    python tag_items.py --export tagged_questions.json
    python tag_items.py --generate-practice --archetype calculation --count 5
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Optional

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_tagging_module():
    """Load tagging module directly to avoid numpy dependency."""
    tagging_base_path = project_root / "extensions" / "knowledge" / "tagging" / "base.py"
    spec = importlib.util.spec_from_file_location(
        "extensions.knowledge.tagging.base", tagging_base_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load tagging module from {tagging_base_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["extensions.knowledge.tagging.base"] = module
    spec.loader.exec_module(module)
    return module


tagging = load_tagging_module()
QuestionArchetype = tagging.QuestionArchetype
ErrorTrap = tagging.ErrorTrap
DifficultyLevel = tagging.DifficultyLevel
TagSet = tagging.TagSet
TaggedQuestion = tagging.TaggedQuestion
TagManager = tagging.TagManager
suggest_archetype = tagging.suggest_archetype
suggest_error_traps = tagging.suggest_error_traps


def print_tag_options():
    """Print available tag options."""
    print("\n=== QUESTION ARCHETYPES ===")
    for archetype in QuestionArchetype:
        print(f"  {archetype.value:25s} - {get_archetype_description(archetype)}")

    print("\n=== ERROR TRAPS ===")
    for trap in ErrorTrap:
        print(f"  {trap.value:25s} - {get_error_trap_description(trap)}")

    print("\n=== DIFFICULTY LEVELS ===")
    for level in DifficultyLevel:
        print(f"  {level.value}")


def get_archetype_description(archetype: QuestionArchetype) -> str:
    """Get description for archetype."""
    descriptions = {
        QuestionArchetype.CALCULATION: "Numerical computation",
        QuestionArchetype.DEFINITION: "Define term/concept",
        QuestionArchetype.DEFINITION_APPLICATION: "Define then apply",
        QuestionArchetype.EXPLAIN_ANALYSE: "Explain and analyse",
        QuestionArchetype.DISCUSS_EVALUATE: "Discuss and evaluate",
        QuestionArchetype.COMPARE_CONTRAST: "Compare and contrast",
        QuestionArchetype.DIAGRAM_LABEL: "Diagram interpretation",
        QuestionArchetype.CASE_STUDY: "Apply to case",
        QuestionArchetype.EXTENDED_RESPONSE: "Long-form answer",
        QuestionArchetype.MULTI_STEP: "Multi-part question",
        QuestionArchetype.DATA_INTERPRETATION: "Interpret data",
        QuestionArchetype.HYPOTHESIS_TESTING: "Experiment design",
        QuestionArchetype.APPLICATION: "Apply concepts",
        QuestionArchetype.SYNTHESIS: "Combine concepts",
        QuestionArchetype.CRITICAL_THINKING: "Evaluate arguments",
        QuestionArchetype.PROCEDURE_DESCRIPTION: "Describe procedures",
        QuestionArchetype.GRAPH_CONSTRUCTION: "Build graphs",
        QuestionArchetype.EQUATION_BALANCING: "Balance equations",
        QuestionArchetype.KEY_TERM_MATCHING: "Match terms",
        QuestionArchetype.SHORT_ANSWER: "Brief answer",
    }
    return descriptions.get(archetype, "")


def get_error_trap_description(trap: ErrorTrap) -> str:
    """Get description for error trap."""
    descriptions = {
        ErrorTrap.SIGN_ERROR: "Positive/negative errors",
        ErrorTrap.UNIT_ERROR: "Unit conversion mistakes",
        ErrorTrap.CALCULATION_ERROR: "Arithmetic errors",
        ErrorTrap.FORMULA_MISUSE: "Wrong formula used",
        ErrorTrap.CONCEPT_MISUNDERSTANDING: "Concept errors",
        ErrorTrap.DIRECTION_ERROR: "Vector/direction errors",
        ErrorTrap.APPROXIMATION_ERROR: "Rounding errors",
        ErrorTrap.VERBAL_ERROR: "Word choice errors",
        ErrorTrap.DEFINITION_INCOMPLETE: "Partial definitions",
        ErrorTrap.STRUCTURE_ERROR: "Answer structure issues",
        ErrorTrap.DIAGRAM_LABEL_ERROR: "Diagram mistakes",
        ErrorTrap.DATA_MISREAD: "Data reading errors",
        ErrorTrap.ASSUMPTION_MADE: "Unwarranted assumptions",
        ErrorTrap.CONTEXT_IGNORED: "Context ignored",
        ErrorTrap.EXPLANATION_TOO_BRIEF: "Too short for marks",
        ErrorTrap.LOGIC_GAP: "Missing logical steps",
        ErrorTrap.COUNTER_EXAMPLE_MISSED: "No counter example",
        ErrorTrap.MARK_ALLOC_IGNORED: "Marks not matched",
    }
    return descriptions.get(trap, "")


def interactive_add(manager: TagManager) -> None:
    """Interactively add a new tagged question."""
    print("\n=== Add New Tagged Question ===")

    question_id = input("Question ID (e.g., math_2023_q5b): ").strip()
    if not question_id:
        print("Error: Question ID is required")
        return

    content = input("Question content: ").strip()
    if not content:
        print("Error: Question content is required")
        return

    source = input("Source (e.g., HSC 2023): ").strip() or "Unknown"
    year = int(input("Year: ").strip() or "2023")

    suggested = suggest_archetype(content)
    print(f"\nSuggested archetypes: {[a.value for a in suggested]}")

    print("\nEnter archetypes (comma-separated, or 'all' for all):")
    print_tag_options()
    archetype_input = input("Archetypes: ").strip()

    if archetype_input.lower() == "all":
        archetypes = list(QuestionArchetype)
    else:
        archetype_list = [a.strip() for a in archetype_input.split(",")]
        archetypes = []
        for a in archetype_list:
            try:
                archetypes.append(QuestionArchetype(a))
            except ValueError:
                print(f"Warning: Unknown archetype '{a}', skipping")

    print("\nEnter error traps to avoid (comma-separated, or 'suggest' for suggestions):")
    error_input = input("Error traps: ").strip()

    if error_input.lower() == "suggest":
        error_traps = suggest_error_traps(
            content, archetypes[0] if archetypes else QuestionArchetype.SHORT_ANSWER
        )
        print(f"Suggested: {[e.value for e in error_traps]}")
    else:
        error_list = [e.strip() for e in error_input.split(",")] if error_input else []
        error_traps = []
        for e in error_list:
            try:
                error_traps.append(ErrorTrap(e))
            except ValueError:
                print(f"Warning: Unknown error trap '{e}', skipping")

    difficulty_input = input("\nDifficulty (easy/medium/hard/extension): ").strip().lower()
    difficulty = (
        DifficultyLevel(difficulty_input)
        if difficulty_input in [d.value for d in DifficultyLevel]
        else None
    )

    marks_input = input("Marks (optional): ").strip()
    marks = int(marks_input) if marks_input else None

    model_answer = input("\nModel answer (optional, press Enter to skip): ").strip()
    marking_notes = input("Marking notes (optional, press Enter to skip): ").strip()

    tags = TagSet(
        archetypes=archetypes,
        error_traps=error_traps,
        difficulty=difficulty,
        marks=marks,
    )

    question = TaggedQuestion(
        question_id=question_id,
        content=content,
        source=source,
        year=year,
        tags=tags,
        model_answer=model_answer or None,
        marking_notes=marking_notes or None,
    )

    manager.add_tagged_question(question)
    print(f"\n✓ Added question: {question_id}")
    print(f"  Archetypes: {[a.value for a in archetypes]}")
    print(f"  Error traps: {[e.value for e in error_traps]}")


def interactive_edit(manager: TagManager, question_id: str) -> None:
    """Interactively edit an existing tagged question."""
    question = manager.get_question(question_id)
    if not question:
        print(f"Error: Question '{question_id}' not found")
        return

    print(f"\n=== Edit Question: {question_id} ===")
    print(f"Current content: {question.content[:80]}...")

    new_content = input("New content (Enter to keep): ").strip()
    if new_content:
        question.content = new_content

    print(f"\nCurrent archetypes: {[a.value for a in question.tags.archetypes]}")
    print("Enter new archetypes (comma-separated, Enter to keep):")
    archetype_input = input(": ").strip()
    if archetype_input:
        archetype_list = [a.strip() for a in archetype_input.split(",")]
        question.tags.archetypes = [QuestionArchetype(a) for a in archetype_list if a]

    print(f"\nCurrent error traps: {[e.value for e in question.tags.error_traps]}")
    print("Enter new error traps (comma-separated, Enter to keep):")
    error_input = input(": ").strip()
    if error_input:
        error_list = [e.strip() for e in error_input.split(",")]
        question.tags.error_traps = [ErrorTrap(e) for e in error_list if e]

    print(f"\nCurrent difficulty: {question.tags.difficulty}")
    difficulty_input = input("New difficulty (Enter to keep): ").strip().lower()
    if difficulty_input:
        question.tags.difficulty = DifficultyLevel(difficulty_input)

    print(f"\nCurrent marks: {question.tags.marks}")
    marks_input = input("New marks (Enter to keep): ").strip()
    if marks_input:
        question.tags.marks = int(marks_input)

    print(f"\n✓ Updated question: {question_id}")


def list_questions(manager: TagManager) -> None:
    """List all tagged questions."""
    questions = list(manager._tagged_questions.values())

    if not questions:
        print("\nNo tagged questions yet. Use --add to add one.")
        return

    print(f"\n=== Tagged Questions ({len(questions)}) ===\n")

    for q in sorted(questions, key=lambda x: (x.source, x.year)):
        archetype_labels = q.tags.get_archetype_labels()
        print(f"{q.question_id}")
        print(f"  Source: {q.source} ({q.year})")
        print(f"  Archetypes: {', '.join(archetype_labels)}")
        print(f"  Difficulty: {q.tags.difficulty.value if q.tags.difficulty else 'N/A'}")
        print(f"  Marks: {q.tags.marks or 'N/A'}")
        print()


def export_questions(manager: TagManager, output_file: str) -> None:
    """Export tagged questions to JSON file."""
    manager.save_to_file(output_file)
    print(f"✓ Exported {len(manager._tagged_questions)} questions to {output_file}")


def generate_practice_set(
    manager: TagManager,
    archetype: Optional[str],
    difficulty: Optional[str],
    count: int,
) -> None:
    """Generate a practice set based on criteria."""
    archetype_enum = QuestionArchetype(archetype) if archetype else None
    difficulty_enum = DifficultyLevel(difficulty) if difficulty else None

    practice = manager.get_practice_set(
        archetype=archetype_enum,
        difficulty=difficulty_enum,
        count=count,
    )

    if not practice:
        print("\nNo matching questions found.")
        return

    print(f"\n=== Practice Set ({len(practice)} questions) ===\n")

    for i, q in enumerate(practice, 1):
        print(f"--- Question {i} ---")
        print(f"ID: {q.question_id}")
        print(f"Source: {q.source} ({q.year})")
        print(f"Marks: {q.tags.marks or 'N/A'}")
        print(f"Archetypes: {', '.join(q.tags.get_archetype_labels())}")
        print(f"Common traps: {', '.join(q.tags.get_error_trap_labels())}")
        print(f"\n{q.content}\n")

        if q.model_answer:
            print(f"[Model Answer Preview]: {q.model_answer[:200]}...")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Tag past-paper questions with archetypes and error traps"
    )
    parser.add_argument(
        "--add", action="store_true", help="Add a new tagged question interactively"
    )
    parser.add_argument("--edit", metavar="ID", help="Edit an existing question by ID")
    parser.add_argument("--list", action="store_true", help="List all tagged questions")
    parser.add_argument("--export", metavar="FILE", help="Export tagged questions to JSON file")
    parser.add_argument(
        "--import", metavar="FILE", dest="import_file", help="Import from JSON file"
    )
    parser.add_argument(
        "--generate-practice",
        action="store_true",
        help="Generate a practice set",
    )
    parser.add_argument(
        "--archetype",
        metavar="TYPE",
        help="Filter practice by archetype",
    )
    parser.add_argument(
        "--difficulty",
        metavar="LEVEL",
        help="Filter practice by difficulty",
    )
    parser.add_argument("--count", type=int, default=5, help="Number of questions for practice set")
    parser.add_argument("--file", "-f", default="tagged_questions.json", help="Tags database file")
    parser.add_argument("--show-tags", action="store_true", help="Show available tag options")

    args = parser.parse_args()

    if args.show_tags:
        print_tag_options()
        return

    if not any(
        [args.add, args.edit, args.list, args.export, args.import_file, args.generate_practice]
    ):
        parser.print_help()
        print("\nExamples:")
        print("  python tag_items.py --add                    # Add new question")
        print("  python tag_items.py --list                  # List all questions")
        print("  python tag_items.py --edit math_2023_q5b    # Edit question")
        print("  python tag_items.py --export my_tags.json   # Export to JSON")
        print("  python tag_items.py --generate-practice --archetype calculation --count 5")
        return

    manager = TagManager(tags_file=args.file)

    if args.import_file:
        try:
            manager.load_from_file(args.import_file)
            print(f"✓ Imported from {args.import_file}")
        except FileNotFoundError:
            print(f"Error: File {args.import_file} not found")
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in {args.import_file}")
        return

    if args.add:
        interactive_add(manager)
        manager.save_to_file()
        print(f"\nSaved to {args.file}")

    elif args.edit:
        interactive_edit(manager, args.edit)
        manager.save_to_file()
        print(f"\nSaved to {args.file}")

    elif args.list:
        list_questions(manager)

    elif args.export:
        export_questions(manager, args.export)

    elif args.generate_practice:
        generate_practice_set(manager, args.archetype, args.difficulty, args.count)


if __name__ == "__main__":
    main()
