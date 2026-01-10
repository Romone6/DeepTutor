"""
Biology Answer Quality Guardrails
==================================

Enforces HSC Biology answer quality: correct terminology, clear cause-effect links,
proper process sequencing, and avoidance of vague language.

Usage:
    from extensions.agents.bio_answer_guardrails import (
        BioGuardrailResult,
        validate_bio_answer,
        rewrite_vague_sentences,
        generate_keyword_bank,
    )

    result = validate_bio_answer(
        question="Explain how enzymes work",
        answer="Enzymes are biological catalysts that speed up reactions.",
        topic="cells_and_molecules",
    )
    print(result.rewrite_suggestions)
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any

from src.logging import get_logger

logger = get_logger("BioGuardrails")


@dataclass
class BioIssue:
    """A detected issue in Biology answer."""

    type: str
    severity: str  # "error", "warning", "info"
    description: str
    original_text: str
    suggestion: str = ""
    keyword_missing: str = ""
    mark_impact: float = 0.0


@dataclass
class BioGuardrailResult:
    """Complete result of validating Biology answer."""

    is_acceptable: bool
    score: float
    issues: list[BioIssue]
    vague_sentences: list[str]
    missing_keywords: list[str]
    missing_links: list[str]
    process_issues: list[str]
    rewrite_suggestions: list[str] = field(default_factory=list)
    keywords_found: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_acceptable": self.is_acceptable,
            "score": self.score,
            "issues": [i.__dict__ for i in self.issues],
            "vague_sentences": self.vague_sentences,
            "missing_keywords": self.missing_keywords,
            "missing_links": self.missing_links,
            "process_issues": self.process_issues,
            "rewrite_suggestions": self.rewrite_suggestions,
            "keywords_found": self.keywords_found,
        }


VAGUE_PATTERNS = {
    "vague_quantifiers": {
        "patterns": [
            r"\b(some|many|lots of|a lot|few|several|quite|rather|somewhat|basically|essentially|kind of|sort of|like|stuff|things)\b",
        ],
        "suggestion": "Use specific quantities or descriptions",
        "examples": {
            "some": "Specify the number or proportion",
            "many": "Give a specific number or percentage",
            "lots of": "Replace with specific quantity",
            "a lot": "Replace with specific quantity",
        },
    },
    "vague_verbs": {
        "patterns": [
            r"\b(affects|influences|helps|does|makes|goes|happens|works)\b",
        ],
        "suggestion": "Use more precise scientific verbs",
        "examples": {
            "affects": "Specify the mechanism (e.g., inhibits, activates, regulates)",
            "influences": "Specify how (e.g., upregulates, downregulates, modulates)",
            "helps": "Specify the function (e.g., catalyzes, facilitates, promotes)",
            "works": "Specify the mechanism",
        },
    },
    "vague_nouns": {
        "patterns": [
            r"\b(process|thing|stuff|way|system|function|part)\b",
        ],
        "suggestion": "Use specific biological terms",
        "examples": {
            "process": "Specify which process (e.g., photosynthesis, mitosis, transcription)",
            "thing": "Replace with specific noun",
            "stuff": "Replace with specific substance (e.g., proteins, enzymes, hormones)",
            "way": "Specify the mechanism or pathway",
            "system": "Specify the system (e.g., nervous, circulatory, immune)",
            "function": "Specify the specific role",
            "part": "Specify the structure (e.g., organelle, tissue, organ)",
        },
    },
    "passive_vague": {
        "patterns": [
            r"\b(is caused by|is related to|is involved in|plays a role in|has an effect on)\b",
        ],
        "suggestion": "Use active, specific language describing mechanisms",
        "examples": {
            "is caused by": "Specify the direct cause and mechanism",
            "is related to": "Explain the specific relationship",
            "is involved in": "Describe the specific role",
        },
    },
}

KEYWORD_BANK = {
    "cells_and_molecules": {
        "required": [
            "cell membrane",
            "cytoplasm",
            "nucleus",
            "organelle",
            "mitochondria",
            "ribosome",
            "DNA",
            "RNA",
            "protein synthesis",
            "ATP",
            "enzyme",
            "catalyst",
            "substrate",
            "active site",
            "denaturation",
        ],
        "recommended": [
            "cell wall",
            "chloroplast",
            "vacuole",
            "nuclear membrane",
            "transcription",
            "translation",
            "mutation",
            "gene",
            "chromosome",
        ],
    },
    "homeostasis": {
        "required": [
            "homeostasis",
            "negative feedback",
            "thermoregulation",
            "osmoregulation",
            "hormone",
            "receptor",
            "effector",
            "stimulus",
            "response",
            "balance",
        ],
        "recommended": [
            "hypothalamus",
            "endocrine system",
            "nervous system",
            "insulin",
            "glucagon",
            "antidiuretic hormone",
            "kidney",
            "osmosis",
        ],
    },
    "genetics": {
        "required": [
            "gene",
            "allele",
            "dominant",
            "recessive",
            "genotype",
            "phenotype",
            "heredity",
            "inheritance",
            "DNA",
            "mutation",
            "variation",
            "selection",
        ],
        "recommended": [
            "Punnett square",
            "Mendel",
            "gamete",
            "zygote",
            "homozygous",
            "heterozygous",
            "codominance",
            "incomplete dominance",
            "sex-linked",
            "chromosome",
        ],
    },
    "evolution": {
        "required": [
            "natural selection",
            "adaptation",
            "evolution",
            "variation",
            "survival",
            "reproduction",
            "species",
            "population",
            "fitness",
            "mutation",
        ],
        "recommended": [
            "speciation",
            "divergence",
            "convergent evolution",
            "divergent evolution",
            "gradualism",
            "punctuated equilibrium",
            "fossil record",
            "selective pressure",
            "common ancestor",
        ],
    },
    "ecology": {
        "required": [
            "ecosystem",
            "community",
            "population",
            "habitat",
            "niche",
            "energy flow",
            "food chain",
            "food web",
            "photosynthesis",
            "respiration",
            "decomposer",
        ],
        "recommended": [
            "biomass",
            "trophic level",
            "keystone species",
            "succession",
            "abiotic",
            "biotic",
            "carrying capacity",
            "competition",
            "predation",
            "symbiosis",
        ],
    },
    "response_and_coordination": {
        "required": [
            "nervous system",
            "neuron",
            "synapse",
            "neurotransmitter",
            "stimulus",
            "response",
            "reflex",
            "hormone",
            "endocrine",
            "receptor",
        ],
        "recommended": [
            "action potential",
            "myelin",
            "dendrite",
            "axon",
            "brain",
            "spinal cord",
            "sensory neuron",
            "motor neuron",
            "interneuron",
            "adrenaline",
        ],
    },
}

CAUSE_EFFECT_MARKERS = {
    "strong": [
        r"\bcauses?\b",
        r"\bdirectly (?:leads to|results in|produces)\b",
        r"\btriggers?\b",
        r"\bactivates?\b",
        r"\binhibits?\b",
        r"\bregulates?\b",
        r"\bcontrols?\b",
        r"\bdue to\b",
        r"\bbecause of\b",
        r"\bconsequently\b",
        r"\btherefore\b",
        r"\bthus\b",
        r"\bleads to\b",
        r"\bresults in\b",
        r"\bproduces\b",
        r"\benables\b",
        r"\bfacilitates\b",
    ],
    "weak": [
        r"\b(is )?affects?\b",
        r"\b(is )?linked to\b",
        r"\b(is )?connected to\b",
        r"\b(has )?something to do with\b",
        r"\b(is )?related to\b",
        r"\bplays a role in\b",
        r"\binvolved in\b",
        r"\bhas an effect on\b",
    ],
}

PROCESS_MARKERS = {
    "sequential": [
        r"\b(first|second|third|next|then|after|before|finally|initially|subsequently)\b",
        r"\b(step \d+|stage \d+|phase \d+)\b",
        r"\b(sequence of|stages of|steps of|process of)\b",
    ],
    "missing_sequence": [
        r"\b(and|but|or)\s+\w+\s+(then|after|before|next)\b",
    ],
}


def extract_keywords_from_text(text: str) -> list[str]:
    """Extract potential biological terms from text."""
    text_lower = text.lower()
    found_keywords = []

    for topic, keywords in KEYWORD_BANK.items():
        for keyword in keywords.get("required", []):
            if keyword.lower() in text_lower:
                found_keywords.append(keyword)
        for keyword in keywords.get("recommended", []):
            if keyword.lower() in text_lower:
                found_keywords.append(keyword)

    return list(set(found_keywords))


def find_vague_language(text: str) -> list[dict[str, Any]]:
    """Find vague language patterns in text."""
    issues = []

    for category, config in VAGUE_PATTERNS.items():
        for pattern in config["patterns"]:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                original = match.group()
                suggestion = config["suggestion"]
                example = config["examples"].get(original.lower(), "")

                issues.append(
                    {
                        "type": category,
                        "severity": "warning",
                        "description": f"Vague language: '{original}'",
                        "original_text": original,
                        "suggestion": f"{suggestion}. {example}" if example else suggestion,
                    }
                )

    return issues


def check_cause_effect_links(text: str) -> list[dict[str, Any]]:
    """Check for cause-effect relationship clarity."""
    issues = []

    text_lower = text.lower()

    for pattern in CAUSE_EFFECT_MARKERS["weak"]:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            original = match.group()
            issues.append(
                {
                    "type": "weak_cause_effect",
                    "severity": "warning",
                    "description": f"Weak cause-effect link: '{original}'",
                    "original_text": original,
                    "suggestion": "Use more specific language: 'causes', 'triggers', 'activates', 'inhibits', or explain the mechanism",
                }
            )

    return issues


def check_process_sequence(text: str) -> list[dict[str, Any]]:
    """Check for proper process sequencing."""
    issues = []

    sentences = re.split(r"[.!?]", text)
    has_sequence_markers = any(
        re.search(pattern, sentence.lower())
        for sentence in sentences
        for pattern in PROCESS_MARKERS["sequential"]
    )

    process_keywords = ["process", "mechanism", "pathway", "cycle", "stages", "steps", "sequence"]
    has_process_words = any(kw in text.lower() for kw in process_keywords)

    if has_process_words and not has_sequence_markers:
        issues.append(
            {
                "type": "missing_sequence",
                "severity": "info",
                "description": "Process described but sequence not clearly indicated",
                "original_text": "",
                "suggestion": "Use sequence markers: 'First... Then... Finally...' or numbered steps",
            }
        )

    return issues


def validate_bio_answer(
    question: str,
    answer: str,
    topic: str = "cells_and_molecules",
    minimum_keywords: int = 3,
) -> BioGuardrailResult:
    """
    Validate Biology answer quality.

    Args:
        question: The Biology question
        answer: Student's answer
        topic: Topic area (cells_and_molecules, homeostasis, genetics, etc.)
        minimum_keywords: Minimum required keywords from the topic

    Returns:
        BioGuardrailResult with issues and suggestions
    """
    issues = []

    keywords_found = extract_keywords_from_text(answer)
    missing_keywords = []

    topic_keywords = KEYWORD_BANK.get(topic, {})
    required = topic_keywords.get("required", [])

    for keyword in required:
        if keyword.lower() not in answer.lower():
            missing_keywords.append(keyword)

    vague_issues = find_vague_language(answer)
    issues.extend([BioIssue(**i) for i in vague_issues])

    link_issues = check_cause_effect_links(answer)
    issues.extend([BioIssue(**i) for i in link_issues])

    process_issues = check_process_sequence(answer)
    issues.extend([BioIssue(**i) for i in process_issues])

    vague_sentences = []
    sentences = re.split(r"[.!?]", answer)
    for sentence in sentences:
        if len(sentence.strip()) > 10:
            vague = find_vague_language(sentence)
            if vague:
                vague_sentences.append(sentence.strip())

    rewrite_suggestions = []
    for issue in issues[:5]:
        if issue.suggestion:
            rewrite_suggestions.append(f"• {issue.description}: {issue.suggestion}")

    if missing_keywords and len(keywords_found) < minimum_keywords:
        missing_msg = f"Missing key terms: {', '.join(missing_keywords[:5])}"
        rewrite_suggestions.append(f"• {missing_msg}")

    score = 100.0
    for issue in issues:
        if issue.severity == "error":
            score -= 5.0
        elif issue.severity == "warning":
            score -= 2.0
        else:
            score -= 0.5

    score -= len(missing_keywords) * 3.0
    score = max(0, min(100, score))

    is_acceptable = score >= 70 and len(keywords_found) >= minimum_keywords

    return BioGuardrailResult(
        is_acceptable=is_acceptable,
        score=round(score, 1),
        issues=issues,
        vague_sentences=vague_sentences,
        missing_keywords=missing_keywords,
        missing_links=[i["original_text"] for i in link_issues],
        process_issues=[i["description"] for i in process_issues],
        rewrite_suggestions=rewrite_suggestions,
        keywords_found=keywords_found,
    )


def rewrite_vague_sentence(sentence: str, context: str = "") -> str:
    """
    Attempt to rewrite a vague sentence with more precise language.

    Args:
        sentence: The vague sentence
        context: Surrounding context for better rewriting

    Returns:
        Rewritten sentence with vague language replaced
    """
    rewritten = sentence

    vague_to_precise = {
        r"\bsome\b": "[specific quantity]",
        r"\bmany\b": "[specific number/percentage]",
        r"\blots of\b": "[specific quantity]",
        r"\ba lot\b": "[specific quantity]",
        r"\baffects\b": "[specific mechanism]",
        r"\binfluences\b": "[specific mechanism]",
        r"\bhelps\b": "[specific function]",
        r"\bworks\b": "[specific mechanism]",
        r"\bprocess\b": "[specific biological process]",
        r"\bthing\b": "[specific noun]",
        r"\bstuff\b": "[specific substance]",
        r"\bway\b": "[specific mechanism]",
        r"\bsystem\b": "[specific system]",
        r"\bfunction\b": "[specific role]",
        r"\bpart\b": "[specific structure]",
        r"\bis caused by\b": "[specific cause and mechanism]",
        r"\bis related to\b": "[specific relationship]",
        r"\bis involved in\b": "[specific role]",
        r"\bplays a role in\b": "[specific role]",
        r"\bhas an effect on\b": "[specific effect]",
    }

    for pattern, replacement in vague_to_precise.items():
        rewritten = re.sub(pattern, replacement, rewritten, flags=re.IGNORECASE)

    return rewritten


def generate_keyword_bank() -> dict[str, dict[str, list[str]]]:
    """
    Generate comprehensive Biology keyword bank.

    Returns:
        Keyword bank dictionary organized by topic
    """
    return KEYWORD_BANK


def get_keywords_for_topic(topic: str) -> dict[str, list[str]]:
    """Get required and recommended keywords for a topic."""
    return KEYWORD_BANK.get(topic, {"required": [], "recommended": []})


def check_required_terms(answer: str, topic: str) -> dict[str, Any]:
    """Check if required terminology is present."""
    topic_keywords = KEYWORD_BANK.get(topic, {"required": [], "recommended": []})
    answer_lower = answer.lower()

    required_found = [kw for kw in topic_keywords.get("required", []) if kw.lower() in answer_lower]
    required_missing = [
        kw for kw in topic_keywords.get("required", []) if kw.lower() not in answer_lower
    ]

    recommended_found = [
        kw for kw in topic_keywords.get("recommended", []) if kw.lower() in answer_lower
    ]
    recommended_missing = [
        kw for kw in topic_keywords.get("recommended", []) if kw.lower() not in answer_lower
    ]

    return {
        "topic": topic,
        "required_found": required_found,
        "required_missing": required_missing,
        "recommended_found": recommended_found,
        "recommended_missing": recommended_missing,
        "completeness_score": len(required_found)
        / max(len(topic_keywords.get("required", [])), 1)
        * 100,
    }


def analyze_answer_structure(answer: str) -> dict[str, Any]:
    """Analyze the structural quality of a Biology answer."""
    sentences = re.split(r"[.!?]+", answer)
    sentences = [s.strip() for s in sentences if s.strip()]

    has_intro = False
    has_conclusion = False
    has_definition = False
    has_process = False
    has_examples = False
    has_links = False

    intro_markers = ["is defined as", "refers to", "means", "involves", "includes"]
    conclusion_markers = ["therefore", "thus", "in conclusion", "this shows", "this demonstrates"]
    process_markers = ["first", "then", "next", "finally", "stage", "step"]
    example_markers = ["for example", "for instance", "such as", "including", "like"]
    link_markers = ["because", "causes", "leads to", "results in", "enables", "allows"]

    for i, sentence in enumerate(sentences):
        s_lower = sentence.lower()

        if i == 0 and any(m in s_lower for m in intro_markers):
            has_intro = True

        if i == len(sentences) - 1 and any(m in s_lower for m in conclusion_markers):
            has_conclusion = True

        if "defined as" in s_lower or "is a" in s_lower or "refers to" in s_lower:
            has_definition = True

        if any(m in s_lower for m in process_markers):
            has_process = True

        if any(m in s_lower for m in example_markers):
            has_examples = True

        if any(m in s_lower for m in link_markers):
            has_links = True

    structure_score = 0
    if has_intro:
        structure_score += 20
    if has_definition:
        structure_score += 20
    if has_process:
        structure_score += 20
    if has_examples:
        structure_score += 20
    if has_links:
        structure_score += 10
    if has_conclusion:
        structure_score += 10

    return {
        "sentence_count": len(sentences),
        "has_intro": has_intro,
        "has_definition": has_definition,
        "has_process": has_process,
        "has_examples": has_examples,
        "has_links": has_links,
        "has_conclusion": has_conclusion,
        "structure_score": structure_score,
        "structure_issues": []
        if structure_score >= 70
        else [
            "Missing introduction" if not has_intro else "",
            "Missing definition" if not has_definition else "",
            "Missing process sequence" if not has_process else "",
            "Missing examples" if not has_examples else "",
            "Missing cause-effect links" if not has_links else "",
            "Missing conclusion" if not has_conclusion else "",
        ],
    }
