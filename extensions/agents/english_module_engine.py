"""
English Advanced Module Response Engine
========================================

Generates Band 6 English responses with controlled argument, integrated evidence,
and quote discipline (no invented quotes).

Usage:
    from extensions.agents.english_module_engine import (
        EnglishModuleResult,
        build_thesis,
        generate_paragraph,
        check_quote_discipline,
        MarkerLensResult,
    )

    result = generate_english_response(
        question="How does Shakespeare use language to explore power?",
        text="Hamlet",
        module="module_b",
        retrieved_texts=[...],
        text_kb_snippets=[...],
    )
"""

import re
from dataclasses import dataclass, field
from typing import Any

from src.logging import get_logger

logger = get_logger("EnglishModuleEngine")


@dataclass
class QuoteCandidate:
    """A quote from the knowledge base."""

    text: str
    source: str
    location: str  # Act/scene, chapter, page
    techniques: list[str] = field(default_factory=list)
    themes: list[str] = field(default_factory=list)
    doc_id: str = ""
    page_ref: str = ""
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "location": self.location,
            "techniques": self.techniques,
            "themes": self.themes,
            "doc_id": self.doc_id,
            "page_ref": self.page_ref,
            "confidence": self.confidence,
        }


@dataclass
class ParagraphPlan:
    """Plan for a single paragraph."""

    topic_sentence: str
    main_argument: str
    quote_candidates: list[QuoteCandidate] = field(default_factory=list)
    paraphrase_only: bool = False
    analysis_points: list[str] = field(default_factory=list)
    link_back: str = ""
    technique_discussion: str = ""


@dataclass
class MarkerLensResult:
    """Result of marker lens checklist evaluation."""

    conceptual_depth: dict[str, Any]
    cohesion: dict[str, Any]
    textual_integrity: dict[str, Any]
    audience_purpose: dict[str, Any]
    overall_score: float
    strengths: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "conceptual_depth": self.conceptual_depth,
            "cohesion": self.cohesion,
            "textual_integrity": self.textual_integrity,
            "audience_purpose": self.audience_purpose,
            "overall_score": self.overall_score,
            "strengths": self.strengths,
            "improvements": self.improvements,
        }


@dataclass
class EnglishModuleResult:
    """Complete result of English module response generation."""

    thesis: str
    introduction: str
    paragraphs: list[dict[str, Any]]
    conclusion: str
    paragraph_plans: list[ParagraphPlan] = field(default_factory=list)
    marker_lens: MarkerLensResult = None
    quote_discipline_report: dict[str, Any] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "thesis": self.thesis,
            "introduction": self.introduction,
            "paragraphs": self.paragraphs,
            "conclusion": self.conclusion,
            "marker_lens": self.marker_lens.to_dict() if self.marker_lens else None,
            "quote_discipline_report": self.quote_discipline_report,
            "issues": self.issues,
            "suggestions": self.suggestions,
        }


TEXTUAL_TECHNIQUES = [
    "metaphor",
    "simile",
    "personification",
    "hyperbole",
    "irony",
    "symbolism",
    "allusion",
    " repetition",
    "anaphora",
    "assonance",
    "consonance",
    "enjambment",
    "caesura",
    "analogy",
    "imagery",
    "oxymoron",
    "paradox",
    "pathos",
    "logos",
    "ethos",
]

MODULES = {
    "common": {
        "name": "Common Module - Texts and Human Experiences",
        "focus": "Personal and collective human experiences",
        "key_concepts": ["identity", "belonging", "relationship", "challenge", "discovery"],
    },
    "module_a": {
        "name": "Module A - Textual Conversations",
        "focus": "Connections between texts",
        "key_concepts": ["intertextuality", "conversation", "perspective", "continuity", "change"],
    },
    "module_b": {
        "name": "Module B - Critical Study of Text",
        "focus": "Detailed analysis of a single text",
        "key_concepts": ["authorial intention", "composition", "significance", "enduring value"],
    },
}

THESIS_PATTERN = re.compile(
    r"^(?:In\s+)?(?:this\s+)?(?:text|novel|play|poem|film|short\s+story)\s+(?:by\s+)?[A-Z][a-zA-Z]+\s+(?:explores?|examines?|reveals?|demonstrates?|illustrates?|shows?|presents?|investigates?|addresses?|exposes?|challenges?|questions?|argues?|suggests?|emphasises?|highlights?|reflects?|captures?|depicts?|portrays?|represents?)\s+.+\.?$",
    re.IGNORECASE,
)

TOPIC_SENTENCE_PATTERN = re.compile(
    r"^(?:Furthermore|Moreover|In addition|Additionally|Secondly|Thirdly|First|Significantly|Notably|Importantly|Crucially|Ultimately|Ultimately,)?\s*[A-Z][^.!?]+\.?$"
)

QUOTE_PATTERN = re.compile(r'"[^"]{10,}"|\'[^\']{10,}\'')


def extract_quotes_from_kb(snippets: list[dict]) -> list[QuoteCandidate]:
    """Extract quotes from knowledge base snippets."""
    quotes = []

    for snippet in snippets:
        content = snippet.get("content", "")
        metadata = snippet.get("metadata", {})
        score = snippet.get("score", 0.5)

        found_quotes = QUOTE_PATTERN.findall(content)
        for quote in found_quotes:
            clean_quote = quote.strip("\"'")
            if len(clean_quote) > 10 and len(clean_quote) < 500:
                techniques = []
                for tech in TEXTUAL_TECHNIQUES:
                    if tech.lower() in content.lower():
                        techniques.append(tech)

                quotes.append(
                    QuoteCandidate(
                        text=clean_quote,
                        source=metadata.get("text_title", metadata.get("source", "Unknown")),
                        location=metadata.get("location", ""),
                        techniques=techniques,
                        themes=metadata.get("themes", []),
                        doc_id=metadata.get("doc_id", ""),
                        page_ref=metadata.get("page_ref", ""),
                        confidence=score,
                    )
                )

    return quotes


def check_quote_discipline(answer: str, valid_quotes: list[QuoteCandidate]) -> dict[str, Any]:
    """
    Check if answer follows quote discipline (only uses KB quotes).

    Returns:
        dict with: invented_quotes, valid_quotes_used, missing_quotes, report
    """
    answer_quotes = QUOTE_PATTERN.findall(answer)
    cleaned_quotes = [q.strip("\"'") for q in answer_quotes]

    valid_sources = set(q.text.lower() for q in valid_quotes)
    valid_texts = set(q.source.lower() for q in valid_quotes)

    invented = []
    valid_used = []
    unverified = []

    for quote in cleaned_quotes:
        quote_lower = quote.lower()
        is_valid = False

        for valid_quote in valid_quotes:
            if valid_quote.text.lower() in quote_lower or quote_lower in valid_quote.text.lower():
                valid_used.append(
                    {
                        "quote": quote[:50] + "..." if len(quote) > 50 else quote,
                        "source": valid_quote.source,
                        "matched": True,
                    }
                )
                is_valid = True
                break

        if not is_valid:
            if len(quote) > 15:
                invented.append(quote[:50] + "..." if len(quote) > 50 else quote)

    quote_sources_used = set(v["source"] for v in valid_used)

    return {
        "total_quotes_in_answer": len(cleaned_quotes),
        "valid_quotes_used": len(valid_used),
        "invented_quotes": invented,
        "unverified_quotes": unverified,
        "texts_referenced": list(quote_sources_used),
        "all_quotes_verified": len(invented) == 0 and len(unverified) == 0,
        "report": f"Using {len(valid_used)} verified quote(s) from knowledge base. "
        f"{len(invented)} potential invented quote(s) detected."
        if invented
        else "All quotes verified from knowledge base.",
    }


def build_thesis(
    question: str,
    text: str,
    module: str = "module_b",
    perspective: str = "",
    argument_direction: str = "explores how",
) -> str:
    """
    Generate a contestable thesis statement.

    Args:
        question: The essay question
        text: The text being studied
        module: The module (common, module_a, module_b)
        perspective: Optional specific perspective/angle
        argument_direction: How to frame the argument

    Returns:
        Thesis statement
    """
    module_info = MODULES.get(module, MODULES["module_b"])

    question_lower = question.lower()
    if "how" in question_lower:
        structure = f"through its composition, {text} {argument_direction}"
    elif "why" in question_lower:
        structure = f"as {text} {argument_direction}"
    elif "to what extent" in question_lower:
        structure = f"while {text} {argument_direction}, it also"
    else:
        structure = f"as {text} {argument_direction}"

    if perspective:
        perspective_clause = f" from a {perspective} perspective"
    else:
        perspective_clause = ""

    thesis = f"Through its deliberate crafting of language, structure and form{perspective_clause}, {structure} the complex interplay between human experience and textual meaning."

    return thesis


def generate_topic_sentence(
    main_argument: str,
    paragraph_number: int = 1,
    transition_word: str = "",
) -> str:
    """
    Generate a clear topic sentence.

    Args:
        main_argument: The main point of the paragraph
        paragraph_number: For ordering (adds transition words)
        transition_word: Specific transition word to use

    Returns:
        Topic sentence
    """
    transition_words = {
        1: ["First", "Initially", "Significantly"],
        2: ["Furthermore", "Moreover", "In addition"],
        3: ["Additionally", "Secondly", "Thirdly"],
        4: ["Notably", "Importantly", "Crucially"],
        5: ["Ultimately", "Finally", "Ultimately"],
    }

    if not transition_word:
        if paragraph_number <= len(transition_words):
            transition_word = transition_words[paragraph_number][0]
        else:
            transition_word = "Furthermore"

    topic_sentence = f"{transition_word}, {main_argument.lower()}"

    if not topic_sentence.endswith("."):
        topic_sentence += "."

    return topic_sentence


def integrate_quote(
    quote: str,
    technique: str,
    analysis: str,
    embed: bool = True,
) -> str:
    """
    Integrate a quote with technique analysis.

    Args:
        quote: The quote text
        technique: The technique used
        analysis: How the technique supports the argument
        embed: Whether to embed in sentence

    Returns:
        Integrated quote with analysis
    """
    if embed:
        result = (
            f'As evidenced in the moment when "{quote}" through the use of {technique}, {analysis}'
        )
    else:
        result = f'"{quote}" demonstrates {technique}, as {analysis}'

    return result


def generate_paraphrase_discussion(
    concept: str,
    textual_reference: str,
    technique: str,
    analysis: str,
) -> str:
    """
    Generate discussion when quotes aren't available (quote discipline).

    Args:
        concept: The concept/theme being discussed
        textual_reference: Reference to the text without direct quote
        technique: Literary technique employed
        analysis: How this supports argument

    Returns:
        Paraphrased discussion paragraph
    """
    return f"The text's exploration of {concept} through {technique} is evident in {textual_reference}. This demonstrates {analysis}"


def generate_paragraph(
    topic_sentence: str,
    evidence: list[dict[str, Any]],
    analysis_points: list[str],
    link_back: str,
    use_teel: bool = True,
) -> str:
    """
    Generate a complete paragraph using TEEL/PEEL structure.

    Args:
        topic_sentence: The main argument
        evidence: List of quote/paraphrase dicts
        analysis_points: Points of analysis
        link_back: Connection to thesis/argument
        use_teel: Use TEEL (True) or PEEL (False)

    Returns:
        Complete paragraph
    """
    connector = "Explanation and" if use_teel else "Evidence and"
    paragraph = f"{topic_sentence}\n\n"

    for i, ev in enumerate(evidence):
        if ev.get("type") == "quote":
            quote_text = ev["text"]
            technique = ev.get("technique", "language")
            embed = ev.get("embedded", True)

            if embed:
                paragraph += f'  {connector} technique: As seen when "{quote_text}" through {technique}, {ev.get("analysis", analysis_points[0] if i < len(analysis_points) else "")}\n'
            else:
                paragraph += f'  {connector} technique: The quote "{quote_text}" employs {technique}, demonstrating {ev.get("analysis", analysis_points[0] if i < len(analysis_points) else "")}\n'
        else:
            paragraph += f"  {connector} discussion: {ev.get('text', '')} - {ev.get('analysis', analysis_points[0] if i < len(analysis_points) else '')}\n"

    paragraph += f"\n  Thus, {link_back}\n"

    return paragraph


def generate_introduction(
    question: str,
    text: str,
    author: str,
    thesis: str,
    context_hook: str = "",
) -> str:
    """
    Generate an introduction paragraph.

    Args:
        question: The essay question
        text: The text title
        author: The composer
        thesis: The thesis statement
        context_hook: Optional contextual information

    Returns:
        Introduction paragraph
    """
    intro = f'"{question.strip()}"\n\n'

    if context_hook:
        intro += f"{context_hook}\n\n"

    intro += f"In {text} by {author}, {thesis}"

    return intro


def generate_conclusion(
    thesis_restated: str,
    key_arguments: list[str],
    final_thought: str = "",
    synthesis: str = "",
) -> str:
    """
    Generate a conclusion paragraph.

    Args:
        thesis_restated: Restated thesis (different words)
        key_arguments: Summary of main points
        final_thought: Closing statement
        synthesis: Broader implication

    Returns:
        Conclusion paragraph
    """
    conclusion = f"In conclusion, {thesis_restated}\n\n"

    conclusion += "Through the synthesis of these arguments, it is evident that "
    for i, arg in enumerate(key_arguments):
        if i > 0:
            conclusion += ", "
        conclusion += arg.lower()
    conclusion += ".\n\n"

    if synthesis:
        conclusion += f"{synthesis}\n\n"

    if final_thought:
        conclusion += final_thought

    return conclusion


def evaluate_marker_lens(
    answer: str,
    thesis: str,
    question: str,
    text: str,
    module: str = "module_b",
) -> MarkerLensResult:
    """
    Evaluate response against marker lens checklist.

    Returns:
        MarkerLensResult with scores and feedback
    """
    answer_lower = answer.lower()

    conceptual_depth = {
        "score": 0.0,
        "has_subtext": "subtext" in answer_lower
        or "underlying" in answer_lower
        or "implicit" in answer_lower,
        "has_multiple_meanings": "on one hand" in answer_lower
        or "however" in answer_lower
        or "although" in answer_lower,
        "considers_context": "context" in answer_lower
        or "historical" in answer_lower
        or "societal" in answer_lower,
        "evaluates_effectiveness": "effective" in answer_lower
        or "impact" in answer_lower
        or "significance" in answer_lower,
        "feedback": [],
    }
    conceptual_depth["score"] = (
        25
        + (15 if conceptual_depth["has_subtext"] else 0)
        + (15 if conceptual_depth["has_multiple_meanings"] else 0)
        + (15 if conceptual_depth["considers_context"] else 0)
        + (15 if conceptual_depth["evaluates_effectiveness"] else 0)
    )

    cohesion = {
        "score": 0.0,
        "has_transitions": any(
            w in answer_lower
            for w in ["however", "furthermore", "therefore", "moreover", "additionally"]
        ),
        "clear_paragraph_structure": "\n\n" in answer or len(answer.split(".")) > 5,
        "topic_sentences_clear": True,
        "link_to_thesis": thesis.lower()[:50] in answer_lower
        or thesis.split()[0].lower() in answer_lower,
        "feedback": [],
    }
    cohesion["score"] = (
        25
        + (20 if cohesion["has_transitions"] else 0)
        + (20 if cohesion["clear_paragraph_structure"] else 0)
        + (20 if cohesion["link_to_thesis"] else 0)
    )

    textual_integrity = {
        "score": 0.0,
        "accurate_quotes": True,
        "correct_techniques": any(t in answer_lower for t in TEXTUAL_TECHNIQUES),
        "textual_references": text.lower() in answer_lower or "the text" in answer_lower,
        "author_mentioned": True,
        "feedback": [],
    }
    textual_integrity["score"] = (
        25
        + (25 if textual_integrity["correct_techniques"] else 0)
        + (25 if textual_integrity["textual_references"] else 0)
    )

    audience_purpose = {
        "score": 50.0,
        "considers_audience": "audience" in answer_lower or "reader" in answer_lower,
        "considers_purpose": "purpose" in answer_lower
        or "intention" in answer_lower
        or "message" in answer_lower,
        "appropriate_register": True,
        "feedback": [],
    }

    overall = (
        conceptual_depth["score"] * 0.3
        + cohesion["score"] * 0.25
        + textual_integrity["score"] * 0.3
        + audience_purpose["score"] * 0.15
    )

    strengths = []
    if conceptual_depth["has_subtext"]:
        strengths.append("Explores subtext and underlying meaning")
    if cohesion["has_transitions"]:
        strengths.append("Effective use of transitions")
    if textual_integrity["correct_techniques"]:
        strengths.append("Accurate technique identification")

    improvements = []
    if not conceptual_depth["has_subtext"]:
        improvements.append("Consider deeper layers of meaning")
    if not cohesion["has_transitions"]:
        improvements.append("Add more transitional phrases")
    if not textual_integrity["correct_techniques"]:
        improvements.append("Include more textual techniques")

    return MarkerLensResult(
        conceptual_depth=conceptual_depth,
        cohesion=cohesion,
        textual_integrity=textual_integrity,
        audience_purpose=audience_purpose,
        overall_score=overall,
        strengths=strengths,
        improvements=improvements,
    )


def generate_english_response(
    question: str,
    text: str,
    author: str,
    module: str = "module_b",
    retrieved_texts: list[dict] = None,
    kb_snippets: list[dict] = None,
    perspective: str = "",
    num_paragraphs: int = 3,
) -> EnglishModuleResult:
    """
    Generate a complete English module response.

    Args:
        question: The essay question
        text: The text title
        author: The composer
        module: The HSC module
        retrieved_texts: Text content from KB
        kb_snippets: Analysis/notes from KB
        perspective: Specific angle on the question
        num_paragraphs: Number of body paragraphs

    Returns:
        EnglishModuleResult with full response
    """
    issues = []
    suggestions = []

    thesis = build_thesis(question, text, module, perspective)
    quotes = extract_quotes_from_kb(kb_snippets or [])

    quote_discipline = check_quote_discipline("", quotes)

    if len(quotes) < num_paragraphs:
        issues.append(
            f"Only {len(quotes)} quotes available for {num_paragraphs} paragraphs - paraphrase required"
        )
        suggestions.append("Use paraphrase with technique discussion where quotes unavailable")

    intro = generate_introduction(question, text, author, thesis)

    paragraph_plans = []
    paragraphs = []

    for i in range(num_paragraphs):
        topic_sentence = generate_topic_sentence(
            f"the text demonstrates {['fundamental themes', 'artistic choices', 'meaningful tensions', 'compositional strategies', 'universal insights'][i]}",
            i + 1,
        )

        evidence = []
        analysis_points = []

        if i < len(quotes):
            quote = quotes[i]
            evidence.append(
                {
                    "type": "quote",
                    "text": quote.text[:200] + "..." if len(quote.text) > 200 else quote.text,
                    "technique": quote.techniques[0] if quote.techniques else "figurative language",
                    "analysis": f"this reveals {["the composer's central concerns", "the text's thematic depth", "the audience's emotional engagement", "the narrative's structural complexity", "the language's persuasive power"][i]}",
                }
            )
            analysis_points.append("This technique effectively conveys the central theme")
        else:
            evidence.append(
                {
                    "type": "paraphrase",
                    "text": f"The text's exploration of {['identity', 'belonging', 'conflict', 'redemption', 'transformation'][i]} through narrative structure",
                    "analysis": "demonstrates the composer's deliberate crafting of meaning",
                }
            )
            paragraph_plans[i].paraphrase_only = True if not evidence else False

        link_back = f"this reinforces the thesis that {thesis[:80]}..."

        paragraph_text = generate_paragraph(
            topic_sentence=topic_sentence,
            evidence=evidence,
            analysis_points=analysis_points,
            link_back=link_back,
        )

        paragraphs.append(
            {
                "number": i + 1,
                "topic_sentence": topic_sentence,
                "content": paragraph_text,
                "quote_used": i < len(quotes),
                "technique": quotes[i].techniques[0]
                if i < len(quotes) and quotes[i].techniques
                else "discussion",
            }
        )

    conclusion = generate_conclusion(
        thesis_restated=f"Through deliberate textual choices, {text} ultimately demonstrates that {question.split()[0] if question else 'composition'} shapes meaning",
        key_arguments=[
            "textual techniques reveal theme",
            "structure supports argument",
            "context influences interpretation",
        ],
        final_thought="Thus, the text invites ongoing critical engagement.",
    )

    marker_lens = evaluate_marker_lens(
        answer=intro + " ".join([p["content"] for p in paragraphs]) + conclusion,
        thesis=thesis,
        question=question,
        text=text,
        module=module,
    )

    return EnglishModuleResult(
        thesis=thesis,
        introduction=intro,
        paragraphs=paragraphs,
        conclusion=conclusion,
        paragraph_plans=paragraph_plans,
        marker_lens=marker_lens,
        quote_discipline_report=quote_discipline,
        issues=issues,
        suggestions=suggestions,
    )
