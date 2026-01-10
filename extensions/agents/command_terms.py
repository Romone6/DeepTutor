"""
Command Terms Engine
====================

Maps NESA HSC command terms to required response structures and marking signals.
Provides scaffold templates for different subjects and command term combinations.

Usage:
    from extensions.agents.command_terms import (
        get_command_term_info,
        generate_scaffold,
        get_subject_scaffold,
        CommandTermEngine,
    )

    # Get command term info
    info = get_command_term_info("analyse", "mathematics")
    print(f"Required structure: {info.structure}")

    # Generate scaffold for a question
    scaffold = generate_scaffold(
        subject="english_adv",
        command_term="analyse",
        question="How does the composer...",
        marks=10,
    )
    print(scaffold)

    # Check if answer follows expected structure
    engine = CommandTermEngine()
    feedback = engine.evaluate_structure(
        answer="The student's answer...",
        command_term="analyse",
        subject="english_adv",
    )
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.logging import get_logger

logger = get_logger("CommandTerms")


class CommandTerm(str, Enum):
    """NESA HSC command terms with their definitions."""

    ANALYSE = "analyse"
    APPLY = "apply"
    ASSESS = "assess"
    CALCULATE = "calculate"
    CLARIFY = "clarify"
    COMPARE = "compare"
    CONTRAST = "contrast"
    CRITIQUE = "critique"
    DEFINE = "define"
    DESCRIBE = "describe"
    DISCUSS = "discuss"
    DISTINGUISH = "distinguish"
    EVALUATE = "evaluate"
    EXAMINE = "examine"
    EXPLAIN = "explain"
    EXPLORE = "explore"
    FORMULATE = "formulate"
    IDENTIFY = "identify"
    INTERPRET = "interpret"
    JUSTIFY = "justify"
    OUTLINE = "outline"
    PREDICT = "predict"
    PROPOSE = "propose"
    STATE = "state"
    SUMMARISE = "summarise"
    SYNTHESISE = "synthesise"


@dataclass
class StructureComponent:
    """A component of the response structure."""

    name: str
    description: str
    required: bool = True
    marking_weight: float = 1.0
    example_phrases: list[str] = field(default_factory=list)


@dataclass
class CommandTermInfo:
    """Information about a command term."""

    term: str
    definition: str
    general_structure: list[StructureComponent]
    marking_signals: list[str]
    common_errors: list[str]
    tips: list[str]


@dataclass
class SubjectScaffold:
    """A subject-specific scaffold for a command term."""

    subject: str
    command_term: str
    template: str
    components: list[StructureComponent]
    word_guidance: str
    example_introductions: list[str]


# Default command term definitions and structures
COMMAND_TERM_DEFINITIONS = {
    CommandTerm.ANALYSE: {
        "definition": "Identify components and examine the relationship between them",
        "marking_signals": [
            "Identifies key components",
            "Examines relationships between components",
            "Shows how parts relate to whole",
            "Breaks down complex ideas systematically",
        ],
        "common_errors": [
            "Describing instead of analysing",
            "Missing relationships between components",
            "Superficial examination only",
            "Not addressing all components",
        ],
        "tips": [
            "Start by identifying key components",
            "Show how each component relates to others",
            "Explain the significance of relationships",
            "Draw conclusions about the whole",
        ],
    },
    CommandTerm.APPLY: {
        "definition": "Use knowledge in new situations",
        "marking_signals": [
            "Correctly identifies relevant concepts",
            "Applies concepts appropriately",
            "Shows understanding of when to use knowledge",
            "Demonstrates practical understanding",
        ],
        "common_errors": [
            "Using wrong concept for the situation",
            "Applying concepts incorrectly",
            "Missing key elements of the application",
            "Not adapting knowledge to context",
        ],
        "tips": [
            "Identify the relevant knowledge/concepts",
            "Adapt knowledge to the specific context",
            "Show each step of application",
            "Explain why this application works",
        ],
    },
    CommandTerm.ASSESS: {
        "definition": "Make a judgement based on criteria",
        "marking_signals": [
            "States clear criteria for judgement",
            "Evaluates against criteria systematically",
            "Considers strengths and weaknesses",
            "States a reasoned conclusion",
        ],
        "common_errors": [
            "No clear criteria stated",
            "Biased or one-sided assessment",
            "No conclusion or unclear conclusion",
            "Ignoring counter-evidence",
        ],
        "tips": [
            "Establish criteria for assessment",
            "Evaluate each criterion systematically",
            "Consider multiple perspectives",
            "Conclude with clear judgement",
        ],
    },
    CommandTerm.CALCULATE: {
        "definition": "Perform mathematical operations",
        "marking_signals": [
            "States correct formula/method",
            "Shows working steps clearly",
            "Correct substitutions",
            "States final answer with units",
        ],
        "common_errors": [
            "Wrong formula selected",
            "Calculation errors",
            "Missing units",
            "Not showing working",
        ],
        "tips": [
            "State the formula before starting",
            "Show each step clearly",
            "Check calculations carefully",
            "Include units in final answer",
        ],
    },
    CommandTerm.COMPARE: {
        "definition": "Show similarities between two or more things",
        "marking_signals": [
            "States similarities clearly",
            "Uses comparative language",
            "Addresses both/all items",
            "Organises comparison logically",
        ],
        "common_errors": [
            "Only describing one item",
            "No comparative language used",
            "Unequal treatment of items",
            "No summary of similarities",
        ],
        "tips": [
            "Identify what to compare",
            "Address each item systematically",
            "Use words like 'similarly', 'both', 'also'",
            "Summarise main similarities",
        ],
    },
    CommandTerm.CONTRAST: {
        "definition": "Show differences between two or more things",
        "marking_signals": [
            "States differences clearly",
            "Uses contrasting language",
            "Addresses both/all items",
            "Organises contrast logically",
        ],
        "common_errors": [
            "Only describing one item",
            "No contrasting language used",
            "Unequal treatment of items",
            "No summary of differences",
        ],
        "tips": [
            "Identify what to contrast",
            "Address each item systematically",
            "Use words like 'however', 'unlike', 'whereas'",
            "Summarise main differences",
        ],
    },
    CommandTerm.CRITIQUE: {
        "definition": "Give a balanced assessment including weaknesses and strengths",
        "marking_signals": [
            "Identifies strengths",
            "Identifies weaknesses",
            "Provides evidence for both",
            "Gives balanced conclusion",
        ],
        "common_errors": [
            "Only negative criticism",
            "Only positive assessment",
            "No evidence provided",
            "Unbalanced treatment",
        ],
        "tips": [
            "Identify both strengths and weaknesses",
            "Support each with evidence",
            "Consider multiple perspectives",
            "Provide balanced conclusion",
        ],
    },
    CommandTerm.DEFINE: {
        "definition": "Give the exact meaning of a word or concept",
        "marking_signals": [
            "States precise meaning",
            "Includes all key components",
            "Uses technical terminology correctly",
            "Provides context where needed",
        ],
        "common_errors": [
            "Incomplete definition",
            "Wrong or vague terminology",
            "Missing key components",
            "Using examples instead of definition",
        ],
        "tips": [
            "State the term clearly",
            "Include all key components",
            "Use precise terminology",
            "Add context if helpful",
        ],
    },
    CommandTerm.DESCRIBE: {
        "definition": "Give an account of characteristics or features",
        "marking_signals": [
            "Lists main characteristics",
            "Uses appropriate vocabulary",
            "Organises features logically",
            "Provides sufficient detail",
        ],
        "common_errors": [
            "Too brief or too detailed",
            "No logical organisation",
            "Missing key features",
            "Confusing description with explanation",
        ],
        "tips": [
            "Identify key features",
            "Organise features logically",
            "Use appropriate technical terms",
            "Provide enough detail for marks",
        ],
    },
    CommandTerm.DISCUSS: {
        "definition": "Present arguments for and against a proposition",
        "marking_signals": [
            "Presents multiple perspectives",
            "Provides evidence for arguments",
            "Considers both sides",
            "Reaches a reasoned conclusion",
        ],
        "common_errors": [
            "One-sided argument only",
            "No evidence provided",
            "No clear conclusion",
            "Disorganised presentation",
        ],
        "tips": [
            "Present multiple perspectives",
            "Support each with evidence",
            "Weigh up the arguments",
            "Conclude with clear position",
        ],
    },
    CommandTerm.EVALUATE: {
        "definition": "Make a judgement based on criteria and evidence",
        "marking_signals": [
            "States criteria for evaluation",
            "Makes judgement against criteria",
            "Provides supporting evidence",
            "Gives reasoned conclusion",
        ],
        "common_errors": [
            "No criteria stated",
            "Judgement not supported by evidence",
            "Ignoring contrary evidence",
            "Unclear or no conclusion",
        ],
        "tips": [
            "State evaluation criteria",
            "Judge against each criterion",
            "Provide supporting evidence",
            "Give clear conclusion",
        ],
    },
    CommandTerm.EXAMINE: {
        "definition": "Investigate in detail",
        "marking_signals": [
            "Thorough investigation",
            "Considers key aspects",
            "Uses evidence systematically",
            "Draws reasoned conclusions",
        ],
        "common_errors": [
            "Superficial treatment",
            "Missing key aspects",
            "No evidence or examples",
            "No conclusions drawn",
        ],
        "tips": [
            "Investigate all key aspects",
            "Use evidence systematically",
            "Consider different angles",
            "Draw out conclusions",
        ],
    },
    CommandTerm.EXPLAIN: {
        "definition": "Make clear why something is the way it is",
        "marking_signals": [
            "States what happens",
            "Gives reasons for what happens",
            "Shows causal relationships",
            "Uses evidence to support",
        ],
        "common_errors": [
            "Describing instead of explaining",
            "No reasons given",
            "Missing causal links",
            "No evidence provided",
        ],
        "tips": [
            "State what happens",
            "Give reasons why",
            "Show cause and effect",
            "Support with evidence",
        ],
    },
    CommandTerm.IDENTIFY: {
        "definition": "Recognise and name",
        "marking_signals": [
            "Correctly recognises item",
            "Names key features",
            "Distinguishes from similar items",
            "Uses correct terminology",
        ],
        "common_errors": [
            "Wrong item identified",
            "Incomplete identification",
            "Using wrong terminology",
            "Not distinguishing from similar",
        ],
        "tips": [
            "State the item clearly",
            "Name key features",
            "Use correct terminology",
            "Show how it differs from similar",
        ],
    },
    CommandTerm.INTERPRET: {
        "definition": "Explain the meaning of information",
        "marking_signals": [
            "Explains the meaning",
            "Shows understanding of significance",
            "Relates to broader context",
            "Uses evidence appropriately",
        ],
        "common_errors": [
            "Just restating information",
            "Missing the deeper meaning",
            "No broader context",
            "Misinterpreting information",
        ],
        "tips": [
            "Explain what the information means",
            "Show its significance",
            "Relate to broader context",
            "Use evidence to support",
        ],
    },
    CommandTerm.JUSTIFY: {
        "definition": "Give reasons for a decision or conclusion",
        "marking_signals": [
            "States the decision/conclusion",
            "Provides logical reasons",
            "Uses evidence to support",
            "Shows why other options are less suitable",
        ],
        "common_errors": [
            "No clear decision stated",
            "Reasons not logical",
            "No supporting evidence",
            "Ignoring why alternatives don't work",
        ],
        "tips": [
            "State your decision clearly",
            "Give logical reasons",
            "Support with evidence",
            "Show why other options are less appropriate",
        ],
    },
    CommandTerm.OUTLINE: {
        "definition": "Give the main features in summary form",
        "marking_signals": [
            "States main features",
            "Omits minor details",
            "Logical order",
            "Clear and concise",
        ],
        "common_errors": [
            "Too much detail",
            "Missing main features",
            "Disorganised presentation",
            "Not concise enough",
        ],
        "tips": [
            "Identify main features only",
            "Leave out minor details",
            "Present in logical order",
            "Be concise and clear",
        ],
    },
    CommandTerm.SUMMARISE: {
        "definition": "Give the main points in condensed form",
        "marking_signals": [
            "Captures main ideas",
            "Omits unnecessary detail",
            "Logical flow",
            "Appropriate length",
        ],
        "common_errors": [
            "Too brief (missing main ideas)",
            "Too detailed",
            "Disorganised",
            "Including unnecessary information",
        ],
        "tips": [
            "Identify the main ideas",
            "Omit unnecessary details",
            "Present in logical order",
            "Keep it concise",
        ],
    },
}

# Default general structure components
DEFAULT_STRUCTURES = {
    CommandTerm.ANALYSE: [
        StructureComponent(
            "Identify",
            "Identify the key components or elements",
            True,
            0.2,
            ["First, identify...", "The key components are..."],
        ),
        StructureComponent(
            "Examine",
            "Examine relationships between components",
            True,
            0.4,
            ["These components relate by...", "The relationship shows..."],
        ),
        StructureComponent(
            "Significance",
            "Explain the significance or implications",
            True,
            0.3,
            ["This indicates...", "The significance is..."],
        ),
        StructureComponent(
            "Conclusion",
            "Draw a conclusion about the whole",
            True,
            0.1,
            ["Therefore...", "In conclusion..."],
        ),
    ],
    CommandTerm.APPLY: [
        StructureComponent(
            "Identify Concept",
            "Identify the relevant knowledge/concepts",
            True,
            0.2,
            ["The relevant concept is...", "This situation requires..."],
        ),
        StructureComponent(
            "Adapt",
            "Adapt knowledge to the specific context",
            True,
            0.4,
            ["Applying this to...", "In this context..."],
        ),
        StructureComponent(
            "Demonstrate",
            "Demonstrate the application",
            True,
            0.3,
            ["This shows that...", "The application demonstrates..."],
        ),
        StructureComponent(
            "Result", "State the result or outcome", True, 0.1, ["Therefore...", "The result is..."]
        ),
    ],
    CommandTerm.ASSESS: [
        StructureComponent(
            "Criteria",
            "State the criteria for assessment",
            True,
            0.2,
            ["The criteria for assessment are...", "To assess this, we consider..."],
        ),
        StructureComponent(
            "Evaluation",
            "Evaluate against each criterion",
            True,
            0.4,
            ["In terms of..., it...", "Regarding the criterion of..."],
        ),
        StructureComponent(
            "Evidence",
            "Provide evidence for evaluation",
            True,
            0.2,
            ["This is shown by...", "Evidence includes..."],
        ),
        StructureComponent(
            "Judgement",
            "State a reasoned judgement",
            True,
            0.2,
            ["Overall, this rates as...", "Based on the criteria..."],
        ),
    ],
    CommandTerm.CALCULATE: [
        StructureComponent(
            "Method",
            "State the formula or method",
            True,
            0.2,
            ["Using the formula...", "The method is..."],
        ),
        StructureComponent(
            "Working", "Show the working steps", True, 0.5, ["Substituting...", "Calculating..."]
        ),
        StructureComponent(
            "Answer",
            "State the final answer with units",
            True,
            0.3,
            ["Therefore, the answer is...", "Final answer:..."],
        ),
    ],
    CommandTerm.EVALUATE: [
        StructureComponent(
            "Criteria",
            "State the evaluation criteria",
            True,
            0.2,
            ["The criteria are...", "To evaluate, we consider..."],
        ),
        StructureComponent(
            "Judgement",
            "Make a judgement against criteria",
            True,
            0.4,
            ["In relation to...", "Against the criterion of..."],
        ),
        StructureComponent(
            "Evidence",
            "Provide supporting evidence",
            True,
            0.2,
            ["This is demonstrated by...", "Evidence shows..."],
        ),
        StructureComponent(
            "Conclusion",
            "Give a reasoned conclusion",
            True,
            0.2,
            ["Therefore, the evaluation is...", "In summary..."],
        ),
    ],
    CommandTerm.EXPLAIN: [
        StructureComponent(
            "What",
            "State what happens or exists",
            True,
            0.2,
            ["This occurs when...", "The situation is..."],
        ),
        StructureComponent(
            "Why",
            "Give reasons or causes",
            True,
            0.5,
            ["This happens because...", "The cause is..."],
        ),
        StructureComponent(
            "Evidence",
            "Support with evidence or examples",
            True,
            0.2,
            ["For example...", "This is shown by..."],
        ),
        StructureComponent(
            "Result",
            "State the consequence or implication",
            True,
            0.1,
            ["Therefore...", "As a result..."],
        ),
    ],
    CommandTerm.DESCRIBE: [
        StructureComponent(
            "Overview",
            "Give a general overview",
            True,
            0.2,
            ["The key features are...", "This involves..."],
        ),
        StructureComponent(
            "Details",
            "Provide specific characteristics",
            True,
            0.5,
            ["Specifically...", "The characteristics include..."],
        ),
        StructureComponent(
            "Examples", "Give relevant examples", True, 0.2, ["For example...", "Such as..."]
        ),
        StructureComponent(
            "Organisation", "Organise features logically", True, 0.1, ["First...", "Secondly..."]
        ),
    ],
    CommandTerm.DISCUSS: [
        StructureComponent(
            "Argument 1",
            "Present one side of the argument",
            True,
            0.25,
            ["On one hand...", "One perspective is..."],
        ),
        StructureComponent(
            "Argument 2",
            "Present the other side",
            True,
            0.25,
            ["On the other hand...", "However,..."],
        ),
        StructureComponent(
            "Analysis",
            "Analyse strengths and weaknesses",
            True,
            0.3,
            ["The strength of this view is...", "A weakness is..."],
        ),
        StructureComponent(
            "Conclusion",
            "Draw a reasoned conclusion",
            True,
            0.2,
            ["Therefore, I conclude...", "In weighing the evidence..."],
        ),
    ],
}


class CommandTermEngine:
    """
    Engine for HSC command term scaffolding and structural analysis.
    """

    def __init__(self):
        self._subject_scaffolds: dict[str, dict[str, SubjectScaffold]] = {}
        self._register_subject_scaffolds()

    def _register_subject_scaffolds(self) -> None:
        """Register subject-specific scaffold variations."""
        self._subject_scaffolds = {
            "mathematics": self._create_mathematics_scaffolds(),
            "maths_adv": self._create_mathematics_scaffolds(),
            "biology": self._create_biology_scaffolds(),
            "business": self._create_business_scaffolds(),
            "business_studies": self._create_business_scaffolds(),
            "legal": self._create_legal_scaffolds(),
            "legal_studies": self._create_legal_scaffolds(),
            "english_adv": self._create_english_scaffolds(),
            "english_advanced": self._create_english_scaffolds(),
            "chemistry": self._create_chemistry_scaffolds(),
            "physics": self._create_physics_scaffolds(),
            "general": self._create_general_scaffolds(),
        }

    def _create_mathematics_scaffolds(self) -> dict[str, SubjectScaffold]:
        """Mathematics-specific scaffolds (method-working-answer-check)."""
        scaffolds = {}

        scaffolds["calculate"] = SubjectScaffold(
            subject="mathematics",
            command_term="calculate",
            template="""**MATHEMATICS RESPONSE STRUCTURE**

1. **STATE METHOD**: State the formula or method to be used
2. **SHOW WORKING**: Display each step clearly with substitutions
3. **GIVE ANSWER**: State final answer with units
4. **CHECK**: Briefly verify the answer makes sense

**Marks Allocation:**
- Method: {method_marks} marks
- Working: {working_marks} marks
- Answer: {answer_marks} marks""",
            components=[
                StructureComponent(
                    "Method",
                    "State formula/method",
                    True,
                    0.2,
                    ["Using the formula...", "The method is..."],
                ),
                StructureComponent(
                    "Working",
                    "Show substitution and calculation",
                    True,
                    0.5,
                    ["Substituting values...", "Calculating..."],
                ),
                StructureComponent(
                    "Answer", "Final answer with units", True, 0.3, ["Therefore...", "Answer:"]
                ),
                StructureComponent(
                    "Check",
                    "Verify answer",
                    False,
                    0.0,
                    ["Checking...", "This makes sense because..."],
                ),
            ],
            word_guidance="Keep working clear and stepwise. No prose needed for maths.",
            example_introductions=[
                "Using the quadratic formula...",
                "The derivative is found by...",
                "Applying Pythagoras' theorem...",
            ],
        )

        scaffolds["explain"] = SubjectScaffold(
            subject="mathematics",
            command_term="explain",
            template="""**MATHEMATICS EXPLANATION STRUCTURE**

1. **CONCEPT**: State the mathematical concept or theorem
2. **REASON**: Explain why it works (the logic/reasoning)
3. **APPLICATION**: Show how to apply it
4. **EXAMPLE**: Give a numerical or algebraic example

**Marks Allocation:**
- Concept: {concept_marks} marks
- Reason: {reason_marks} marks
- Application: {application_marks} marks
- Example: {example_marks} marks""",
            components=[
                StructureComponent(
                    "Concept",
                    "State the concept/theorem",
                    True,
                    0.2,
                    ["The concept of...", "The theorem states that..."],
                ),
                StructureComponent(
                    "Reason",
                    "Explain why it works",
                    True,
                    0.3,
                    ["This works because...", "The reasoning is..."],
                ),
                StructureComponent(
                    "Application",
                    "Show application",
                    True,
                    0.3,
                    ["To apply this...", "In this case..."],
                ),
                StructureComponent(
                    "Example", "Give example", True, 0.2, ["For example...", "A case in point..."]
                ),
            ],
            word_guidance="Use mathematical language precisely. Show logical progression.",
            example_introductions=[
                "This function is decreasing because...",
                "The limit exists because...",
                "This result follows from...",
            ],
        )

        scaffolds["analyse"] = SubjectScaffold(
            subject="mathematics",
            command_term="analyse",
            template="""**MATHEMATICAL ANALYSIS STRUCTURE**

1. **IDENTIFY**: Identify the mathematical features or components
2. **RELATIONSHIPS**: Examine relationships between features
3. **PROPERTIES**: Analyse properties and patterns
4. **CONCLUSION**: State what this reveals about the problem

**Marks Allocation:**
- Identify: {identify_marks} marks
- Relationships: {relationships_marks} marks
- Properties: {properties_marks} marks
- Conclusion: {conclusion_marks} marks""",
            components=[
                StructureComponent(
                    "Identify",
                    "Identify key features",
                    True,
                    0.25,
                    ["The key features are...", "This involves..."],
                ),
                StructureComponent(
                    "Relationships",
                    "Examine relationships",
                    True,
                    0.35,
                    ["The relationship between... is...", "These connect by..."],
                ),
                StructureComponent(
                    "Properties",
                    "Analyse properties",
                    True,
                    0.25,
                    ["A key property is...", "This shows that..."],
                ),
                StructureComponent(
                    "Conclusion",
                    "State implication",
                    True,
                    0.15,
                    ["This reveals...", "Therefore..."],
                ),
            ],
            word_guidance="Be precise with mathematical terminology. Show logical analysis.",
            example_introductions=[
                "The function has the following properties...",
                "Analysing the gradient...",
                "Examining the behaviour...",
            ],
        )

        return scaffolds

    def _create_biology_scaffolds(self) -> dict[str, SubjectScaffold]:
        """Biology-specific scaffolds (definition-process-link-example)."""
        scaffolds = {}

        scaffolds["explain"] = SubjectScaffold(
            subject="biology",
            command_term="explain",
            template="""**BIOLOGY EXPLANATION STRUCTURE**

1. **DEFINE**: Define key terms or processes
2. **PROCESS**: Describe the process or mechanism
3. **LINK**: Explain cause-effect relationships
4. **EXAMPLE**: Give biological example

**Marks Allocation:**
- Define: {define_marks} marks
- Process: {process_marks} marks
- Link: {link_marks_bio} marks
- Example: {example_marks} marks""",
            components=[
                StructureComponent(
                    "Define",
                    "Define key terms",
                    True,
                    0.15,
                    [" is defined as...", "The term means..."],
                ),
                StructureComponent(
                    "Process",
                    "Describe the process",
                    True,
                    0.35,
                    ["The process involves...", "This occurs through..."],
                ),
                StructureComponent(
                    "Link",
                    "Explain cause-effect",
                    True,
                    0.3,
                    ["This leads to...", "As a result..."],
                ),
                StructureComponent(
                    "Example", "Give example", True, 0.2, ["For example...", "In mammals, this..."]
                ),
            ],
            word_guidance="Use correct biological terminology. Link mechanisms to outcomes.",
            example_introductions=[
                "Photosynthesis is the process by which...",
                "This occurs because...",
                "Enzymes work by...",
            ],
        )

        scaffolds["analyse"] = SubjectScaffold(
            subject="biology",
            command_term="analyse",
            template="""**BIOLOGICAL ANALYSIS STRUCTURE**

1. **IDENTIFY**: Identify the biological components or factors
2. **EXAMINE**: Examine how components interact or function
3. **IMPACT**: Analyse the impact or consequence
4. **SIGNIFICANCE**: State the biological significance

**Marks Allocation:**
- Identify: {identify_marks} marks
- Examine: {examine_marks} marks
- Impact: {impact_marks} marks
- Significance: {significance_marks} marks""",
            components=[
                StructureComponent(
                    "Identify",
                    "Identify components",
                    True,
                    0.2,
                    ["The key factors are...", "Components include..."],
                ),
                StructureComponent(
                    "Examine",
                    "Examine interaction",
                    True,
                    0.35,
                    ["These interact by...", "The function is..."],
                ),
                StructureComponent(
                    "Impact",
                    "Analyse impact",
                    True,
                    0.3,
                    ["This results in...", "The effect is..."],
                ),
                StructureComponent(
                    "Significance",
                    "State significance",
                    True,
                    0.15,
                    ["This is significant because...", "Biologically, this means..."],
                ),
            ],
            word_guidance="Use biological terminology. Connect structure to function.",
            example_introductions=[
                "The structure of the cell membrane allows...",
                "Analysing the enzyme's active site...",
                "The data shows that...",
            ],
        )

        return scaffolds

    def _create_business_scaffolds(self) -> dict[str, SubjectScaffold]:
        """Business Studies scaffolds (concept-link-example-judgement)."""
        scaffolds = {}

        scaffolds["analyse"] = SubjectScaffold(
            subject="business",
            command_term="analyse",
            template="""**BUSINESS ANALYSIS STRUCTURE**

1. **CONCEPT**: Identify the business concept or factor
2. **LINK**: Explain the relationship or impact
3. **EXAMPLE**: Provide business example or case study evidence
4. **JUDGEMENT**: State the implications for the business

**Marks Allocation:**
- Concept: {concept_marks} marks
- Link: {link_marks_business} marks
- Example: {example_marks} marks
- Judgement: {judgement_marks} marks""",
            components=[
                StructureComponent(
                    "Concept",
                    "Identify business concept",
                    True,
                    0.2,
                    ["The concept of...", "Key factors include..."],
                ),
                StructureComponent(
                    "Link",
                    "Explain relationship/impact",
                    True,
                    0.35,
                    ["This impacts by...", "The relationship shows..."],
                ),
                StructureComponent(
                    "Example",
                    "Give example/evidence",
                    True,
                    0.25,
                    ["For example...", "Case study evidence shows..."],
                ),
                StructureComponent(
                    "Judgement",
                    "State implications",
                    True,
                    0.2,
                    ["Therefore, the business should...", "This means..."],
                ),
            ],
            word_guidance="Use business terminology. Link theory to real business situations.",
            example_introductions=[
                "The impact of globalisation on...",
                "This marketing strategy affects...",
                "An analysis of the financial data shows...",
            ],
        )

        scaffolds["evaluate"] = SubjectScaffold(
            subject="business",
            command_term="evaluate",
            template="""**BUSINESS EVALUATION STRUCTURE**

1. **CRITERIA**: State the criteria for evaluation
2. **EVIDENCE**: Provide evidence for each criterion
3. **WEIGH**: Weigh up strengths and weaknesses
4. **CONCLUSION**: Give a reasoned recommendation

**Marks Allocation:**
- Criteria: {criteria_marks} marks
- Evidence: {evidence_marks_legal} marks
- Weigh: {weigh_marks} marks
- Conclusion: {conclusion_marks} marks""",
            components=[
                StructureComponent(
                    "Criteria",
                    "State evaluation criteria",
                    True,
                    0.15,
                    ["Criteria for evaluation include...", "To evaluate, we consider..."],
                ),
                StructureComponent(
                    "Evidence",
                    "Provide evidence",
                    True,
                    0.35,
                    ["Evidence shows...", "Data indicates..."],
                ),
                StructureComponent(
                    "Weigh",
                    "Weigh up pros and cons",
                    True,
                    0.3,
                    ["Strengths include...", "However, weaknesses are..."],
                ),
                StructureComponent(
                    "Conclusion",
                    "Give recommendation",
                    True,
                    0.2,
                    ["Therefore, I recommend...", "The best option is..."],
                ),
            ],
            word_guidance="Use business case studies and data. Weigh options clearly.",
            example_introductions=[
                "To evaluate this strategy, we consider...",
                "The strengths of this approach are...",
                "However, there are limitations...",
            ],
        )

        return scaffolds

    def _create_legal_scaffolds(self) -> dict[str, SubjectScaffold]:
        """Legal Studies scaffolds (issue-law-application-conclusion)."""
        scaffolds = {}

        scaffolds["analyse"] = SubjectScaffold(
            subject="legal",
            command_term="analyse",
            template="""**LEGAL ANALYSIS STRUCTURE (ILAC)**

1. **ISSUE**: Identify the legal issue(s) to be determined
2. **LAW**: State the relevant legal principles/legislation
3. **APPLICATION**: Apply the law to the facts
4. **CONCLUSION**: Draw a reasoned conclusion

**Marks Allocation:**
- Issue: {issue_marks} marks
- Law: {law_marks} marks
- Application: {application_marks_legal} marks
- Conclusion: {conclusion_marks} marks""",
            components=[
                StructureComponent(
                    "Issue",
                    "Identify the legal issue",
                    True,
                    0.15,
                    ["The legal issue is...", "The question to determine is..."],
                ),
                StructureComponent(
                    "Law",
                    "State relevant law",
                    True,
                    0.25,
                    ["The relevant law is...", "Section s of the...Act states..."],
                ),
                StructureComponent(
                    "Application",
                    "Apply law to facts",
                    True,
                    0.4,
                    ["In this case...", "Applying this to the facts..."],
                ),
                StructureComponent(
                    "Conclusion",
                    "Draw conclusion",
                    True,
                    0.2,
                    ["Therefore...", "It can be concluded that..."],
                ),
            ],
            word_guidance="Use legal terminology correctly. Apply law precisely to facts.",
            example_introductions=[
                "The issue here is whether...",
                "The legal principle established in...states...",
                "Applying this to the facts of the case...",
            ],
        )

        scaffolds["discuss"] = SubjectScaffold(
            subject="legal",
            command_term="discuss",
            template="""**LEGAL DISCUSSION STRUCTURE**

1. **ISSUE**: State the legal issue or proposition
2. **ARGUMENTS**: Present arguments for and against
3. **ANALYSIS**: Analyse legal implications and effectiveness
4. **EVALUATION**: Evaluate the law/reform and give opinion

**Marks Allocation:**
- Issue: {issue_marks} marks
- Arguments: {arguments_marks} marks
- Analysis: {analysis_marks} marks
- Evaluation: {evaluation_marks} marks""",
            components=[
                StructureComponent(
                    "Issue",
                    "State the issue",
                    True,
                    0.1,
                    ["The issue is whether...", "This essay discusses..."],
                ),
                StructureComponent(
                    "Arguments",
                    "Present arguments",
                    True,
                    0.35,
                    ["On one hand...", "However, the opposing view is..."],
                ),
                StructureComponent(
                    "Analysis",
                    "Analyse implications",
                    True,
                    0.3,
                    ["This means that...", "The legal implication is..."],
                ),
                StructureComponent(
                    "Evaluation",
                    "Give opinion",
                    True,
                    0.25,
                    ["In my view...", "The law should be reformed because..."],
                ),
            ],
            word_guidance="Use legal cases and legislation. Present balanced arguments.",
            example_introductions=[
                "The effectiveness of this law can be assessed by...",
                "Arguments for reform include...",
                "The current law fails to address...",
            ],
        )

        return scaffolds

    def _create_english_scaffolds(self) -> dict[str, SubjectScaffold]:
        """English Advanced scaffolds (thesis-evidence-analysis-link)."""
        scaffolds = {}

        scaffolds["analyse"] = SubjectScaffold(
            subject="english_adv",
            command_term="analyse",
            template="""**ENGLISH ANALYSIS STRUCTURE (TEAL)**

1. **THESIS**: State your interpretive thesis
2. **EVIDENCE**: Provide textual evidence (quote)
3. **ANALYSIS**: Explain how evidence supports thesis
4. **LINK**: Link back to the question and thesis

**Marks Allocation:**
- Thesis: {thesis_marks} marks
- Evidence: {evidence_marks} marks
- Analysis: {analysis_marks} marks
- Link: {link_marks} marks""",
            components=[
                StructureComponent(
                    "Thesis",
                    "State interpretive claim",
                    True,
                    0.15,
                    ["The composer positions us to believe that...", "This text explores..."],
                ),
                StructureComponent(
                    "Evidence",
                    "Provide textual evidence",
                    True,
                    0.25,
                    ["As seen in the line '...'", "The visual technique of..."],
                ),
                StructureComponent(
                    "Analysis",
                    "Explain evidence's effect",
                    True,
                    0.4,
                    ["This shows that...", "The composer uses this to..."],
                ),
                StructureComponent(
                    "Link",
                    "Connect to question",
                    True,
                    0.2,
                    ["Therefore, the text...", "This reveals..."],
                ),
            ],
            word_guidance="Use subject terminology. Connect techniques to meaning and context.",
            example_introductions=[
                "The composer's use of metaphor...",
                "Through the character of..., the composer...",
                "The imagery of '...' conveys...",
            ],
        )

        scaffolds["evaluate"] = SubjectScaffold(
            subject="english_adv",
            command_term="evaluate",
            template="""**ENGLISH EVALUATION STRUCTURE**

1. **CRITERIA**: State criteria for evaluation
2. **EVIDENCE**: Provide textual evidence
3. **ANALYSIS**: Evaluate effectiveness of techniques
4. **JUDGEMENT**: Give overall evaluation

**Marks Allocation:**
- Criteria: {criteria_marks} marks
- Evidence: {evidence_marks} marks
- Analysis: {analysis_marks} marks
- Judgement: {judgement_marks} marks""",
            components=[
                StructureComponent(
                    "Criteria",
                    "State evaluation criteria",
                    True,
                    0.15,
                    ["To evaluate, we consider...", "Effective techniques are those that..."],
                ),
                StructureComponent(
                    "Evidence",
                    "Provide textual examples",
                    True,
                    0.25,
                    ["For example, the use of...", "The technique of..."],
                ),
                StructureComponent(
                    "Analysis",
                    "Evaluate effectiveness",
                    True,
                    0.35,
                    [
                        "This technique is effective because...",
                        "However, this is less successful...",
                    ],
                ),
                StructureComponent(
                    "Judgement",
                    "Give overall evaluation",
                    True,
                    0.25,
                    [
                        "Overall, the text successfully...",
                        "The composer's purpose is achieved through...",
                    ],
                ),
            ],
            word_guidance="Evaluate how techniques convey meaning. Support with textual evidence.",
            example_introductions=[
                "The composer's representation of...",
                "This is evident through the use of...",
                "The effectiveness of this is seen in...",
            ],
        )

        return scaffolds

    def _create_chemistry_scaffolds(self) -> dict[str, SubjectScaffold]:
        """Chemistry-specific scaffolds."""
        return self._create_physics_scaffolds()  # Use similar structure

    def _create_physics_scaffolds(self) -> dict[str, SubjectScaffold]:
        """Physics-specific scaffolds."""
        scaffolds = {}

        scaffolds["explain"] = SubjectScaffold(
            subject="physics",
            command_term="explain",
            template="""**PHYSICS EXPLANATION STRUCTURE**

1. **PRINCIPLE**: State the physical principle or law
2. **FORMULA**: State the relevant formula with symbols
3. **APPLICATION**: Apply to the specific situation
4. **RESULT**: State the result with units

**Marks Allocation:**
- Principle: {principle_marks} marks
- Formula: {formula_marks} marks
- Application: {application_marks} marks
- Result: {result_marks} marks""",
            components=[
                StructureComponent(
                    "Principle",
                    "State the principle",
                    True,
                    0.2,
                    ["According to...", "The principle of..."],
                ),
                StructureComponent(
                    "Formula",
                    "State the formula",
                    True,
                    0.2,
                    ["Using the formula...", "Where F =..."],
                ),
                StructureComponent(
                    "Application",
                    "Apply to situation",
                    True,
                    0.4,
                    ["Substituting...", "Therefore..."],
                ),
                StructureComponent(
                    "Result", "State result with units", True, 0.2, ["The answer is...", "Result:"]
                ),
            ],
            word_guidance="Use correct physics terminology and units. Show logical reasoning.",
            example_introductions=[
                "Using Newton's second law...",
                "The force acting is given by...",
                "Applying the conservation of energy...",
            ],
        )

        return scaffolds

    def _create_general_scaffolds(self) -> dict[str, SubjectScaffold]:
        """General/across-subject scaffolds."""
        scaffolds = {}

        scaffolds["describe"] = SubjectScaffold(
            subject="general",
            command_term="describe",
            template="""**GENERAL DESCRIPTION STRUCTURE**

1. **OVERVIEW**: Give a general description
2. **DETAILS**: Provide specific characteristics
3. **EXAMPLES**: Give examples or illustrations
4. **SUMMARY**: Brief summary of key points

**Marks Allocation:**
- Overview: {overview_marks} marks
- Details: {details_marks} marks
- Examples: {example_marks} marks
- Summary: {summary_marks} marks""",
            components=[
                StructureComponent(
                    "Overview",
                    "General description",
                    True,
                    0.2,
                    ["The main features are...", "This involves..."],
                ),
                StructureComponent(
                    "Details",
                    "Specific characteristics",
                    True,
                    0.4,
                    ["Specifically...", "Key aspects include..."],
                ),
                StructureComponent(
                    "Examples", "Give examples", True, 0.25, ["For example...", "Such as..."]
                ),
                StructureComponent(
                    "Summary",
                    "Brief summary",
                    True,
                    0.15,
                    ["In summary...", "The key points are..."],
                ),
            ],
            word_guidance="Be clear and organised. Provide appropriate detail.",
            example_introductions=[
                "The main characteristics are...",
                "This includes features such as...",
                "A clear example is...",
            ],
        )

        scaffolds["compare"] = SubjectScaffold(
            subject="general",
            command_term="compare",
            template="""**COMPARISON STRUCTURE**

1. **ITEM 1**: Describe first item
2. **ITEM 2**: Describe second item
3. **SIMILARITIES**: Show how they are similar
4. **DIFFERENCES**: Show how they differ
5. **SUMMARY**: Brief comparison conclusion

**Marks Allocation:**
- Items 1 & 2: {concept_marks} marks
- Similarities: {similarities_marks} marks
- Differences: {differences_marks} marks
- Summary: {summary_marks} marks""",
            components=[
                StructureComponent(
                    "Item 1",
                    "Describe first item",
                    True,
                    0.15,
                    ["The first concept/item is...", "In the case of..."],
                ),
                StructureComponent(
                    "Item 2",
                    "Describe second item",
                    True,
                    0.15,
                    ["The second concept/item is...", "Similarly..."],
                ),
                StructureComponent(
                    "Similarities",
                    "Show similarities",
                    True,
                    0.25,
                    ["Both share...", "Similarly, both..."],
                ),
                StructureComponent(
                    "Differences",
                    "Show differences",
                    True,
                    0.3,
                    ["However, unlike...", "Whereas..."],
                ),
                StructureComponent(
                    "Summary",
                    "Brief conclusion",
                    True,
                    0.15,
                    ["In comparison, the main difference is...", "Therefore, they differ in..."],
                ),
            ],
            word_guidance="Use comparative language. Address both items equally.",
            example_introductions=[
                "Both X and Y share the characteristic of...",
                "However, X differs from Y in that...",
                "While X shows..., Y demonstrates...",
            ],
        )

        return scaffolds

    def get_command_term_info(self, term: str, subject: str = "general") -> CommandTermInfo:
        """Get information about a command term."""
        try:
            command_term = CommandTerm(term.lower())
        except ValueError:
            logger.warning(f"Unknown command term: {term}")
            command_term = CommandTerm.EXPLAIN

        definition_data = COMMAND_TERM_DEFINITIONS.get(
            command_term, COMMAND_TERM_DEFINITIONS[CommandTerm.EXPLAIN]
        )
        structure = DEFAULT_STRUCTURES.get(command_term, DEFAULT_STRUCTURES[CommandTerm.EXPLAIN])

        return CommandTermInfo(
            term=term,
            definition=definition_data["definition"],
            general_structure=structure,
            marking_signals=definition_data["marking_signals"],
            common_errors=definition_data["common_errors"],
            tips=definition_data["tips"],
        )

    def get_scaffold(
        self,
        subject: str,
        command_term: str,
        marks: int = 10,
        question: str = "",
    ) -> SubjectScaffold | None:
        """Get a subject-specific scaffold for a command term."""
        subject_lower = subject.lower().replace(" ", "_").replace("-", "_")

        # Try exact match first
        if subject_lower in self._subject_scaffolds:
            subject_scaffolds = self._subject_scaffolds[subject_lower]
            if command_term.lower() in subject_scaffolds:
                scaffold = subject_scaffolds[command_term.lower()]
                return self._format_scaffold(scaffold, marks, question)

        # Try without underscores
        subject_simple = subject_lower.replace("_", " ")
        if subject_simple in self._subject_scaffolds:
            subject_scaffolds = self._subject_scaffolds[subject_simple]
            if command_term.lower() in subject_scaffolds:
                scaffold = subject_scaffolds[command_term.lower()]
                return self._format_scaffold(scaffold, marks, question)

        # Fall back to general
        if command_term.lower() in self._subject_scaffolds.get("general", {}):
            scaffold = self._subject_scaffolds["general"][command_term.lower()]
            return self._format_scaffold(scaffold, marks, question)

        logger.warning(f"No scaffold found for {subject}/{command_term}")
        return None

    def _format_scaffold(
        self,
        scaffold: SubjectScaffold,
        marks: int,
        question: str,
    ) -> SubjectScaffold:
        """Format a scaffold with marks and question."""
        import re

        template = scaffold.template

        def replace_expr(match):
            expr = match.group(1)
            try:
                result = eval(expr, {"marks": marks})
                return str(int(result))
            except:
                return match.group(0)

        template = re.sub(r"\{marks \* ([^}]+):\.0f\}", replace_expr, template)
        template = re.sub(r"\{marks \* ([^}]+)\}", replace_expr, template)

        marks_vars = {
            "method_marks": int(marks * 0.2),
            "working_marks": int(marks * 0.5),
            "answer_marks": int(marks * 0.3),
            "concept_marks": int(marks * 0.2),
            "reason_marks": int(marks * 0.3),
            "application_marks": int(marks * 0.3),
            "example_marks": int(marks * 0.2),
            "identify_marks": int(marks * 0.25),
            "relationships_marks": int(marks * 0.35),
            "properties_marks": int(marks * 0.25),
            "conclusion_marks": int(marks * 0.15),
            "define_marks": int(marks * 0.15),
            "process_marks": int(marks * 0.35),
            "link_marks_bio": int(marks * 0.3),
            "significance_marks": int(marks * 0.15),
            "examine_marks": int(marks * 0.35),
            "impact_marks": int(marks * 0.3),
            "link_marks_business": int(marks * 0.35),
            "judgement_marks": int(marks * 0.2),
            "criteria_marks": int(marks * 0.15),
            "evidence_marks_legal": int(marks * 0.35),
            "weigh_marks": int(marks * 0.3),
            "evaluation_marks": int(marks * 0.25),
            "issue_marks": int(marks * 0.15),
            "law_marks": int(marks * 0.25),
            "application_marks_legal": int(marks * 0.4),
            "arguments_marks": int(marks * 0.35),
            "analysis_marks": int(marks * 0.3),
            "thesis_marks": int(marks * 0.15),
            "evidence_marks": int(marks * 0.25),
            "link_marks": int(marks * 0.2),
            "principle_marks": int(marks * 0.2),
            "formula_marks": int(marks * 0.2),
            "result_marks": int(marks * 0.2),
            "overview_marks": int(marks * 0.2),
            "details_marks": int(marks * 0.4),
            "summary_marks": int(marks * 0.15),
            "similarities_marks": int(marks * 0.25),
            "differences_marks": int(marks * 0.3),
        }

        template = template.format(**marks_vars)

        if question:
            template = f"**Question:** {question}\n\n" + template

        return SubjectScaffold(
            subject=scaffold.subject,
            command_term=scaffold.command_term,
            template=template,
            components=scaffold.components,
            word_guidance=scaffold.word_guidance,
            example_introductions=scaffold.example_introductions,
        )

    def evaluate_structure(
        self,
        answer: str,
        command_term: str,
        subject: str,
        scaffold: SubjectScaffold | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate if an answer follows the expected structure.

        Returns a report on structural completeness and advice.
        """
        if scaffold is None:
            scaffold = self.get_scaffold(subject, command_term)
            if scaffold is None:
                return {
                    "structural_score": 1.0,
                    "components_found": [],
                    "missing_components": [],
                    "complete": True,
                    "scaffold": None,
                    "advice": ["No scaffold available for this subject/command term"],
                }

        answer_lower = answer.lower()
        components_found = {}
        missing_components = []

        for component in scaffold.components:
            if component.required:
                # Check for component indicators
                found = False
                for phrase in component.example_phrases:
                    if phrase.lower() in answer_lower:
                        found = True
                        break

                # Also check for key structural markers
                component_keywords = component.name.lower().split()
                for keyword in component_keywords:
                    if keyword in answer_lower:
                        found = True
                        break

                if found:
                    components_found[component.name] = True
                else:
                    missing_components.append(component.name)

        # Calculate structural score
        required_count = sum(1 for c in scaffold.components if c.required)
        found_count = len(components_found)
        structural_score = found_count / required_count if required_count > 0 else 1.0

        # Generate feedback
        feedback = {
            "structural_score": round(structural_score, 2),
            "components_found": list(components_found.keys()),
            "missing_components": missing_components,
            "complete": len(missing_components) == 0,
            "scaffold": scaffold.template if scaffold else None,
            "advice": [],
        }

        # Add specific advice for missing components
        if missing_components:
            feedback["advice"].append(
                f"⚠️ Missing required structure: {', '.join(missing_components)}"
            )
            for component in scaffold.components:
                if component.name in missing_components:
                    feedback["advice"].append(
                        f"→ Include {component.name}: {component.description}"
                    )
                    if component.example_phrases:
                        feedback["advice"].append(f"   Try: {component.example_phrases[0]}")
        else:
            feedback["advice"].append("✅ Response follows expected structure")

        # Check word length
        word_count = len(answer.split())
        expected_words = self._estimate_word_count(scaffold, marks=10)
        if word_count < expected_words * 0.5:
            feedback["advice"].append(
                f"⚠️ Response may be too brief ({word_count} words). "
                f"Target around {expected_words} words for full marks."
            )

        return feedback

    def _estimate_word_count(self, scaffold: SubjectScaffold | None, marks: int) -> int:
        """Estimate appropriate word count based on marks and scaffold."""
        if scaffold is None:
            return marks * 30
        base_words_per_mark = 15
        component_count = len([c for c in scaffold.components if c.required])
        return int(marks * base_words_per_mark / component_count * 2)


def get_command_term_info(term: str, subject: str = "general") -> CommandTermInfo:
    """Convenience function to get command term info."""
    engine = CommandTermEngine()
    return engine.get_command_term_info(term, subject)


def generate_scaffold(
    subject: str,
    command_term: str,
    question: str = "",
    marks: int = 10,
) -> str | None:
    """Convenience function to generate a scaffold template."""
    engine = CommandTermEngine()
    scaffold = engine.get_scaffold(subject, command_term, marks, question)
    return scaffold.template if scaffold else None


def get_subject_scaffold(
    subject: str, command_term: str, marks: int = 10
) -> SubjectScaffold | None:
    """Convenience function to get a full scaffold object."""
    engine = CommandTermEngine()
    return engine.get_scaffold(subject, command_term, marks)


def evaluate_answer_structure(
    answer: str,
    command_term: str,
    subject: str,
) -> dict[str, Any]:
    """Convenience function to evaluate structural completeness."""
    engine = CommandTermEngine()
    scaffold = engine.get_scaffold(subject, command_term)
    return engine.evaluate_structure(answer, command_term, subject, scaffold)


# Mapping of common command term variations
COMMAND_TERM_SYNONYMS = {
    "analyze": "analyse",
    "critically evaluate": "evaluate",
    "critically analyse": "analyse",
    "examine in detail": "examine",
    "account for": "explain",
    "give an account of": "describe",
    "what is meant by": "define",
    "elaborate": "explain",
    "solve": "calculate",
    "work out": "calculate",
    "determine": "calculate",
    "find": "calculate",
    "verify": "calculate",
    "prove": "explain",
}


def resolve_command_term(term: str) -> str:
    """Resolve command term variations to standard form."""
    term_lower = term.lower().strip()
    return COMMAND_TERM_SYNONYMS.get(term_lower, term_lower)
