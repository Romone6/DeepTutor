"""
RAG Prompt Injection Utility

Injects knowledge base snippets into prompts to ensure all tutoring is
syllabus/exam grounded by default (no free-floating answers).

Usage:
    from extensions.knowledge.rag_injector import inject_rag_context, inject_rag_into_prompts

    # Inject RAG context before LLM call
    system_prompt, user_prompt = inject_rag_context(
        kb_name="hsc_math_adv",
        system_prompt=original_system,
        user_prompt=original_user,
        topic="calculus derivatives",
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from extensions.knowledge import retrieve, Snippet, RetrievalResult

if TYPE_CHECKING:
    from extensions.utils.evidence import EvidenceMap


@dataclass
class RAGContext:
    """Container for RAG-enhanced prompts."""

    system_prompt: str
    user_prompt: str
    retrieval_result: RetrievalResult
    snippets_injected: int
    kb_empty: bool
    evidence_map: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "snippets_injected": self.snippets_injected,
            "kb_empty": self.kb_empty,
            "total_chunks": self.retrieval_result.total_chunks,
            "error": self.retrieval_result.error,
            "evidence_map": self.evidence_map,
        }


KB_EMPTY_WARNING = """
⚠️ KNOWLEDGE BASE EMPTY OR NOT FOUND
The knowledge base '{kb_name}' does not contain relevant information for this request.

IMPORTANT: If you must answer, you must:
1. Clearly state that the knowledge base has no coverage for this topic
2. Ask the user to provide the missing document or specify the topic more precisely
3. Do NOT provide a free-floating answer without source grounding

Example response:
"I couldn't find information about '{topic}' in the knowledge base. 
To help you better, please either:
- Provide a document or URL containing the relevant material, or
- Ask about a different topic that is covered in the syllabus."

Remember: DeepTutor should always be syllabus/exam grounded. Never make up information.
"""

SYSTEM_PROMPT_SUFFIX = """

## KNOWLEDGE BASE CONTEXT
The following information is retrieved from the syllabus and study materials.
Use this context to ensure your answer is accurate and grounded in the course content.

---

{snippets}

---

IMPORTANT: Your response must be based on the above context. 
Do not provide information that contradicts the retrieved material.
If the context is insufficient, acknowledge this limitation.
"""


def format_snippets(snippets: list[Snippet], max_snippets: int = 5, max_chars: int = 3000) -> str:
    """Format retrieved snippets into a compact section for prompts."""
    if not snippets:
        return ""

    selected = snippets[:max_snippets]
    parts = []

    for i, snippet in enumerate(selected, 1):
        topic = snippet.metadata.get("topic", "general")
        doc_id = snippet.metadata.get("doc_id", "unknown")
        content = snippet.text

        parts.append(f"### Source {i} [{topic}] (ID: {doc_id})")
        parts.append(content)
        parts.append("")

        if sum(len(p) for p in parts) > max_chars:
            break

    return "\n".join(parts)


async def inject_rag_context(
    kb_name: str,
    system_prompt: str,
    user_prompt: str,
    topic: Optional[str] = None,
    max_snippets: int = 5,
    max_chars: int = 3000,
    top_k: int = 8,
    include_evidence_map: bool = True,
) -> RAGContext:
    """
    Inject RAG context into prompts.

    Args:
        kb_name: Knowledge base name (e.g., 'hsc_math_adv')
        system_prompt: Original system prompt
        user_prompt: Original user prompt (used for retrieval query)
        topic: Optional topic hint for better retrieval
        max_snippets: Maximum number of snippets to include
        max_chars: Maximum character length for snippets section
        top_k: Number of results to retrieve
        include_evidence_map: Whether to include evidence_map in response

    Returns:
        RAGContext with enhanced prompts and evidence_map
    """
    from extensions.utils.evidence import create_evidence_map

    query = topic or user_prompt

    retrieval_result = await retrieve(kb_name, query, top_k=top_k)

    if retrieval_result.error or not retrieval_result.snippets:
        kb_empty = True
        snippets_text = KB_EMPTY_WARNING.format(kb_name=kb_name, topic=topic or query[:50])
        evidence_map_dict = None
    else:
        kb_empty = False
        snippets_text = format_snippets(
            retrieval_result.snippets,
            max_snippets=max_snippets,
            max_chars=max_chars,
        )

        if include_evidence_map:
            evidence_map = create_evidence_map(
                snippets=retrieval_result.snippets,
                query=query,
                kb_name=kb_name,
                max_snippets=max_snippets,
            )
            evidence_map_dict = evidence_map.to_dict()
        else:
            evidence_map_dict = None

    enhanced_system = system_prompt + SYSTEM_PROMPT_SUFFIX.format(snippets=snippets_text)

    return RAGContext(
        system_prompt=enhanced_system,
        user_prompt=user_prompt,
        retrieval_result=retrieval_result,
        snippets_injected=len(retrieval_result.snippets) if not kb_empty else 0,
        kb_empty=kb_empty,
        evidence_map=evidence_map_dict,
    )


def inject_rag_into_prompts_sync(
    kb_name: str,
    system_prompt: str,
    user_prompt: str,
    topic: Optional[str] = None,
    include_evidence_map: bool = True,
) -> RAGContext:
    """
    Synchronous wrapper for inject_rag_context.

    Note: This will create a new event loop if called from a sync context.
    For async code, use inject_rag_context directly.
    """
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import warnings

        warnings.warn(
            "Cannot inject RAG context from within a running async loop. "
            "Use inject_rag_context() directly in async code.",
            RuntimeWarning,
        )
        return RAGContext(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            retrieval_result=RetrievalResult(
                query=user_prompt,
                snippets=[],
                kb_name=kb_name,
                error="Cannot call async from sync context",
            ),
            snippets_injected=0,
            kb_empty=True,
            evidence_map=None,
        )

    return asyncio.run(
        inject_rag_context(
            kb_name=kb_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            topic=topic,
            include_evidence_map=include_evidence_map,
        )
    )


class RAGPromptMixin:
    """
    Mixin class for agents that want RAG injection capability.

    Usage:
        class MyAgent(RAGPromptMixin, BaseAgent):
            def __init__(self, *args, kb_name="hsc_math_adv", **kwargs):
                super().__init__(*args, **kwargs)
                self.kb_name = kb_name

            async def my_method(self, user_message: str):
                # Get enhanced prompts with RAG
                context = await self.inject_rag(user_message)
                response = await self.call_llm(
                    system_prompt=context.system_prompt,
                    user_prompt=context.user_prompt,
                )
                return response
    """

    kb_name: str = "hsc_math_adv"
    rag_enabled: bool = True
    rag_max_snippets: int = 5
    rag_top_k: int = 8

    async def inject_rag(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        topic: Optional[str] = None,
        include_evidence_map: bool = True,
    ) -> RAGContext:
        """
        Inject RAG context into prompts.

        Args:
            user_prompt: User message
            system_prompt: System prompt (uses self.prompts.get("system") if not provided)
            topic: Topic hint for retrieval
            include_evidence_map: Whether to include evidence_map in response

        Returns:
            RAGContext with enhanced prompts and evidence_map
        """
        if not getattr(self, "rag_enabled", True):
            return RAGContext(
                system_prompt=system_prompt or "",
                user_prompt=user_prompt,
                retrieval_result=RetrievalResult(
                    query=user_prompt,
                    snippets=[],
                    kb_name=getattr(self, "kb_name", "unknown"),
                ),
                snippets_injected=0,
                kb_empty=False,
                evidence_map=None,
            )

        kb = getattr(self, "kb_name", "hsc_math_adv")
        sys_prompt = system_prompt or (self.prompts.get("system") if self.prompts else "")

        return await inject_rag_context(
            kb_name=kb,
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
            topic=topic or user_prompt,
            max_snippets=getattr(self, "rag_max_snippets", 5),
            top_k=getattr(self, "rag_top_k", 8),
            include_evidence_map=include_evidence_map,
        )


async def ensure_rag_grounded(
    kb_name: str,
    query: str,
    min_snippets: int = 1,
) -> tuple[bool, list[Snippet]]:
    """
    Check if RAG retrieval returns sufficient results.

    Args:
        kb_name: Knowledge base name
        query: Search query
        min_snippets: Minimum snippets required

    Returns:
        Tuple of (sufficient, snippets)
    """
    result = await retrieve(kb_name, query, top_k=10)

    if result.error or not result.snippets:
        return False, []

    return len(result.snippets) >= min_snippets, result.snippets
