"""
RAG Integration Examples

Examples showing how to inject RAG context into each DeepTutor module.

Contents:
1. Solve Module Integration
2. Guide Module Integration
3. Question Module Integration
4. Research Module Integration
5. Co-Writer Module Integration
6. Utility helper function
"""

from __future__ import annotations

from pathlib import Path
import sys

_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


# =============================================================================
# 1. SOLVE MODULE INTEGRATION
# =============================================================================
#
# Example: Integrating RAG into the Solve Agent
#
# BEFORE:
#     async def process(self, question, current_step, ...):
#         response = await self.call_llm(
#             system_prompt=self.prompts.get("system"),
#             user_prompt=f"Question: {question}\nStep: {current_step.step_target}",
#         )
#         return {"response": response}
#
# AFTER:
#     from extensions.knowledge import RAGBaseAgent
#
#     class SolveAgent(RAGBaseAgent):
#         async def process(self, question, current_step, ...):
#             # Get RAG-enhanced prompts
#             rag_context = await self.get_rag_context(
#                 user_prompt=f"Question: {question}\nStep: {current_step.step_target}",
#                 topic=f"{question[:100]} {current_step.step_target[:50]}",
#             )
#
#             if rag_context.kb_empty:
#                 return {
#                     "error": "KB has no coverage",
#                     "instruction": "Please provide relevant study materials.",
#                 }
#
#             response = await self.call_llm(
#                 system_prompt=rag_context.system_prompt,
#                 user_prompt=rag_context.user_prompt,
#             )
#
#             return {
#                 "response": response,
#                 "snippets_used": rag_context.snippets_injected,
#             }
#

# =============================================================================
# 2. GUIDE MODULE INTEGRATION
# =============================================================================
#
# Example: Integrating RAG into the Guide Chat Agent
#
# BEFORE:
#     class ChatAgent(BaseGuideAgent):
#         async def process(self, knowledge, chat_history, user_question):
#             response = await self.call_llm(
#                 system_prompt=self.prompts.get("system"),
#                 user_prompt=f"Knowledge: {knowledge}\nHistory: {chat_history}\nQuestion: {user_question}",
#             )
#             return {"answer": response}
#
# AFTER:
#     from extensions.knowledge import RAGBaseAgent
#
#     class ChatAgent(RAGBaseAgent):
#         async def process(self, knowledge, chat_history, user_question):
#             # Build combined query
#             topic = knowledge.get("knowledge_title", "")
#             query = f"{topic} {user_question}"
#
#             rag_context = await self.get_rag_context(
#                 user_prompt=f"Question: {user_question}\nKnowledge: {knowledge}",
#                 topic=topic,
#             )
#
#             if rag_context.kb_empty:
#                 return {
#                     "answer": rag_context.retrieval_result.error,
#                     "kb_empty": True,
#                 }
#
#             response = await self.call_llm(
#                 system_prompt=rag_context.system_prompt,
#                 user_prompt=rag_context.user_prompt,
#             )
#
#             return {
#                 "answer": response,
#                 "snippets_used": rag_context.snippets_injected,
#             }
#

# =============================================================================
# 3. QUESTION MODULE INTEGRATION
# =============================================================================
#
# Example: Integrating RAG into the Question Generation Agent
#
# BEFORE:
#     class QuestionGenerationAgent:
#         async def generate(self, requirement):
#             response = await self.client.chat.completions.create(
#                 messages=[
#                     {"role": "system", "content": self.system_prompt},
#                     {"role": "user", "content": f"Requirement: {requirement}"},
#                 ]
#             )
#             return self.parse_response(response)
#
# AFTER:
#     class QuestionGenerationAgent:
#         async def generate_with_rag(self, requirement, kb_name):
#             from extensions.knowledge import retrieve, format_snippets
#
#             # Retrieve relevant knowledge
#             result = await retrieve(
#                 kb_name=kb_name,
#                 query=requirement.get("knowledge_point", ""),
#                 top_k=5,
#             )
#
#             if not result.snippets:
#                 return {
#                     "task_rejected": True,
#                     "reason": "Required knowledge is missing from the knowledge base.",
#                 }
#
#             # Format snippets for the prompt
#             snippets_text = format_snippets(result.snippets)
#
#             # Build enhanced system prompt
#             enhanced_prompt = (
#                 f"{self.system_prompt}\n\n"
#                 "## RELEVANT KNOWLEDGE BASE CONTENT\n"
#                 "Use the following information from the syllabus to ensure question accuracy:\n\n"
#                 f"{snippets_text}\n\n"
#                 "Remember: The question must be based on the above knowledge."
#             )
#
#             response = await self.client.chat.completions.create(
#                 messages=[
#                     {"role": "system", "content": enhanced_prompt},
#                     {"role": "user", "content": f"Requirement: {requirement}"},
#                 ]
#             )
#
#             return self.parse_response(response)
#

# =============================================================================
# 4. RESEARCH MODULE INTEGRATION
# =============================================================================
#
# Example: Integrating RAG into the Research Agent
#
# BEFORE:
#     class ResearchAgent(BaseAgent):
#         async def process(self, topic):
#             response = await self.call_llm(
#                 system_prompt=self.prompts.get("system"),
#                 user_prompt=f"Research topic: {topic}",
#             )
#             return {"report": response}
#
# AFTER:
#     from extensions.knowledge import RAGBaseAgent
#
#     class ResearchAgent(RAGBaseAgent):
#         async def process(self, topic):
#             # Get RAG context for grounding
#             rag_context = await self.get_rag_context(
#                 user_prompt=f"Research: {topic}",
#                 topic=topic,
#             )
#
#             if rag_context.kb_empty:
#                 # Fall back to web research only
#                 return await self.research_with_web(topic)
#
#             response = await self.call_llm(
#                 system_prompt=rag_context.system_prompt,
#                 user_prompt=rag_context.user_prompt,
#             )
#
#             return {
#                 "report": response,
#                 "grounded_in_kb": True,
#                 "snippets_used": rag_context.snippets_injected,
#             }
#

# =============================================================================
# 5. CO-WRITER MODULE INTEGRATION
# =============================================================================
#
# Example: Integrating RAG into the Co-Writer Narrator Agent
#
# BEFORE:
#     class NarratorAgent(BaseCoWriterAgent):
#         async def process(self, context, writing_goal):
#             response = await self.call_llm(
#                 system_prompt=self.prompts.get("system"),
#                 user_prompt=f"Context: {context}\nGoal: {writing_goal}",
#             )
#             return {"narrative": response}
#
# AFTER:
#     from extensions.knowledge import RAGBaseAgent
#
#     class NarratorAgent(RAGBaseAgent):
#         async def process(self, context, writing_goal):
#             # Check for relevant syllabus content
#             rag_context = await self.get_rag_context(
#                 user_prompt=f"Writing Goal: {writing_goal}\nContext: {context}",
#                 topic=writing_goal,
#             )
#
#             response = await self.call_llm(
#                 system_prompt=rag_context.system_prompt,
#                 user_prompt=rag_context.user_prompt,
#             )
#
#             return {
#                 "narrative": response,
#                 "syllabus_grounded": not rag_context.kb_empty,
#                 "snippets_used": rag_context.snippets_injected,
#             }
#

# =============================================================================
# UTILITY: Quick integration helper for existing agents
# =============================================================================


async def integrate_rag_into_agent(
    agent,
    user_prompt: str,
    system_prompt: str | None = None,
    topic: str | None = None,
    kb_name: str = "hsc_math_adv",
):
    """
    Helper function to inject RAG into any existing agent.

    Args:
        agent: Agent with call_llm method
        user_prompt: User message
        system_prompt: System prompt (or None to use agent.prompts)
        topic: Topic hint for retrieval
        kb_name: Knowledge base name

    Returns:
        Tuple of (response, metadata)
    """
    from extensions.knowledge import inject_rag_context

    sys_prompt = system_prompt or (agent.prompts.get("system") if agent.prompts else "")

    rag_context = await inject_rag_context(
        kb_name=kb_name,
        system_prompt=sys_prompt,
        user_prompt=user_prompt,
        topic=topic or user_prompt,
    )

    response = await agent.call_llm(
        user_prompt=rag_context.user_prompt,
        system_prompt=rag_context.system_prompt,
    )

    metadata = {
        "snippets_injected": rag_context.snippets_injected,
        "kb_empty": rag_context.kb_empty,
        "total_chunks": rag_context.retrieval_result.total_chunks,
    }

    return response, metadata


__all__ = ["integrate_rag_into_agent"]
