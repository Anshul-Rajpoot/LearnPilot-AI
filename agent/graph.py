from __future__ import annotations

from typing import Any, Callable

from agent.router import detect_intent
from agent.state import AgentState
from agent.tools import TOOL_REGISTRY

TOOL_LABELS = {
    "rag": "RAG Retrieval",
    "quiz": "Quiz Generator",
    "summary": "Summarization",
    "evaluate": "Answer Evaluation",
    "invalid": "Input Validation",
}


class LearnPilotAI:
    """Explicit agent workflow: route -> retrieve -> select tool -> generate.

    The orchestration is intentionally transparent and framework-light so each
    step can be inspected, tested and explained during an interview.
    """

    def __init__(self, retriever: Any, generator: Callable[[str], str] | None = None):
        self.retriever = retriever
        self.generator = generator

    def prepare(self, query: str, top_k: int = 5, prefer_semantic: bool = True) -> AgentState:
        """Run the agent through routing, retrieval and tool selection."""
        state = AgentState(user_query=query)
        state.intent = detect_intent(query)

        if state.intent == "invalid":
            state.error = "Please enter a question or learning request."
            state.tool_used = TOOL_LABELS["invalid"]
            return state

        state.tool_used = TOOL_LABELS[state.intent]
        retrieved = self.retriever.search(query, top_k=top_k, prefer_semantic=prefer_semantic)
        state.retrieved_context = self.retriever.records(retrieved)

        tool = TOOL_REGISTRY[state.intent]
        result = tool(query, state.retrieved_context)
        state.tool_result = {
            "name": result.name,
            "prompt": result.prompt,
            "metadata": result.metadata,
        }
        return state

    def run(self, query: str, top_k: int = 5, prefer_semantic: bool = True) -> AgentState:
        """Run the complete workflow including LLM generation."""
        state = self.prepare(query, top_k=top_k, prefer_semantic=prefer_semantic)
        if state.error:
            return state
        if self.generator is None:
            raise RuntimeError("LearnPilot AI.run requires a generator. Use prepare() for UI streaming.")
        state.final_response = self.generator(state.tool_result["prompt"])
        return state
