from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from retrieval.prompts import build_prompt


@dataclass(frozen=True)
class ToolResult:
    name: str
    prompt: str
    metadata: dict[str, Any]


def rag_tool(query: str, context: list[dict[str, Any]]) -> ToolResult:
    return ToolResult("rag", build_prompt("rag", query, context), {"context_count": len(context)})


def quiz_tool(query: str, context: list[dict[str, Any]]) -> ToolResult:
    return ToolResult("quiz", build_prompt("quiz", query, context), {"context_count": len(context)})


def summary_tool(query: str, context: list[dict[str, Any]]) -> ToolResult:
    return ToolResult("summary", build_prompt("summary", query, context), {"context_count": len(context)})


def evaluation_tool(query: str, context: list[dict[str, Any]]) -> ToolResult:
    return ToolResult("evaluate", build_prompt("evaluate", query, context), {"context_count": len(context)})


TOOL_REGISTRY: dict[str, Callable[[str, list[dict[str, Any]]], ToolResult]] = {
    "rag": rag_tool,
    "quiz": quiz_tool,
    "summary": summary_tool,
    "evaluate": evaluation_tool,
}
