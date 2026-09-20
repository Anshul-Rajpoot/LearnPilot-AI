from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    user_query: str
    intent: str = "unknown"
    retrieved_context: list[dict[str, Any]] = field(default_factory=list)
    tool_used: str | None = None
    tool_result: dict[str, Any] = field(default_factory=dict)
    final_response: str = ""
    error: str | None = None
