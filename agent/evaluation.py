from __future__ import annotations

from typing import Any, Sequence

from .router import route_query

EVAL_DATASET: list[dict[str, str]] = [
    {"question": "Explain TCP three-way handshake from my course material.", "expected_tool": "rag"},
    {"question": "Give me 10 MCQs on TCP three-way handshake.", "expected_tool": "quiz"},
    {"question": "Summarize this lecture.", "expected_tool": "summary"},
    {"question": "Check my answer: TCP uses UDP because UDP is reliable.", "expected_tool": "evaluate"},
]


def evaluate_tool_selection(samples: Sequence[dict[str, str]] | None = None) -> list[dict[str, Any]]:
    dataset = list(samples or EVAL_DATASET)
    return [
        {
            **sample,
            "selected_tool": route_query(sample["question"]),
            "tool_correct": route_query(sample["question"]) == sample["expected_tool"],
        }
        for sample in dataset
    ]
