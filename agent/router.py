from __future__ import annotations

import re

INTENTS = ("rag", "quiz", "summary", "evaluate")


def detect_intent(query: str) -> str:
    """Route a learning request to the capability that best matches it."""
    if not query or not query.strip():
        return "invalid"
    text = query.lower().strip()

    if re.search(r"\b(mcq|mcqs|quiz|multiple choice|objective questions?)\b", text):
        return "quiz"
    if re.search(r"\b(summarize|summarise|summary|overview|key points)\b", text):
        return "summary"
    if re.search(r"\b(my answer is|evaluate my answer|check my answer|is my answer correct|am i correct|evaluate)\b", text):
        return "evaluate"
    return "rag"


def route_query(query: str) -> str:
    return detect_intent(query)
