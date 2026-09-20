from __future__ import annotations

from typing import Any


def context_text(context: list[dict[str, Any]]) -> str:
    if not context:
        return "No relevant course material was retrieved."
    lines = []
    for i, item in enumerate(context, 1):
        title = item.get("title", "Untitled")
        number = item.get("number", "?")
        start = item.get("start", "?")
        end = item.get("end", "?")
        text = str(item.get("text", "")).strip()
        lines.append(f"[{i}] Video {number} | {title} | {start}s-{end}s\n{text}")
    return "\n\n".join(lines)


def build_prompt(tool: str, query: str, context: list[dict[str, Any]]) -> str:
    source = context_text(context)
    base = f"""You are LearnPilot AI, a course-grounded learning assistant.\n\nCourse material:\n{source}\n\nUser request:\n{query}\n\nRules:\n- Prefer the supplied course material over general knowledge.\n- If the material does not support an answer, say that clearly.\n- Do not invent video numbers or timestamps.\n- Keep the response clear and useful for a student.\n"""

    if tool == "quiz":
        return base + """\nCreate the requested MCQs from the retrieved material. Use four options, mark the correct answer, and give a short explanation for each.\n"""
    if tool == "summary":
        return base + """\nGive a concise summary followed by key concepts. Mention relevant videos/timestamps when supported by the material.\n"""
    if tool == "evaluate":
        return base + """\nEvaluate the student's answer as CORRECT, PARTIALLY CORRECT, or INCORRECT. Explain why using the course context and state the corrected concept when needed.\n"""
    return base + """\nAnswer the question and cite the relevant video number/title and timestamp when the retrieved material supports it.\n"""
