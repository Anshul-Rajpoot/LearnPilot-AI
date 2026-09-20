from __future__ import annotations

import json
from typing import Iterator

import requests

from config import EMBEDDING_MODEL, LLM_MODEL, OLLAMA_BASE_URL, REQUEST_TIMEOUT


class OllamaError(RuntimeError):
    """Raised when the local Ollama service cannot satisfy a request."""


def is_available(timeout: int = 2) -> bool:
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=timeout)
        return response.ok
    except requests.RequestException:
        return False


def embed(texts: list[str]) -> list[list[float]]:
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": EMBEDDING_MODEL, "input": texts},
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        embeddings = payload.get("embeddings")
        if not embeddings:
            raise OllamaError("Ollama returned no embeddings.")
        return embeddings
    except (requests.RequestException, ValueError) as exc:
        raise OllamaError(f"Embedding request failed: {exc}") from exc


def generate(prompt: str) -> str:
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        answer = payload.get("response", "").strip()
        if not answer:
            raise OllamaError("Ollama returned an empty response.")
        return answer
    except (requests.RequestException, ValueError) as exc:
        raise OllamaError(f"Generation request failed: {exc}") from exc


def stream(prompt: str) -> Iterator[str]:
    try:
        with requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": LLM_MODEL, "prompt": prompt, "stream": True},
            stream=True,
            timeout=REQUEST_TIMEOUT,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise OllamaError(f"Invalid JSON from Ollama: {line}") from exc

                if "error" in payload:
                    raise OllamaError(str(payload["error"]))

                token = payload.get("response", "")
                if token:
                    yield token
                if payload.get("done"):
                    break
    except (requests.RequestException, ValueError) as exc:
        raise OllamaError(f"Streaming request failed: {exc}") from exc
