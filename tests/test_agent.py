import pandas as pd
import pytest

from agent.graph import LearnPilotAI
from llm.ollama import OllamaError, stream


class FakeRetriever:
    def search(self, query, top_k=5, prefer_semantic=True):
        return pd.DataFrame([{"title":"HTML", "number":1, "start":0, "end":10, "text":"HTML is markup."}])

    def records(self, result):
        return result.to_dict(orient="records")


def test_agent_routes_retrieves_and_generates():
    agent = LearnPilotAI(FakeRetriever(), lambda prompt: "generated")
    state = agent.run("Give me 5 MCQs on HTML")
    assert state.intent == "quiz"
    assert state.tool_used == "Quiz Generator"
    assert state.retrieved_context
    assert state.final_response == "generated"


def test_stream_raises_on_ollama_error_payload(monkeypatch):
    class FakeResponse:
        def __iter__(self):
            return iter([])

    def fake_post(*args, **kwargs):
        class MockResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def iter_lines(self, decode_unicode=True):
                return iter(['{"error":"model not found"}'])

            def raise_for_status(self):
                return None

        return MockResponse()

    monkeypatch.setattr("requests.post", fake_post)

    with pytest.raises(OllamaError, match="model not found"):
        list(stream("hello"))
