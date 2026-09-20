from agent.router import detect_intent


def test_router_intents():
    assert detect_intent("Explain CSS box model") == "rag"
    assert detect_intent("Give me 10 MCQs on HTML forms") == "quiz"
    assert detect_intent("Summarize this lecture") == "summary"
    assert detect_intent("Check my answer: TCP is reliable because of acknowledgements") == "evaluate"
    assert detect_intent("") == "invalid"
