from agent.router import detect_intent, route_query


def test_detect_intent_rag():
    assert detect_intent("Explain TCP three-way handshake from my course material.") == "rag"


def test_detect_intent_quiz():
    assert detect_intent("Give me 10 MCQs on TCP three-way handshake.") == "quiz"


def test_detect_intent_summary():
    assert detect_intent("Summarize this lecture.") == "summary"


def test_detect_intent_evaluation():
    assert detect_intent("My answer is: TCP uses UDP because UDP is reliable.") == "evaluate"


def test_route_query_uses_expected_tool():
    assert route_query("Explain TCP three-way handshake from my course material.") == "rag"
    assert route_query("Give me 10 MCQs on TCP three-way handshake.") == "quiz"
    assert route_query("Summarize this lecture.") == "summary"
    assert route_query("My answer is: TCP uses UDP because UDP is reliable.") == "evaluate"
