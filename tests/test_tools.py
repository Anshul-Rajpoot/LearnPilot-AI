from agent.tools import TOOL_REGISTRY


def test_all_tools_build_grounded_prompts():
    context = [{"title": "HTML", "number": 2, "start": 10, "end": 20, "text": "HTML structures a webpage."}]
    for name, tool in TOOL_REGISTRY.items():
        result = tool("test request", context)
        assert result.name == name
        assert "HTML" in result.prompt
        assert "test request" in result.prompt
