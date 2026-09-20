from agent.evaluation import evaluate_tool_selection


def test_evaluation_dataset_routes_expected_tools():
    results = evaluate_tool_selection()
    assert len(results) == 4
    assert all(item["tool_correct"] for item in results)
