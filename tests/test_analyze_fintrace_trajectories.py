"""Tests for the FinTrace trajectory adapter."""

from benchmarks.analyze_fintrace_trajectories import analyse_rows


def test_fintrace_adapter_extracts_tool_calls_and_output():
    rows = [
        {
            "id": 123,
            "source_query": "Find the revenue for Apple.",
            "task_type": "Financial QA",
            "output_trajectory": [
                {"role": "user", "content": "Find the revenue for Apple."},
                {
                    "role": "assistant",
                    "content": "I will retrieve the revenue.",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "income_statement",
                                "arguments": '{"symbol":"AAPL"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "content": '{"revenue": 100}',
                },
                {
                    "role": "assistant",
                    "content": "Apple revenue was 100.",
                },
            ],
        }
    ]

    report = analyse_rows(rows, limit=1)

    assert report["rows_analyzed"] == 1
    assert report["results"][0]["tool_actions"] == 1
    assert report["results"][0]["trajectory_messages"] == 4
    assert report["accuracy_metrics"].startswith("not computed")
    assert report["golden_trajectory_note"].startswith("golden_trajectories")
