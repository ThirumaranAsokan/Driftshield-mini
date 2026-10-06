from benchmarks.collect_atif import collect
from benchmarks.ingest_atif import convert_trajectory


def _payload():
    return {
        "schema_version": "ATIF-v1.7",
        "session_id": "finance:q001",
        "agent": {"name": "finance", "model_name": "test/model"},
        "steps": [
            {"step_id": 1, "timestamp": "2026-10-06T20:00:00Z", "source": "system", "message": "system"},
            {"step_id": 2, "timestamp": "2026-10-06T20:00:01Z", "source": "user", "message": "Find revenue."},
            {
                "step_id": 3,
                "timestamp": "2026-10-06T20:00:02Z",
                "source": "agent",
                "message": "I will search.",
                "model_name": "test/model",
                "tool_calls": [
                    {"tool_call_id": "call-1", "function_name": "web_search", "arguments": {"search_query": "revenue"}}
                ],
                "observation": {"results": [{"source_call_id": "call-1", "content": "result"}]},
                "metrics": {"prompt_tokens": 100, "completion_tokens": 25, "extra": {"duration_seconds": 1.5}},
            },
        ],
        "final_metrics": {"total_prompt_tokens": 100, "total_completion_tokens": 25, "total_steps": 3},
    }


def test_convert_finance_style_atif_preserves_tool_calls_and_real_metrics():
    result = convert_trajectory(_payload(), "trajectory_atif.json")
    run = result["scenarios"][0]["runs"][0]

    assert run["run_id"] == "finance:q001"
    assert run["expected_detectors"] == []
    assert run["metadata"]["labels_status"] == "unlabelled"

    llm_event = run["events"][1]
    tool_event = run["events"][2]
    assert llm_event["action_type"] == "llm_request"
    assert llm_event["token_count"] == 125
    assert llm_event["duration_ms"] == 1500.0
    assert tool_event["action_type"] == "tool_call"
    assert tool_event["action_name"] == "web_search"
    assert tool_event["output_data"]["result"] == "result"


def test_collect_atif_does_not_record_local_filesystem_paths(tmp_path):
    trajectory = tmp_path / "nested" / "trajectory_atif.json"
    trajectory.parent.mkdir()
    trajectory.write_text(__import__("json").dumps(_payload()), encoding="utf-8")

    dataset = collect(tmp_path)
    source_file = dataset["scenarios"][0]["runs"][0]["metadata"]["source_file"]

    assert source_file == "trajectory_atif.json"
    assert str(tmp_path) not in source_file
