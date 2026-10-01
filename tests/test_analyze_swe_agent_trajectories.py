import json

from benchmarks.analyze_swe_agent_trajectories import analyse_rows, extract_actions, extract_signals


def test_extract_actions_from_swe_style_text():
    text = "Action: ls -la\n"
    assert extract_actions(text) == ["ls -la"]


def test_extract_actions_from_nebius_apply_patch_text():
    text = """Will execute following command for apply_patch: ```
cd //repo && git apply -v patch.diff --allow-empty
```"""
    assert extract_actions(text) == [
        "apply_patch: cd //repo && git apply -v patch.diff --allow-empty"
    ]


def test_extract_signals_counts_ai_text_and_actions():
    row = {
        "target": False,
        "trajectory": json.dumps([
            {"role": "system", "text": "system"},
            {"role": "ai", "text": "Action: ls\n"},
            {"role": "user", "text": "files"},
            {"role": "ai", "text": "Action: ls\n"},
            {"role": "ai", "text": "Action: ls\n"},
            {"role": "ai", "text": "Action: ls\n"},
        ]),
    }
    signals = extract_signals(row)
    assert signals.steps == 6
    assert signals.max_consecutive_action == 4


def test_analysis_keeps_outcome_separate_from_detector_accuracy():
    rows = [
        {"target": True, "trajectory": [{"role": "ai", "text": "Action: ls"}]},
        {"target": False, "trajectory": [{"role": "ai", "text": "Action: ls"}]},
    ]
    result = analyse_rows(rows)
    assert result["rows_analyzed"] == 2
    assert "accuracy_metrics" in result
    assert result["by_target"]["target_true"]["trajectories"] == 1
    assert result["by_target"]["target_false"]["trajectories"] == 1

def test_extract_signals_accepts_content_field_and_assistant_role():
    row = {
        "target": True,
        "trajectory": [
            {
                "role": "assistant",
                "content": (
                    "Will execute following command for apply_patch: ```"
                    "cd //repo && git apply patch.diff --allow-empty"
                    "```"
                ),
            }
        ],
    }
    signals = extract_signals(row)
    assert signals.action_names == (
        "apply_patch: cd //repo && git apply patch.diff --allow-empty",
    )


def test_extract_signals_finds_actions_in_observation_role():
    row = {
        "target": False,
        "trajectory": [
            {
                "role": "user",
                "text": (
                    "Will execute following command for apply_patch: ```"
                    "cd //repo && git apply patch.diff --allow-empty"
                    "```"
                ),
            }
        ],
    }
    signals = extract_signals(row)
    assert signals.action_names == (
        "apply_patch: cd //repo && git apply patch.diff --allow-empty",
    )


def test_extract_signals_finds_actions_in_eval_logs():
    row = {
        "target": True,
        "trajectory": [],
        "eval_logs": (
            "Will execute following command for apply_patch: ```"
            "cd //repo && git apply patch.diff --allow-empty"
            "```"
        ),
    }
    signals = extract_signals(row)
    assert signals.action_names == (
        "apply_patch: cd //repo && git apply patch.diff --allow-empty",
    )


def test_extract_actions_from_swe_agent_interface_command():
    text = """Let's inspect the file.

```
open azure_functions_worker/dispatcher.py
```"""
    assert extract_actions(text) == ["open azure_functions_worker/dispatcher.py"]
