from benchmarks.prepare_pilot_dataset import prepare_dataset


def test_prepare_dataset_redacts_sensitive_fields(tmp_path):
    source = tmp_path / "pilot.json"
    source.write_text(
        '{"scenarios":[{"name":"agent","runs":[{"expected_detectors":[],'
        '"events":[{"action_name":"call","input_data":{"api_key":"secret"},'
        '"output_data":{"text":"Bearer abcdefghijklmnop"}}]}]}]}',
        encoding="utf-8",
    )

    result = prepare_dataset(source)

    event = result["scenarios"][0]["runs"][0]["events"][0]
    assert "api_key" not in event["input_data"]
    assert event["output_data"]["text"] == "[REDACTED]"
    assert result["metadata"]["labels_preserved"] is True
    assert result["metadata"]["secrets_redacted"] is True
