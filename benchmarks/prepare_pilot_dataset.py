"""Prepare a labelled pilot dataset for local evaluation.

The input and output use the documented pilot schema. The command removes
common secret-bearing fields and redacts secret-like strings before the
dataset is handed to the validation harness. It does not create labels.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

SENSITIVE_KEYS = {
    "authorization",
    "api_key",
    "apikey",
    "access_token",
    "auth_token",
    "client_secret",
    "cookie",
    "password",
    "private_key",
    "secret",
    "session_token",
    "webhook_url",
}

SECRET_PATTERNS = (
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]{12,}"),
    re.compile(r"(?i)(api[_-]?key|access[_-]?token|secret|password)\s*[:=]\s*[^\s,;]+"),
    re.compile(r"(?i)-----BEGIN [A-Z ]+ PRIVATE KEY-----.*?-----END [A-Z ]+ PRIVATE KEY-----"),
)


def _redact(value: Any, key: str | None = None) -> Any:
    if key and key.lower() in SENSITIVE_KEYS:
        return "[REDACTED]"

    if isinstance(value, dict):
        return {
            str(k): _redact(v, str(k))
            for k, v in value.items()
            if str(k).lower() not in SENSITIVE_KEYS
        }

    if isinstance(value, list):
        return [_redact(item) for item in value]

    if isinstance(value, str):
        result = value
        for pattern in SECRET_PATTERNS:
            result = pattern.sub("[REDACTED]", result)
        return result

    return value


def prepare_dataset(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Dataset must contain a non-empty 'scenarios' list")

    cleaned = _redact(payload)
    cleaned["metadata"] = {
        **(cleaned.get("metadata") if isinstance(cleaned.get("metadata"), dict) else {}),
        "prepared_for_pilot_validation": True,
        "labels_preserved": True,
        "secrets_redacted": True,
    }
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Redact sensitive fields from a pilot dataset")
    parser.add_argument("dataset", type=Path, help="Input labelled pilot JSON")
    parser.add_argument("--output", type=Path, required=True, help="Redacted output JSON")
    args = parser.parse_args()

    report = prepare_dataset(args.dataset)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Prepared pilot dataset: {args.output}")


if __name__ == "__main__":
    main()
