"""Build a concise evidence report from labelled trace-validation JSON.

The report only reformats measured results. It does not create labels, infer
production accuracy, or fill missing measurements.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    lines = [
        "# DriftShield Mini validation evidence",
        "",
        "## Scope",
        "",
        f"Scenarios: {len(data.get('scenarios', []))}",
        "",
        "## Detector metrics",
        "",
        "| Detector | TP | FP | FN | TN | Precision | Recall | F1 | FPR | Mean latency (events) | Max latency (events) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    metrics = data.get("metrics", {})
    for name in sorted(metrics):
        item = metrics[name]
        lines.append(
            "| {name} | {tp} | {fp} | {fn} | {tn} | {precision} | {recall} | "
            "{f1} | {fpr} | {mean_latency} | {max_latency} |".format(
                name=name,
                tp=item.get("tp"),
                fp=item.get("fp"),
                fn=item.get("fn"),
                tn=item.get("tn"),
                precision=item.get("precision"),
                recall=item.get("recall"),
                f1=item.get("f1"),
                fpr=item.get("false_positive_rate"),
                mean_latency=item.get("mean_detection_latency_events"),
                max_latency=item.get("max_detection_latency_events"),
            )
        )

    lines.extend([
        "",
        "## Evidence rules",
        "",
        "- Labels must come from the agreed external labelling process.",
        "- Calibration and holdout/evaluation data must remain separate.",
        "- Missing measurements remain missing; this report does not infer them.",
        "- Synthetic benchmark results must not be presented as production accuracy.",
    ])
    report = "\n".join(lines) + "\n"
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report)


if __name__ == "__main__":
    main()
