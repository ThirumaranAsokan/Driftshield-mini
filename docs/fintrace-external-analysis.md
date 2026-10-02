# FinTrace external analysis

This document describes how to run DriftShield Mini against the public FinTrace evaluation trajectories.

FinTrace contains 800 financial-agent evaluation records. Each record includes a financial query, an output trajectory, and a golden reference trajectory. The current public dataset does not provide independent action_loop, goal_drift, or resource_spike labels for DriftShield, so this analysis is behavioural evidence rather than detector accuracy validation.

## Run the analysis

Install the optional dataset dependency:

    python -m pip install -e ".[external-validation]"

Run all 800 records:

    python benchmarks/analyze_fintrace_trajectories.py --output fintrace-report.json

For a small parser check:

    python benchmarks/analyze_fintrace_trajectories.py --limit 20

The adapter can also read a locally downloaded JSON copy:

    python benchmarks/analyze_fintrace_trajectories.py --input testset.json --limit 20

## What is extracted

The adapter reads output_trajectory messages and records:

- assistant tool calls as tool_call events
- assistant text as llm_request output events
- tool names and arguments
- trajectory message position
- estimated token consumption when provider usage telemetry is not present

The existing DriftShield detectors are used without changing their core behaviour.

## Verified CI result

GitHub Actions FinTrace validation run #3 completed successfully on commit c7d1a47750f5421a857970800027f499efb38b95.

The generated report covered all 800 FinTrace records and contained:

- 7,960 tool actions
- 9,469 normalized trajectory messages
- 684,915 estimated trajectory tokens
- 360 records with a goal_drift signal
- 306 records with an action_loop signal
- 0 records above the 50,000-token resource threshold
- 32 distinct task types

These are detector observations from the adapter run, not independently labelled correctness measurements.

## What the results mean

The report can show how often DriftShield emits signals while processing the external financial trajectories.

It does not provide:

- TP / FP / FN / TN
- precision / recall / F1
- false-positive rate

Those metrics require independent DriftShield detector labels.

The FinTrace reference_answer, output_answer, and golden trajectory are evaluation material for the FinTrace benchmark. A different tool path or output does not automatically mean DriftShield goal drift. The adapter therefore does not turn golden-trajectory differences into detector labels.

Resource-spike results need extra care. The public trajectory does not provide a provider-usage record for every message, so this adapter uses a rough text/argument token estimate. Those results must not be presented as measured API token usage or production resource consumption.

## Next validation step

Use a manually reviewed subset of these financial trajectories to create independent labels:

- action_loop
- goal_drift
- resource_spike

Then convert those reviewed records into the existing labelled-trace format used by benchmarks/validate_trace_dataset.py. Keep calibration and holdout records separate.

Only after that step should DriftShield precision, recall, F1, FPR, and detection latency be reported for this external financial workload.
