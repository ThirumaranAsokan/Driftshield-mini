# FinTrace external analysis

This document describes how to run DriftShield Mini against the public FinTrace evaluation trajectories.

FinTrace contains 800 financial-agent evaluation records. Each record includes a financial query, an output trajectory, and a golden reference trajectory. The current public dataset does not provide independent action_loop, goal_drift, or resource_spike labels for DriftShield, so this analysis is trajectory evidence rather than detector accuracy validation.

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

## Prepare an independent review set

Build a blinded, deterministic review set from the FinTrace test split:

    python benchmarks/build_fintrace_review_set.py --limit 300 --seed 20261015 --output validation/fintrace_review.json

To use a locally downloaded JSON copy instead, pass --input path/to/fintrace.json. The input may be a JSON list or an object containing a data or rows list. The script uses reservoir sampling over the supplied rows. By default, 200 records are marked calibration and 100 holdout; if the input has fewer records, the actual counts are reported in the output.

The reviewer file includes task prompts and observed output-trajectory messages/tool calls. It excludes DriftShield predictions, FinTrace reference answers, golden trajectories, and task outcome scores. Detector labels and reasons are blank when generated. Reviewers should independently fill labels.action_loop, labels.goal_drift, labels.resource_spike, their matching review_reason fields, reviewer_id, and review_confidence. Mark unclear cases for adjudication rather than guessing.

The excerpt is capped at 80 messages by default; use --max-messages to change that cap. This file is a review aid, not a claim that every trajectory has been fully reviewed. Treat the output as potentially sensitive and do not commit review files containing raw trajectory text.

Keep calibration and holdout records separate. Do not calculate accuracy metrics until labels are frozen and the holdout has not been used to tune thresholds.

## Next validation step

Use the reviewed subset to create independent labels for:

- action_loop
- goal_drift
- resource_spike

Then convert those reviewed records into the existing labelled-trace format used by benchmarks/validate_trace_dataset.py. Keep calibration and holdout records separate.

Only after that step should DriftShield precision, recall, F1, FPR, and detection latency be reported for this external financial workload.
