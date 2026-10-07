# Finance Agent v2 real-agent validation

DriftShield's first real-agent validation path uses the public
Finance Agent v2 workload.

Finance Agent v2 runs its agent with ATIF export enabled. The Vals model-library
writes \`trajectory_atif.json\` into each question's run directory. The exported
ATIF record contains the user/system messages, agent turns, tool calls and
observations, per-turn token metrics, model information, timestamps, and
duration metadata.

## Workflow

1. Run Finance Agent v2 locally using its documented setup and API credentials.
2. Keep the generated \`trajectory_atif.json\` files as the execution record.
3. Convert each trajectory:

~~~bash
python benchmarks/ingest_atif.py path/to/trajectory_atif.json \\
  --output finance_run.json
~~~

4. Review and redact the resulting trace before sharing or storing it outside
the execution environment.
5. Run the existing validation harness only after independent detector labels
have been assigned:

~~~bash
python benchmarks/validate_trace_dataset.py labelled_finance.json \\
  --output finance_validation.json
~~~

## Label rule

The converter deliberately writes:

~~~json
"expected_detectors": []
~~~

and marks the run as \`unlabelled\`. It does not infer labels from Finance
Agent's answer rubric, tool names, repetition, token counts, or final answer.
Finance Agent answer quality and DriftShield detector correctness are separate
measurements.

A review subset must be independently labelled for \`action_loop\`,
\`goal_drift\`, and \`resource_spike\` before precision, recall, F1, or false
positive rate are reported.

## Telemetry fidelity

The converter uses provider/runtime telemetry from ATIF for LLM events:
prompt tokens + completion tokens and per-step duration when present. Tool
arguments and observations are preserved. ATIF currently identifies its export
as partial fidelity and explicitly omits tool-result timestamps, so the
conversion does not invent tool durations.

## First pilot

Start with a small, reproducible subset of the public Finance Agent questions
rather than the full benchmark. Freeze the question IDs and model/tool
configuration used for the run, preserve the raw ATIF files, and create a
separate calibration/holdout labelling record before evaluating detectors.


### Frozen pilot set

The first controlled run uses the 10-question subset in
`validation/finance_agent_v2_pilot.txt`. The file is copied from the public
Finance Agent v2 question workload and deliberately excludes the benchmark
rubrics. Keep this question set unchanged for the first calibration/holdout
exercise so that reruns remain comparable.

Run the Finance Agent with the selected model and documented tool/API
configuration, using one question per run where practical. Preserve the raw
`trajectory_atif.json` output for every question before conversion.

After execution, collect the raw ATIF files with:

~~~bash
python benchmarks/collect_atif.py path/to/finance-agent-logs \\
  --output finance_runs.json
~~~

Then create a separate reviewer copy and independently label each reviewed run
for `action_loop`, `goal_drift`, and `resource_spike`. Do not inspect or use
DriftShield predictions as the reason for assigning those labels. Freeze the
calibration and holdout split before calculating detector metrics.

Package publication remains separate from this validation work.
