# DriftShield Mini benchmark

run_benchmark.py is a deterministic validation benchmark for the three current
detector families: action loops, goal drift, and resource spikes.

It uses labelled synthetic traces and reports:

- TP / FP / FN / TN
- precision, recall, F1 and false-positive rate
- detection latency measured in trace events
- monitoring time per event and throughput, plus a relative comparison with a minimal event-construction loop

The goal-drift benchmark uses a deterministic local fake embedder, so it does not
download a model or depend on network access.

The benchmark fails CI if any labelled case produces an unexpected detector result.
The overhead comparison is a low-level reference measurement, not a production
performance claim.

## Real trace validation

validate_trace_dataset.py is the path for the next validation stage: labelled
traces captured from representative agents.

It deliberately requires labels in the dataset instead of generating them. This
keeps TP/FP/FN/TN measurements auditable.

Run:

    python benchmarks/validate_trace_dataset.py path/to/traces.json
    python benchmarks/validate_trace_dataset.py path/to/traces.json --output report.json

Each scenario contains an agent goal, calibration settings, and ordered runs. Each
run contains events plus expected_detectors. Detector values are:

- action_loop
- goal_drift
- resource_spike

For production validation, collect traces from the target workload, apply labels
independently, preserve the event ordering, and keep a separate holdout set.
Do not report synthetic benchmark metrics as customer accuracy.

## Overhead

The existing overhead measurement is a low-level reference comparison and is not a
production performance claim. Production overhead should be measured with the same
agent workload with and without monitoring, using representative traces and
reporting latency, CPU, memory, storage growth, and events/sec.
