# DriftShield Mini pilot validation

Use representative traces from the intended deployment workload. Keep calibration
runs separate from evaluation/holdout runs.

For each evaluation run, record expected detector labels from an agreed labelling
process; never derive labels from DriftShield output.

## Measurements

- TP / FP / FN / TN
- precision / recall / F1
- false-positive rate
- first-detection event latency
- controlled workload overhead and SQLite storage

Use:
- `benchmarks/validate_trace_dataset.py`
- `benchmarks/baseline_robustness.py`
- `benchmarks/measure_overhead.py`

## Evidence rules

Report trace population, date range, workload characteristics, calibration/holdout
sizes, retention/redaction policy, and webhook configuration. Synthetic benchmark
metrics and synthetic overhead must not be presented as production performance.
