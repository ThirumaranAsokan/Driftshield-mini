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

The benchmark fails CI if any labelled case produces an unexpected detector result. The overhead comparison is a low-level reference measurement, not a production performance claim.
Production validation requires representative labelled traces from the target workload.

Run:

    python benchmarks/run_benchmark.py
