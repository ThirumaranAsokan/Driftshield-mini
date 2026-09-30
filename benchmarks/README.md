# DriftShield Mini benchmark

run_benchmark.py is a deterministic validation benchmark for the three current
detector families: action loops, goal drift, and resource spikes.

It uses labelled synthetic traces and reports:

- TP / FP / FN / TN
- precision, recall, F1 and false-positive rate
- detection latency measured in trace events
- an estimated monitoring overhead against a minimal event-construction loop

The goal-drift benchmark uses a deterministic local fake embedder, so it does not
download a model or depend on network access.

These numbers are engineering benchmark results, not production accuracy claims.
Production validation requires representative labelled traces from the target workload.

Run:

    python benchmarks/run_benchmark.py
