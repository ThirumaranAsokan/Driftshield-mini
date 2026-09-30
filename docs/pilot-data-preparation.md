# Pilot data preparation

DriftShield Mini keeps pilot validation local-first. Before a labelled trace
dataset is passed to the validation harness, prepare a redacted copy:

```bash
python benchmarks/prepare_pilot_dataset.py pilot_raw.json --output pilot_redacted.json
python benchmarks/validate_trace_dataset.py pilot_redacted.json --output validation.json
python benchmarks/build_evidence_report.py validation.json --output evidence.md
```

## What preparation does

- Preserves the documented scenario/run/event structure.
- Preserves `expected_detectors`; it never creates or changes labels.
- Removes common secret-bearing keys such as API keys, passwords, access
  tokens, cookies, private keys, and webhook URLs.
- Redacts common bearer-token and secret-like string patterns.
- Writes a separate output file so the original source is not modified.

## Pilot handling rules

1. Collect only traces authorised for the pilot.
2. Keep raw traces local and access-controlled.
3. Prepare a redacted evaluation copy before sharing or committing anything.
4. Do not commit raw traces, credentials, tokens, customer data, or webhook URLs.
5. Keep calibration and holdout/evaluation runs separate.
6. Record the trace population, date range, workload characteristics, and
   labelling process with the evidence report.

The redaction utility is a safety aid, not a guarantee of anonymisation.
Review the resulting dataset before external transfer.
