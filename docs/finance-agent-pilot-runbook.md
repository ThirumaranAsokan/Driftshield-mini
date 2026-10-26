# Finance Agent pilot: collection and validation

This runbook is for a small, controlled pilot of DriftShield Mini against real Finance Agent runs. It is a collection and inspection step, not an accuracy benchmark. Do not publish a package or report detector accuracy from unlabelled runs.

## 1. Access and safe setup

- Confirm you have approved access to the Finance Agent v2 repository/runtime and permission to run it.
- Use a test environment and non-sensitive financial tasks. Do not use real customer data, production credentials, or live trading/payment actions.
- Set provider/API spending limits before running anything. Start with 3–5 short runs; expand only after inspecting their cost and logs.
- Keep API keys in environment variables or the runtime's secret manager. Never commit credentials, .env files, raw logs, or trajectory excerpts.
- Record the repository commit, model/provider, runtime configuration, task prompt, run start/end time, and any retries.

If access is gated or unavailable, stop here and record the blocker. Do not substitute a public dataset and call it a live pilot.

## 2. Preserve raw run evidence

For every run, retain the original trajectory_atif.json emitted by the runtime. Keep an untouched copy before converting it. Where available, also preserve provider usage/cost records that report actual prompt tokens, completion tokens, cached tokens, and cost. Keep these records linked to the trajectory using a stable run/session ID.

The trajectory's token estimates or text length are not a substitute for provider usage telemetry. If usage data is missing, mark it unavailable rather than deriving an asserted actual value.

Store pilot material outside the repository in a restricted local directory. Before sharing an excerpt for review, remove secrets and personal or account data while preserving tool-call order and the evidence needed to judge the detector.

Suggested local layout:

```text
pilot-data/
  raw/
    run-001/trajectory_atif.json
    run-002/trajectory_atif.json
  provider-usage/
    run-001.json
    run-002.json
  notes/
    pilot-manifest.json
```

## 3. Convert and collect trajectories

From the repository root, with the optional validation dependencies installed. **Before using the validation harness on ATIF-converted data, ensure the unlabelled-run safeguards from PR #28 are present in your checkout. The current main branch must not be used to calculate metrics on unlabelled runs.**

```bash
python -m pip install -e ".[external-validation]"
python benchmarks/ingest_atif.py pilot-data/raw/run-001/trajectory_atif.json --output /tmp/run-001-validation.json
python benchmarks/collect_atif.py pilot-data/raw --output /tmp/finance-agent-pilot.json
python benchmarks/validate_trace_dataset.py /tmp/finance-agent-pilot.json
```

The collector searches recursively for files named trajectory_atif.json. Check the printed run count against the pilot manifest. Inspect the converted output before drawing conclusions: confirm that each source session appears once, tool calls are preserved in order, goal text is present, and available ATIF token/duration metrics were carried through.

Keep the original trajectories separate from converted validation JSON. The converted format is for running DriftShield; it does not replace the raw evidence.

## 4. Run DriftShield and inspect output

Run the existing validation harness against the collected JSON and preserve the complete report and command output. Record the exact source commit and input-file checksum with the results.

All ATIF-converted runs are marked unlabelled. Therefore:

- They can show which signals DriftShield emitted and when.
- They must not be counted as confirmed positives or negatives.
- Precision, recall, F1, and false-positive rate must remain undefined until independent labels exist.
- A signal count is not a detector accuracy score.
- Estimated token totals are not proof of provider usage-limit exhaustion.

If the harness or report counts an unlabelled run as a true/false positive or negative, stop and fix that issue before continuing.

## 5. Independent review before any accuracy claim

Prepare a review sheet that does not expose DriftShield predictions to reviewers. For each run, reviewers should label action_loop, goal_drift, and resource_spike independently, with a short evidence-based reason and confidence. Use uncertain when the trace does not contain enough evidence. In particular, do not infer resource spikes from text length alone when provider usage or reliable run-level totals are absent.

Use the calibration subset to clarify the rubric and tune thresholds. Resolve disagreements and freeze the rubric before evaluating the holdout subset. Keep holdout labels hidden from anyone tuning the detector. Report per-detector TP, FP, FN, TN, precision, recall, F1, false-positive rate, and detection latency only where the labelled examples and metric denominators support them. Include sample counts and uncertainty; do not generalize from a tiny pilot to production performance.

## 6. Pilot completion checklist

- [ ] Runtime access approved; model/provider and code revision recorded.
- [ ] Budget limits configured; only test tasks used.
- [ ] Raw ATIF trajectories preserved and checksummed.
- [ ] Provider usage/cost evidence saved where available, with missing fields explicitly noted.
- [ ] Run count and session IDs reconcile with the pilot manifest.
- [ ] Conversion output and full validation report retained outside the repository.
- [ ] No unlabelled run used to calculate accuracy metrics.
- [ ] Independent review completed before calibration/holdout metrics are reported.
- [ ] Any limitations, missing telemetry, and failures are written up honestly.

Package publication remains out of scope until real-agent validation and the remaining release checks are complete.
