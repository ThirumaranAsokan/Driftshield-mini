# Independent gold validation

## Dataset

Use **nebius/SWE-agent-trajectories**, a public dataset of 80,036 SWE-agent runs. Its documented row fields include `instance_id`, `model_name`, `target`, `trajectory`, `exit_status`, `generated_patch`, and `eval_logs`. The dataset is CC-BY-4.0, with additional per-repository licensing requirements.

The `target` field means SWE-bench issue resolution. It is **not** a DriftShield detector label and must not be converted into action-loop, goal-drift, or resource-spike labels.

## Review design

Create a deterministic sample of 300 trajectories:

- 200 calibration records
- 100 locked holdout records
- Do not inspect DriftShield predictions while creating labels.
- Ideally use two independent reviewers for the holdout.
- Resolve disagreements before unblinding detector predictions.

Run:

    python -m pip install -e ".[external-validation]"
    python benchmarks/build_gold_review_set.py --limit 300 --output validation/gold_review.json

The script creates a blinded reviewer file and a private audit manifest. Do **not** commit the private manifest or raw trajectory data.

## Label definitions

### action_loop

`true` only when the trajectory contains a repeated action pattern that is reasonably evidence of execution stagnation rather than an intentional repeated operation.

Strong evidence includes four or more consecutive equivalent tool/action invocations, or an alternating/repeating sequence that continues without meaningful progress.

Do not label ordinary repetition as a loop merely because the same tool is legitimately used several times.

### goal_drift

`true` when the agent's behaviour/output materially departs from the task goal stated in the trajectory and the departure is not a justified intermediate step toward completing that goal.

Do not infer drift solely from a failed task. A task can fail without goal drift, and an agent can drift while eventually recovering.

### resource_spike

`true` only when the trajectory provides independent evidence of unusually excessive resource use relative to the agreed review rule.

For this dataset, reviewer evidence may include extreme trajectory length, unusually many actions, or explicit runtime/resource information when present. Because provider token telemetry is not available for every row, this label must not be described as measured API-token overuse.

## Required reviewer fields

Each record must contain:

    labels.action_loop
    labels.goal_drift
    labels.resource_spike
    review_reason.action_loop
    review_reason.goal_drift
    review_reason.resource_spike
    reviewer_id
    review_confidence

Allowed labels are `true` or `false`. If evidence is genuinely insufficient, mark the record for adjudication rather than guessing.

## Converting labels to DriftShield validation

After the holdout labels are frozen, convert each reviewed trajectory into the existing labelled-trace schema:

    {
      "name": "...",
      "goal": "...",
      "calibration_runs": 0,
      "runs": [
        {
          "run_id": "...",
          "expected_detectors": [
            "action_loop"
          ],
          "events": [...]
        }
      ]
    }

The conversion must happen **after** independent labelling. Never use DriftShield output to populate `expected_detectors`.

Then run:

    python benchmarks/validate_trace_dataset.py validation/gold_holdout.json --output validation/gold_holdout-results.json

The validator reports, separately for all three detector families:

- TP
- FP
- FN
- TN
- precision
- recall
- F1
- false-positive rate
- detection latency

## Important limitation

The public SWE-agent dataset gives a task-success target, but that target is not the ground truth for any DriftShield detector. Independent human review is therefore still required before any precision/recall/F1/FPR claim is made.


# External validation datasets

## tau2-bench

This repository now validates against published tau2-bench result files from the upstream sierra-research/tau2-bench repository.

The CI workflow downloads three published GPT-4.1 result files covering the airline, retail, and telecom domains. The upstream files contain task definitions, complete simulation messages, tool calls, rewards, termination reasons, and agent-cost fields where available.

The analysis is intentionally an external behavioural validation, not detector ground truth.

### Action-loop association

The current external proxy is deliberately simple and auditable:

- extract actual assistant tool_calls from the stored simulation messages;
- flag a loop signal when the same tool name occurs four or more times consecutively;
- compare that signal with the upstream benchmark outcome (reward < 1.0).

This does not mean reward < 1.0 is an action_loop label. It measures association between an observable trajectory pattern and benchmark failure.

### Resource association

Where upstream simulations expose agent_cost, the analysis computes a P99 cost threshold within the downloaded corpus and measures its association with benchmark failure.

Where message-level provider usage is present, the analysis also computes a P99 provider-token threshold and reports the same association.

These are stronger than the SWE-agent text-length estimate when telemetry exists, but they are still not resource_spike ground truth.

### Goal drift

No independent goal_drift label is supplied by tau2-bench. The benchmark task reward, termination reason, or agent-error review must not be relabelled as goal drift. The analysis therefore records goal_drift as not directly evaluable.

### Reproducibility

Run locally after downloading the upstream files:

    python benchmarks/analyze_tau2_bench.py external-data/tau2-bench/*.json --output validation-results/tau2-bench-analysis.json

CI performs the download and analysis automatically and uploads the JSON result as the tau2-bench-analysis artifact.

## Accuracy claims

Only the independent gold-labelled dataset is eligible for product-level TP/FP/FN/TN, precision, recall, F1, and FPR claims.

SWE-agent and tau2-bench results remain explicitly labelled as external outcome associations.
