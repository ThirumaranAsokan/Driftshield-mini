# DriftShield Mini

Agent monitoring for AI applications.

DriftShield Mini is an **in-process monitoring library** for agentic applications. It records observable agent behaviour locally and looks for three signal families:

- **Action loops** — repeated tool calls or repeating action sequences.
- **Goal drift** — semantic distance between a declared goal and agent output.
- **Resource spikes** — unusually high token, tool-call, or runtime consumption relative to a learned baseline, plus hard safety limits.

Traces, drift events, and baselines are stored in local SQLite. Goal embeddings use a local sentence-transformers model. Optional webhook alerts support Slack, Discord, and generic HTTP endpoints.

> DriftShield reports observed signals. It does not prove that an agent has violated its task, and an alert is not automatically a failure. Production deployments should evaluate thresholds and false positives/negatives against their own workloads.

## Project status

**Current status: alpha / source release.**

Version **0.2.2** is the current project version. The package has **not yet been published to PyPI**. Until the release artifacts have been built and verified, install DriftShield from the Git repository.

The repository currently has automated tests and CI for Python 3.10, 3.11, and 3.12, plus import-level checks for the supported framework adapters. The validation suite also contains deterministic synthetic benchmarks and separate pilot/external-analysis tooling. The detector test suite includes boundary cases for repeated actions, repeated sequences, interleaved runs, resource thresholds, severity bands, and active-run counter lifecycle.

Synthetic benchmark results and external trajectory analysis are **not production accuracy claims**.

## Installation

### Install from source

```bash
git clone https://github.com/ThirumaranAsokan/Driftshield-mini.git
cd Driftshield-mini
python -m pip install -e .
```

### Framework integrations

Install only the adapter dependencies you need:

```bash
python -m pip install -e ".[langchain]"
python -m pip install -e ".[crewai]"
python -m pip install -e ".[autogen]"
python -m pip install -e ".[llama-index]"
python -m pip install -e ".[openai]"
python -m pip install -e ".[semantic-kernel]"
python -m pip install -e ".[haystack]"
python -m pip install -e ".[google-adk]"
```

Development and test dependencies:

```bash
python -m pip install -e ".[dev]"
```

External trajectory analysis tools:

```bash
python -m pip install -e ".[external-validation]"
```

## Quick start

The core API can be used without a framework adapter:

```python
from driftshield_mini import DriftMonitor

monitor = DriftMonitor(
    agent_id="my-agent",
    goal_description="Summarise financial reports",
    calibration_runs=30,
)

run_id = monitor.start_run()

monitor.record_event(
    action_type="tool_call",
    action_name="search_reports",
    run_id=run_id,
    token_count=120,
)

monitor.record_event(
    action_type="llm_request",
    action_name="agent_complete",
    run_id=run_id,
    token_count=450,
    output_data={"text": "Revenue increased during the quarter."},
)

monitor.end_run(run_id)

for event in monitor.get_recent_alerts():
    print(event.detector.value, event.severity.value, event.message)

monitor.close()
```

For framework-specific integrations, use the adapters described below.

## Detection

### Action-loop detection

The action-loop detector watches recent `tool_call` events and identifies repeated action patterns.

Examples:

```text
search_inventory
search_inventory
search_inventory
search_inventory
```

or:

```text
search → format → search → format → search → format
```

Detection is based on observed action names and sequence repetition. It does not inspect tool semantics or prove that a repeated action is invalid.

### Goal drift

The goal detector compares a declared goal with textual agent output using embeddings and cosine similarity.

```python
monitor = DriftMonitor(
    agent_id="research-agent",
    goal_description="Summarise financial reports",
    similarity_threshold=0.5,
)
```

The default embedding model is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Goal similarity is a monitoring signal, not a formal task-correctness test.

### Resource spikes

DriftShield tracks run-level:

- token consumption
- tool calls
- execution duration

After calibration, these measurements are compared with the stored baseline. Independent hard limits also protect against extreme resource consumption before a statistical baseline exists.

## Calibration and baselines

Calibration is configurable:

```python
monitor = DriftMonitor(
    agent_id="my-agent",
    calibration_runs=30,
)
```

The stored baseline includes:

- mean/std tokens per run
- mean/std tools per run
- mean/std duration
- common action sequences
- mean/std goal similarity when valid samples are available

The baseline is recalculated from recent stored runs up to the configured calibration window. It is therefore an **adaptive rolling baseline**, not a permanently frozen first-N-run baseline.

Extreme safety limits and loop detection can still operate while calibration is pending.

For production use, evaluate baseline contamination and workload changes with representative traces.

## Alerts

Optional webhook alerts support:

- Slack
- Discord
- generic HTTP webhooks

Webhook delivery runs in the background so a slow notification endpoint does not block detector execution.

```python
monitor = DriftMonitor(
    agent_id="my-agent",
    alert_webhook="https://example.com/webhook",
    min_alert_severity="HIGH",
    alert_cooldown=60,
)
```

Do not put credentials, tokens, or webhook URLs directly into source code or committed trace data. Treat webhook URLs as secrets.

## Local storage

By default, DriftShield stores data at:

```text
~/.driftshield/driftshield.db
```

A custom SQLite path can be supplied:

```python
monitor = DriftMonitor(
    agent_id="my-agent",
    db_path="/path/to/driftshield.db",
)
```

SQLite uses WAL mode and thread-local connections.

Traces can contain agent inputs, outputs, tool names, and metadata. Keeping the database local does not automatically make sensitive data safe. Apply appropriate retention, access-control, and redaction policies.

## Offline / air-gapped operation

The goal detector uses `sentence-transformers/all-MiniLM-L6-v2`.

To stage the model locally:

```bash
driftshield download-model
```

The loader checks for a package-local model and the local Hugging Face cache before falling back to an online download. For genuinely air-gapped deployment, pre-stage the model and verify the resulting installation in the target environment.

## Supported integrations

Current adapters are:

```python
from driftshield_mini.crewai import DriftCrew
from driftshield_mini.autogen import DriftAutogenAgent
from driftshield_mini.llama_index import DriftLlamaIndexHandler
from driftshield_mini.openai_assistants import DriftOpenAIClient
from driftshield_mini.semantic_kernel import DriftKernelFilter
from driftshield_mini.haystack import DriftHaystackTracer
from driftshield_mini.google_adk import DriftADKCallbacks
```

The core/manual API is exposed through:

```python
from driftshield_mini import DriftMonitor
```

Framework APIs change frequently. The AutoGen adapter targets the legacy `pyautogen` 0.2.x API (`pyautogen>=0.2,<0.3`). The OpenAI adapter targets the legacy Assistants/Threads API exposed by the OpenAI Python client. Test the exact dependency versions used by your deployment.

## CLI

```bash
# Recent alerts
driftshield alerts --last 24h

# Baseline
driftshield baseline my-agent

# Runs
driftshield runs my-agent

# Traces
driftshield traces my-agent --run latest

# Export trace records
driftshield export --agent my-agent --output audit.csv

# Export drift incidents
driftshield export --agent my-agent --output drift.json --drift-only --format json

# Stage the embedding model locally
driftshield download-model
```

Exports are structured trace/incident records in CSV or JSON. They can support audit workflows; exporting records does not by itself establish FCA, EU AI Act, or other regulatory compliance.

## Programmatic callbacks

```python
def handle_drift(event):
    if event.severity.value == "CRITICAL":
        agent.stop()

monitor.on_drift(handle_drift)
```

Callbacks execute in the monitoring path, so application callbacks should be short and failure-tolerant.

## Configuration

```python
monitor = DriftMonitor(
    agent_id="my-agent",
    goal_description="Summarise financial reports",
    calibration_runs=30,
    loop_window=20,
    loop_max_repeats=4,
    similarity_threshold=0.5,
    spike_multiplier=2.5,
    min_alert_severity="MED",
    alert_cooldown=60.0,
)
```

These are starting values, not universal optimal settings. Evaluate them against the target workload.

## Validation and evidence

The repository separates engineering benchmarks from real-trace validation.

### Deterministic synthetic benchmark

The benchmark suite contains labelled synthetic detector cases and reports:

- TP / FP / FN / TN
- precision / recall / F1
- false-positive rate
- first-detection event latency

These results are useful for regression testing but **must not be presented as production accuracy**.

### Labelled pilot traces

For genuine detector validation, use representative traces with independent expected-detector labels. Keep calibration runs separate from evaluation/holdout runs.

The validation tools are:

```bash
python benchmarks/prepare_pilot_dataset.py pilot_raw.json --output pilot_redacted.json
python benchmarks/validate_trace_dataset.py pilot_redacted.json --output validation.json
python benchmarks/build_evidence_report.py validation.json --output evidence.md
```

The preparation utility redacts common secret-bearing fields and preserves supplied labels. It is a safety aid, not a guarantee of anonymisation.

### External SWE-agent trajectory analysis

The repository also contains a separate analysis path for the public Nebius SWE-agent trajectories dataset. A verified 1,000-trajectory run achieved 99.6% action extraction and produced 126 loop signals under the current external-analysis rule.

Those signals are **exploratory behavioural evidence only**. The dataset's target field represents SWE-bench issue resolution, not independent DriftShield detector labels, so it cannot provide DriftShield precision, recall, F1, FPR, TP, FP, FN, or TN.

See [docs/swe-agent-external-analysis.md](docs/swe-agent-external-analysis.md).

## FinTrace external trajectory validation

A dedicated external-analysis path now covers the public **FinTrace** financial-agent trajectory set.

The repository now includes:

- `benchmarks/analyze_fintrace_trajectories.py` for normalising FinTrace's nested trajectory schema into DriftShield monitoring events.
- A regression test covering the nested `turn/output/function_call/message` structure.
- `.github/workflows/fintrace-validation.yml` to run the analysis in CI and retain the result as a workflow artifact.
- [docs/fintrace-external-analysis.md](docs/fintrace-external-analysis.md) documenting the method, evidence, and limitations.

The verified CI analysis ran all **800** FinTrace records and observed:

- **7,960** tool actions
- **9,469** normalized trajectory messages
- **684,915** estimated trajectory tokens
- **360** rows with a goal-drift signal
- **306** rows with an action-loop signal
- **0** rows above the 50,000 estimated-token resource threshold
- **32** task types

These are detector observations, not independent accuracy measurements. FinTrace does not provide independent DriftShield detector labels, so the repository does **not** claim FinTrace precision, recall, F1, FPR, TP, FP, FN, or TN from this analysis. The dataset's `golden_trajectories` are reference/evaluation material, not DriftShield detector labels, and the resource estimate is based on trajectory text/tool arguments rather than provider usage telemetry.

The next validation step is an independently labelled review subset, with calibration and holdout traces kept separate. The package is **not being published to PyPI yet**; release/package work remains pending until the validation work is complete.


## Finance Agent v2 real-agent pilot

The first controlled real-agent validation path uses the public Vals Finance Agent v2 workload. The repository includes:

- `benchmarks/ingest_atif.py` for ATIF v1.7 trajectory conversion
- `benchmarks/collect_atif.py` for collecting multiple Finance Agent runs
- `docs/finance-agent-real-validation.md` for the execution and labelling procedure
- `validation/finance_agent_v2_pilot.txt` containing the frozen 10-question pilot set

The pilot is deliberately label-neutral at ingestion time. Finance Agent answer quality, benchmark rubrics, and task outcomes are not used as DriftShield detector labels. The reviewed runs must be independently labelled for `action_loop`, `goal_drift`, and `resource_spike` before detector metrics are calculated.

The intended sequence is:

1. run the real Finance Agent workload and preserve the raw ATIF traces
2. convert and review the traces without assigning inferred detector labels
3. independently label the three detector families
4. freeze calibration and holdout records
5. calculate precision, recall, F1, false-positive rate, detection latency, and overhead

This pilot is the next evidence step; it is not yet a finance detector-accuracy claim. Package publication remains separate and the package is not published to PyPI.

## Validation status and what is next

The project now has a validation pipeline that is intentionally separated into three levels:

1. **Deterministic synthetic tests** for regression coverage. These report TP/FP/FN/TN, precision, recall, F1, false-positive rate, and detection latency for controlled labelled cases.
2. **External trajectory analysis** using public SWE-agent, FinTrace, and tau2-bench material. These runs exercise the detectors against real agent trajectories and are useful for real-trace evidence, parser coverage, and failure-mode discovery. They do **not** provide independent DriftShield detector labels, so their outcome associations are not product accuracy metrics.
3. **Independent labelled validation**, which is the remaining step before making real-world detector accuracy claims. The repository contains the tooling to build a deterministic 300-trajectory review set from the public `nebius/SWE-agent-trajectories` dataset, split into 200 calibration records and 100 holdout records.

### What has been verified

The repository CI currently checks:

- Python 3.10, 3.11, and 3.12
- Ruff and the test suite
- package build and wheel smoke testing
- eight framework integration paths: CrewAI, OpenAI, Semantic Kernel, LlamaIndex, AutoGen, LangChain, Google ADK, and Haystack
- FinTrace parsing and analysis across all 800 records
- SWE-agent external trajectory analysis
- published tau2-bench trajectory analysis
- deterministic gold-review-set generation and validation tooling

The current development pass also adds focused detector boundary tests and makes per-run detector cleanup explicit. These changes are intentionally kept separate from package publication; PyPI publishing has not been performed.

### Current engineering gate

The codebase now includes regression coverage for concurrent run goals, shared resource-counter updates, detector-exception observability, and detector lifecycle cleanup. Resource counters remain available for active runs and are released explicitly when a run ends, avoiding silent eviction of active-run state. These tests are intended to catch cross-run state leakage and silent detector failures before new datasets or adapters are added.

The most important remaining validation gate is **independent review of representative agent trajectories**.

For the gold-validation process:

1. Build the 300-record review set with `benchmarks/build_gold_review_set.py`.
2. Review `action_loop`, `goal_drift`, and `resource_spike` independently of DriftShield's predictions.
3. Record a short reason and confidence for every label.
4. Keep the 200 calibration records separate from the 100 holdout records.
5. Ideally have two reviewers label the holdout and adjudicate disagreements before looking at detector results.
6. Convert the frozen labels with `benchmarks/convert_gold_review.py`.
7. Run `benchmarks/validate_trace_dataset.py` and report the resulting confusion matrices and metrics.

Do not use SWE-bench success/failure, FinTrace golden trajectories, or DriftShield's own predictions as substitutes for detector ground truth.

See [docs/gold-validation.md](docs/gold-validation.md) and [CONTRIBUTING.md](CONTRIBUTING.md) for the review workflow.

## Development

Run the tests:

```bash
pytest -q
```

Run linting:

```bash
ruff check .
```

Build and inspect the package locally before any release:

```bash
python -m pip install build
python -m build
```

The generated `dist/` artifacts are release candidates only; publishing is a separate step.

## Data handling and regulated environments

DriftShield keeps monitoring data on the machine by default. That can be useful when an application should not send full agent traces to an external monitoring service, but local storage does not remove the need for data controls.

For sensitive or regulated workloads:

- collect only authorised traces
- keep raw traces access-controlled
- redact or minimise sensitive fields before sharing
- keep credentials and tokens out of trace datasets
- define retention policies
- validate detector performance against representative workloads

DriftShield itself does not make a deployment compliant with a specific regulation.

## Why I built this

AI agents can fail in ways that ordinary request/response monitoring does not capture: repeated tool loops, semantic task deviation, and unexpectedly expensive execution.

I built DriftShield Mini as a small monitoring layer that can run beside an existing agent without requiring a separate observability service.

## License

MIT License. See [LICENSE](LICENSE).

## Project

GitHub: https://github.com/ThirumaranAsokan/Driftshield-mini
