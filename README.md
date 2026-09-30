# DriftShield Mini

Privacy-first, local-first behavioural monitoring for AI agents.

DriftShield Mini runs alongside an agent and looks for three observable forms of abnormal behaviour:

1. **Action loops** — repeated tool calls or repeating tool-name sequences.
2. **Goal drift** — semantic distance between an agent's declared/run goal and its textual output.
3. **Resource spikes** — unusually high token, tool-call, or runtime consumption compared with a learned baseline, plus hard safety limits.

It is designed as an **in-process library**. Traces and baselines are stored locally in SQLite. Goal embeddings run locally on CPU. Optional webhook alerts can be sent to Slack, Discord, or a generic HTTP endpoint.

> DriftShield detects behavioural signals. It does not prove that an agent has violated its task or that an alert is always a failure. Production deployments should tune thresholds and evaluate false positives/negatives against their own workloads.

## Installation

For the audited development branch, install from source:

```bash
git clone https://github.com/ThirumaranAsokan/Driftshield-mini.git
cd Driftshield-mini
pip install -e .
```

After v0.2.2 is published to PyPI, the versioned installation will be:

```bash
pip install driftshield-mini==0.2.2
```

Framework integrations are optional:

```bash
pip install "driftshield-mini[langchain]"
pip install "driftshield-mini[crewai]"
pip install "driftshield-mini[autogen]"
pip install "driftshield-mini[llama-index]"
pip install "driftshield-mini[openai]"
pip install "driftshield-mini[semantic-kernel]"
pip install "driftshield-mini[haystack]"
pip install "driftshield-mini[google-adk]"
```

Install development dependencies with:

```bash
pip install -e ".[dev]"
```

## Quick start

```python
from driftshield_mini import DriftMonitor

monitor = DriftMonitor(
    agent_id="logistics-v2",
    goal_description="Optimise routing plans",
    alert_webhook="https://hooks.slack.com/services/...",
)

agent = monitor.wrap(existing_agent)
result = agent.invoke({"input": "optimise route for order #4821"})

monitor.close()
```

The wrapper shown above targets the standard `invoke()` style used by the core/manual integration. Framework-specific adapters are available separately.

## How detection works

### 1. Action-loop detection

The detector watches recent `tool_call` events.

Examples it can identify include:

```text
search_inventory
search_inventory
search_inventory
search_inventory

or

search → format → search → format → search → format
```

Detection is based on observed tool names and sequence repetition. It does **not** inspect tool semantics or prove that repeated calls are invalid.

### 2. Goal drift

The goal detector embeds the declared goal and textual agent output with the local `sentence-transformers/all-MiniLM-L6-v2` model and compares them using cosine similarity.

A similarity threshold can be supplied directly:

```python
DriftMonitor(
    agent_id="research-agent",
    goal_description="Summarise financial reports",
    similarity_threshold=0.5,
)
```

During calibration, DriftShield also records run-level goal/output similarity statistics. When enough valid goal/output samples exist, the calibrated baseline can contribute to the threshold.

Semantic similarity is a monitoring signal, not a formal task-correctness test.

### 3. Resource spikes

DriftShield maintains run-level totals for:

- token consumption
- tool calls
- execution duration

After calibration, these are compared with the stored baseline. Independent hard limits also protect against extreme resource consumption before a statistical baseline exists.

## Calibration

Calibration is configurable:

```python
DriftMonitor(
    agent_id="my-agent",
    calibration_runs=30,
)
```

The baseline currently contains:

- mean/std tokens per run
- mean/std tools per run
- mean/std duration
- common action sequences
- mean/std goal similarity when valid goal/output samples are available

The baseline is recalculated from recent stored runs up to the configured calibration window. Therefore, this is an **adaptive rolling baseline**, not a permanently frozen first-30-run baseline.

Extreme safety limits and loop detection can still produce alerts during calibration.

For production use, baseline contamination and changing workload distributions should be evaluated with representative data.

## Alerts

Alerts can be delivered through:

- Slack webhooks
- Discord webhooks
- generic HTTP webhooks

Webhook delivery is dispatched in the background so a slow or unavailable notification endpoint does not block the monitored agent.

The alert dispatcher also applies severity filtering and a per-agent/per-detector cooldown.

Example:

```python
monitor = DriftMonitor(
    agent_id="my-agent",
    alert_webhook="https://example.com/webhook",
    min_alert_severity="HIGH",
    alert_cooldown=60,
)
```

Do not place credentials or sensitive data directly in source code. Treat webhook URLs as secrets.

## Local storage

By default, DriftShield stores data in:

```text
~/.driftshield/driftshield.db
```

You can provide another path:

```python
DriftMonitor(
    agent_id="my-agent",
    db_path="/path/to/driftshield.db",
)
```

SQLite uses WAL mode and thread-local connections.

Traces may contain agent inputs, outputs, tool names and metadata. Local storage does not automatically make sensitive data safe; apply your own retention and access controls.

## Offline / air-gapped embedding model

The goal detector uses `sentence-transformers/all-MiniLM-L6-v2`.

To prepare a local model:

```bash
driftshield download-model
```

The loader checks the packaged model and local Hugging Face cache first. If no local model is available, the current implementation can fall back to downloading the model. For genuinely air-gapped operation, pre-stage the model before deployment.

## Supported integrations

Current adapters include:

```python
from driftshield_mini import DriftMonitor
from driftshield_mini.crewai import DriftCrew
from driftshield_mini.autogen import DriftAutogenAgent
from driftshield_mini.llama_index import DriftLlamaIndexHandler
from driftshield_mini.openai_assistants import DriftOpenAIClient
from driftshield_mini.semantic_kernel import DriftKernelFilter
from driftshield_mini.haystack import DriftHaystackTracer
from driftshield_mini.google_adk import DriftADKCallbacks
```

Framework APIs change frequently. The AutoGen adapter currently targets the legacy `pyautogen` 0.2.x API (`pyautogen>=0.2,<0.3`). The OpenAI adapter currently targets the legacy Assistants/Threads API exposed by the OpenAI Python client. Compatibility should be tested against the exact framework/client versions used by your deployment.

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

# Prepare the embedding model locally
driftshield download-model
```

Exports are structured trace/incident records in CSV or JSON. They can support audit and compliance workflows; **exporting records does not by itself establish FCA, EU AI Act, or other regulatory compliance.**

## Programmatic drift callbacks

```python
def handle_drift(event):
    if event.severity.value == "CRITICAL":
        agent.stop()

monitor.on_drift(handle_drift)
```

Callbacks run in the monitoring path, so application callbacks should be short and failure-tolerant.

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

These are starting points, not universal optimal values. Evaluate them against your workload.

## Engineering status

DriftShield Mini is currently an **alpha-stage engineering project**.

The project has:

- local SQLite trace storage
- three detector families
- configurable baselines
- local embeddings
- webhook alerts
- CLI inspection/export
- framework adapters
- automated tests and CI for supported Python versions

The benchmark suite in `tests/test_benchmark.py` provides deterministic detector cases. The next validation step is to run it against representative real traces and report:

- true positives
- false positives
- false negatives
- detection latency
- monitoring overhead
- behaviour under contaminated/non-stationary baselines

## Development

Run the test suite:

```bash
pytest -q
```

Run linting:

```bash
ruff check .
```

## Why I built this

AI agents can fail in ways that ordinary request/response monitoring does not capture: repeated tool loops, semantic task deviation, and unexpectedly expensive execution.

DriftShield Mini is intended to provide a small, local monitoring layer that can sit beside an existing agent without requiring a separate observability service.

## License

MIT License. See [LICENSE](LICENSE).

## Project

GitHub: https://github.com/ThirumaranAsokan/Driftshield-mini
