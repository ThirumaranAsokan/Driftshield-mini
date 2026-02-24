# DriftShield Mini

**Real-time behavioural drift detection for agentic AI systems.**

Wraps existing LangChain and CrewAI agents, monitors their live behaviour against a learned baseline, and fires Slack/Discord alerts when drift is detected. No dashboard. No cloud dependency. Zero infrastructure.

## Installation

```bash
pip install driftshield
```

## Quick Start

### LangChain

```python
from driftshield import DriftMonitor

monitor = DriftMonitor(
    agent_id="logistics-v2",
    alert_webhook="https://hooks.slack.com/...",
    calibration_runs=30,  # baseline from first 30 runs
)

agent = monitor.wrap(existing_agent)

# That's it. Run the agent normally.
# DriftShield captures traces, builds baseline, and alerts automatically.
result = agent.invoke({"input": "optimise route for order #4821"})
```

### CrewAI

```python
from driftshield.crewai import DriftCrew

crew = DriftCrew(
    crew=existing_crew,
    agent_id="research-team-v1",
    alert_webhook="https://discord.com/api/webhooks/...",
)

result = crew.kickoff()
```

### Manual Instrumentation

```python
from driftshield import DriftMonitor

monitor = DriftMonitor(agent_id="my-agent")

run_id = monitor.start_run(goal="Summarise Q4 financial reports")

# Record events as your agent runs
drifts = monitor.record_event(
    action_type="tool_call",
    action_name="search_database",
    token_count=150,
)

if drifts:
    print(f"Drift detected: {drifts[0].message}")

monitor.end_run()
```

## Three Drift Detectors

| Detector | What it catches | How it works |
|----------|----------------|--------------|
| **Action Loop** | Agents stuck in repetitive cycles | Sliding window sequence matching |
| **Goal Drift** | Agent wandering from its objective | Sentence embeddings + cosine similarity |
| **Resource Spike** | Abnormal token/time consumption | Statistical deviation from baseline |

## CLI

```bash
# View recent alerts
driftshield alerts --last 24h

# Inspect traces for a specific agent
driftshield traces logistics-v2 --run latest

# Show baseline statistics
driftshield baseline logistics-v2

# List recent runs
driftshield runs logistics-v2
```

## Configuration

```python
monitor = DriftMonitor(
    agent_id="my-agent",
    alert_webhook="https://hooks.slack.com/...",
    goal_description="Summarise financial reports",
    calibration_runs=30,         # runs before baseline is ready
    loop_window=20,              # sliding window for loop detection
    loop_max_repeats=4,          # repeats before flagging
    similarity_threshold=0.5,    # goal drift sensitivity
    spike_multiplier=2.5,        # std deviations for resource spike
    min_alert_severity="MED",    # minimum severity to alert on
    alert_cooldown=60.0,         # seconds between alerts per detector
)
```

## Drift Callbacks

```python
def handle_drift(event):
    if event.severity.value == "CRITICAL":
        # Kill the agent, page on-call, etc.
        pass

monitor.on_drift(handle_drift)
```

## Tech Stack

- Python 3.10+
- SQLite (local, zero config)
- sentence-transformers (CPU, no API calls)
- scikit-learn (basic statistics)
- httpx (webhook delivery)
- click + rich (CLI)

## License

MIT
