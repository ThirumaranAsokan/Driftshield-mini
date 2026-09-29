


# DriftShield Mini
```markdown
Privacy-first, zero-infrastructure drift detection for AI agents.

Your LangChain agent just called the same API 47 times. Your CrewAI crew burned £200 in tokens overnight. Your research agent started writing marketing copy instead of financial summaries.

You didn't find out until morning.

DriftShield Mini catches this stuff in real-time.It wraps your existing agent, watches what it does, and pings you on Slack or Discord the moment something goes sideways. No dashboard. No cloud. No account to create. Just a local Python library running alongside your agent.

```

---

##  Installation

Install the updated v0.2.0 package via `pip`:

```bash
pip install driftshield-mini==0.2.0

```

To install with specific framework extras:

```bash
pip install "driftshield-mini[autogen,llama-index,langchain]"

```

---

##  What It Actually Does

DriftShield Mini monitors three core vectors:

1. **Loop Detection**: Is your agent calling the same tool repeatedly or stuck in an infinite cycle (e.g., `search → format → search → format`)? DriftShield flags the pattern before it eats your budget.
2. **Goal Drift**: Is your agent staying on task? DriftShield uses local CPU embeddings to measure how far the agent's recent outputs have drifted from its initial objective.
3. **Resource Spikes**: Is a run burning significantly more tokens or runtime than normal? DriftShield learns baseline behavior and flags anomalous executions.

> **100% Local & Private**: All traces go to a local SQLite database. Embeddings run on your CPU. Nothing leaves your machine except the alerts you explicitly direct to Slack or Discord.

---

##  Supported Frameworks (v0.2.0)

Import framework-specific wrappers using `driftshield_mini`:

```python
from driftshield_mini import DriftMonitor                 # LangChain / manual API
from driftshield_mini.crewai import DriftCrew                 # CrewAI
from driftshield_mini.autogen import DriftAutogenAgent        # Microsoft AutoGen
from driftshield_mini.llama_index import DriftLlamaIndexHandler  # LlamaIndex
from driftshield_mini.openai_assistants import DriftOpenAIClient # OpenAI Assistants
from driftshield_mini.semantic_kernel import DriftKernelFilter   # Semantic Kernel
from driftshield_mini.haystack import DriftHaystackTracer        # Haystack
from driftshield_mini.google_adk import DriftADKCallbacks        # Google ADK

```

### Quick Framework Examples

####  LangChain / Custom API

```python
from driftshield_mini import DriftMonitor

monitor = DriftMonitor(
    agent_id="logistics-v2",
    alert_webhook="[https://hooks.slack.com/services/](https://hooks.slack.com/services/)...",
    goal_description="Optimise routing plans"
)

agent = monitor.wrap(existing_agent)
result = agent.invoke({"input": "optimise route for order #4821"})

```

####  CrewAI

```python
from driftshield_mini.crewai import DriftCrew

crew = DriftCrew(
    crew=existing_crew,
    agent_id="research-team-v1",
    alert_webhook="[https://discord.com/api/webhooks/](https://discord.com/api/webhooks/)...",
)

result = crew.kickoff()

```

####  Microsoft AutoGen

```python
monitored = DriftAutogenAgent(
    agent=assistant, 
    agent_id="autogen-analyst-v1",
    alert_webhook="[https://hooks.slack.com/](https://hooks.slack.com/)..."
)
user_proxy.initiate_chat(monitored.agent, message="Summarise Q3 results")

```

####  LlamaIndex

```python
from llama_index.core.callbacks import CallbackManager
from llama_index.core import Settings

handler = DriftLlamaIndexHandler(agent_id="rag-agent-v1", goal_description="Answer policy questions")
Settings.callback_manager = CallbackManager([handler.callback_handler])

```

####  OpenAI Assistants

```python
client = DriftOpenAIClient(openai.OpenAI(), agent_id="assistant-v1", goal_description="Process invoices")
client.run_assistant(thread_id=t.id, assistant_id=a.id)

```

####  Semantic Kernel

```python
DriftKernelFilter(agent_id="sk-agent-v1", goal_description="Reconcile invoices").apply_to_kernel(kernel)

```

####  Haystack

```python
DriftHaystackTracer(agent_id="haystack-v1").enable()   # Call once before running pipelines

```

####  Google ADK

```python
drift = DriftADKCallbacks(agent_id="adk-v1")
agent = Agent(..., before_tool_callback=drift.before_tool, after_tool_callback=drift.after_tool)

```

---

##  How Calibration Works

For the first 30 runs (configurable), DriftShield quietly observes your agent to establish baseline averages for token consumption, execution times, and tool call sequences. No alerts are fired during this warm-up phase.

Once calibrated, it flags statistical anomalies. You can inspect your agent's baseline at any time:

```bash
driftshield baseline my-agent

```

> **Note**: Even during initial calibration, DriftShield uses hard safety bounds to flag extreme loops (such as 50 identical sequential tool calls).

---

##  Sample Alert Payload

When drift is detected, structured payloads are dispatched to your configured webhook:

```json
{
  "agent_id": "logistics-v2",
  "detector": "action_loop",
  "severity": "HIGH",
  "message": "Action loop: search_inventory called 6x in 45s",
  "suggested_action": "Check search_inventory input/output for stale data or error loops",
  "context": {
    "tool_name": "search_inventory",
    "repeat_count": 6,
    "recent_actions": ["search_inventory", "search_inventory", "search_inventory"]
  }
}

```

---

##  CLI Reference

### Commands Overview

| Command | Description |
| --- | --- |
| `alerts` | View recent drift alerts across agents. |
| `baseline` | Show calculated baseline statistics for an agent. |
| `download-model` | Download embedding models for offline/air-gapped execution. |
| `export` | Export compliance logs formatted for regulatory frameworks (FCA / EU AI Act). |
| `runs` | List recent execution runs for an agent. |
| `traces` | View detailed trace logs for specific agent runs. |

### CLI Usage Examples

```bash
# View alerts triggered in the last 24 hours
driftshield alerts --last 24h

# Inspect execution traces for a specific run
driftshield traces logistics-v2 --run latest

# Export audit logs for compliance (CSV or JSON)
driftshield export --agent logistics-v2 --output audit.csv
driftshield export --agent logistics-v2 --output drift.json --drift-only --format json

# Pre-download embedding models for air-gapped environments
driftshield download-model

```

---

##  Detailed Configuration Options

Defaults are pre-tuned, but all parameters can be customized programmatically:

```python
from driftshield_mini import DriftMonitor

monitor = DriftMonitor(
    agent_id="my-agent",
    alert_webhook="[https://hooks.slack.com/](https://hooks.slack.com/)...",
    goal_description="Summarise financial reports",
    calibration_runs=30,         # Runs before baseline detection activates
    loop_window=20,              # Number of recent actions inspected for loops
    loop_max_repeats=4,          # Repeated tool calls allowed before alerting
    similarity_threshold=0.5,    # Goal drift sensitivity (lower = stricter)
    spike_multiplier=2.5,        # Standard deviation multiplier for resource spikes
    min_alert_severity="MED",    # Minimum severity required to trigger webhooks
    alert_cooldown=60.0          # Seconds to wait before resending duplicate alerts
)

```

---

##  Programmatic Hooks

Attach custom event listeners to act on alerts programmatically:

```python
def handle_drift(event):
    if event.severity.value == "CRITICAL":
        agent.stop()        # Terminate execution
        notify_oncall()     # Trigger on-call notification

monitor.on_drift(handle_drift)

```

---

##  Why I Built This

I kept hearing the same story: a developer builds an agent, it passes local tests, but runs wild overnight in production leaving them with a high API bill and a broken service. Complex observability platforms exist, but they often require hosted infrastructure, SaaS sign-ups, and dashboard overhead.

DriftShield Mini is designed as a minimal, lightweight utility to inform you when an agent strays off course, with zero external platform dependencies.

---

##  License

Distributed under the [MIT License](LICENSE).

```

```
