# Supported frameworks (v0.2.0)

All wrappers share the same detectors (action loop, goal drift, resource spike),
local SQLite storage, and Slack/Discord alerting. Nothing leaves the machine.

```python
from driftshield_mini import DriftMonitor                    # LangChain / manual API
from driftshield_mini.crewai import DriftCrew                 # CrewAI
from driftshield_mini.autogen import DriftAutogenAgent        # Microsoft AutoGen
from driftshield_mini.llama_index import DriftLlamaIndexHandler  # LlamaIndex
from driftshield_mini.openai_assistants import DriftOpenAIClient # OpenAI Assistants
from driftshield_mini.semantic_kernel import DriftKernelFilter   # Semantic Kernel
from driftshield_mini.haystack import DriftHaystackTracer        # Haystack
from driftshield_mini.google_adk import DriftADKCallbacks        # Google ADK
```

## AutoGen (Microsoft)

```python
monitored = DriftAutogenAgent(agent=assistant, agent_id="autogen-analyst-v1",
                              alert_webhook="https://hooks.slack.com/...")
user_proxy.initiate_chat(monitored.agent, message="Summarise Q3 results")
```

## LlamaIndex

```python
from llama_index.core.callbacks import CallbackManager
from llama_index.core import Settings
handler = DriftLlamaIndexHandler(agent_id="rag-agent-v1", goal_description="Answer policy questions")
Settings.callback_manager = CallbackManager([handler.callback_handler])
```

## OpenAI Assistants

```python
client = DriftOpenAIClient(openai.OpenAI(), agent_id="assistant-v1", goal_description="Process invoices")
client.run_assistant(thread_id=t.id, assistant_id=a.id)
```

## Semantic Kernel

```python
DriftKernelFilter(agent_id="sk-agent-v1", goal_description="Reconcile invoices").apply_to_kernel(kernel)
```

## Haystack

```python
DriftHaystackTracer(agent_id="haystack-v1").enable()   # once, before running pipelines
```

## Google ADK

```python
drift = DriftADKCallbacks(agent_id="adk-v1")
agent = Agent(..., before_tool_callback=drift.before_tool, after_tool_callback=drift.after_tool)
```

## Compliance audit export

```bash
driftshield export --agent my-agent --output audit.csv
driftshield export --agent my-agent --output drift.json --drift-only --format json
```

## Offline / air-gapped

```bash
driftshield download-model
```
