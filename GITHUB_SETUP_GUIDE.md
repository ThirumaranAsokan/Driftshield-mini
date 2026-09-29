# DriftShield-Mini v0.2.0 — Windows Setup Guide

## What this zip contains

```
Driftshield-mini-v0.2.0/
├── driftshield_mini/          ← the RENAMED package (was "driftshield")
│   ├── __init__.py            ← MODIFIED (v0.2.0, new imports)
│   ├── cli.py                 ← MODIFIED (+ export, + download-model)
│   ├── crewai.py              ← FIXED (tool-level events via CrewAI event bus)
│   ├── autogen.py             ← NEW (Microsoft AutoGen)
│   ├── llama_index.py         ← NEW (LlamaIndex RAG agents)
│   ├── openai_assistants.py   ← NEW (OpenAI Assistants / Responses API)
│   ├── semantic_kernel.py     ← NEW (Microsoft Semantic Kernel)
│   ├── haystack.py            ← NEW (Haystack, popular in EU/UK)
│   ├── google_adk.py          ← NEW (Google Agent Development Kit)
│   ├── embeddings.py          ← NEW (offline model loading — no runtime HF downloads)
│   ├── export.py              ← NEW (CSV/JSON audit export for FCA / EU AI Act)
│   ├── detectors/
│   │   └── goal_drift.py      ← MODIFIED (uses offline embeddings loader)
│   ├── models.py              ← copy from your repo, rename imports
│   ├── monitor.py             ← copy from your repo, rename imports
│   ├── alerts/__init__.py     ← copy from your repo, rename imports
│   ├── baseline/__init__.py   ← copy from your repo, rename imports
│   ├── storage/__init__.py    ← copy from your repo, rename imports (unchanged strings OK)
│   └── detectors/action_loop.py, base.py, resource_spike.py  ← copy, rename imports
├── driftshield/
│   └── __init__.py            ← NEW back-compat shim (old imports still work)
├── tests/
│   └── test_wrappers.py       ← NEW
├── pyproject.toml             ← MODIFIED (v0.2.0, extras per framework)
└── SETUP_WINDOWS.md           ← this file
```

## Installation steps (VS Code, Windows)

1. Unzip this folder. Copy the NEW/MODIFIED files into your repo (overwrite).
2. Rename your repo folder `driftshield/` → `driftshield_mini/`.
3. In the renamed folder, replace imports (VS Code: Ctrl+Shift+H, scope = driftshield_mini folder):
   - Find `from driftshield` → Replace `from driftshield_mini`
   - Find `import driftshield` → Replace `import driftshield_mini`
   - Do NOT touch `.driftshield` or `driftshield.db` strings (storage paths).
4. Copy `driftshield/__init__.py` (shim) into a new `driftshield/` folder at repo root.
5. Replace `pyproject.toml` with the one in this zip. Delete the old `driftshield/cli.py` if a duplicate exists.
6. Install & test:

```powershell
cd path\to\Driftshield-mini
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m pytest tests/test_wrappers.py -v
driftshield --help
driftshield download-model     # bundle the embedding model for offline use
```

## Compliance commands (for the FCA one-pager)

```powershell
driftshield export --agent my-agent --output audit.csv           # full audit trail
driftshield export --agent my-agent --output drift.json --drift-only --format json
```
