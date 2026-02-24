# Contributing to DriftShield

DriftShield is in early development (v0.1). Contributions, bug reports, and feedback are welcome.

## Getting Started

```bash
git clone https://github.com/YOUR_USERNAME/driftshield.git
cd driftshield
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,langchain]"
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Reporting Issues

If you've seen an agent drift that DriftShield didn't catch (or flagged incorrectly), please open an issue with:

1. Which detector was involved (action_loop, goal_drift, resource_spike)
2. What the agent was doing
3. What you expected vs what happened

## Code Style

- Python 3.10+
- Formatted with `ruff`
- Type hints on all public functions
