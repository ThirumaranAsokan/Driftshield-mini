# Getting DriftShield on GitHub — Step by Step

## Your Final Repo Structure

```
driftshield/
├── .gitignore
├── LICENSE                     # MIT — removes friction for adoption
├── README.md                   # First thing prospects see
├── CONTRIBUTING.md             # Signals "real project" to beta testers
├── pyproject.toml              # Modern Python packaging (replaces setup.py)
│
├── driftshield/                # The actual library
│   ├── __init__.py             # Public API: DriftMonitor, models
│   ├── monitor.py              # Core wrapper — the main entry point
│   ├── models.py               # TraceEvent, DriftEvent, BaselineStats
│   ├── cli.py                  # `driftshield alerts`, `traces`, etc.
│   ├── crewai.py               # CrewAI integration
│   │
│   ├── detectors/              # The three drift detectors
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract base class
│   │   ├── action_loop.py      # Catches repetitive cycles
│   │   ├── goal_drift.py       # Semantic distance from objective
│   │   └── resource_spike.py   # Token/time/API anomalies
│   │
│   ├── storage/                # SQLite persistence layer
│   │   └── __init__.py         # TraceStore class
│   │
│   ├── alerts/                 # Webhook dispatching
│   │   └── __init__.py         # Slack, Discord, generic
│   │
│   └── baseline/               # Calibration logic
│       └── __init__.py         # Calibrator class
│
└── tests/                      # Test suite
    ├── test_core.py            # Pytest version
    └── test_standalone.py      # Zero-dependency version
```

## Step 1: Create the GitHub Repo

1. Go to https://github.com/new
2. Repository name: `driftshield` (or `driftshield-mini`)
3. Description: "Real-time behavioural drift detection for agentic AI systems"
4. Set to **Public** (your spec says open-source to build trust)
5. Do NOT add a README/gitignore/license (we already have them)
6. Click "Create repository"

## Step 2: Push Your Code

Open your terminal and run these commands:

```bash
# Navigate to wherever you downloaded the driftshield folder
cd path/to/driftshield

# Initialise git
git init

# Add all files
git add .

# First commit
git commit -m "feat: initial DriftShield v0.1 — 3 drift detectors, SQLite storage, CLI, webhook alerts"

# Connect to GitHub (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/driftshield.git

# Push
git branch -M main
git push -u origin main
```

## Step 3: Verify It Looks Right

After pushing, check your repo page. You should see:
- README.md rendered nicely at the bottom
- Clean folder structure at the top
- License badge showing MIT

## Step 4: Set Up the Basics

### Add Topics (for discoverability)
Go to your repo → click the gear icon next to "About" → add topics:
`python`, `ai-agents`, `langchain`, `crewai`, `monitoring`, `drift-detection`, `observability`, `llm`

### Enable Issues
Should be on by default. This is where beta testers will report bugs.

### Optional: Add a Release
Go to Releases → "Create a new release"
- Tag: `v0.1.0`
- Title: `v0.1.0 — Initial Release`
- Description: Copy the key features from your README

## What NOT to Do Yet

- ❌ Don't set up GitHub Actions/CI yet (overkill for week 1)
- ❌ Don't publish to PyPI yet (wait until you have beta feedback)
- ❌ Don't create a GitHub org (use your personal account for now)
- ❌ Don't add a docs site (README is enough for v0.1)

## When to Add More

| Milestone | Add to repo |
|-----------|-------------|
| First 3 beta users | GitHub Actions for tests |
| 10+ GitHub stars | Publish to PyPI (`pip install driftshield`) |
| v0.2 with dashboard | Separate `docs/` folder or ReadTheDocs |
| First paying customer | GitHub org + SECURITY.md |

## Making It Installable From GitHub (Right Now)

Even without PyPI, beta testers can install directly:

```bash
pip install git+https://github.com/YOUR_USERNAME/driftshield.git
```

Share this one-liner with your beta testers. It's much easier than
asking them to clone and install manually.
