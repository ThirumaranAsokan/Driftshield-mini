# Contributing to DriftShield

Thanks for taking a look at DriftShield. The project is still in alpha. Contributions that improve correctness, testing, integrations, documentation, and validation are especially useful.

The package is not published to PyPI. Release publication is intentionally separate from normal development and validation.

## Before you start

DriftShield is an in-process monitoring library for AI agents. The three detector families are:

- `action_loop`
- `goal_drift`
- `resource_spike`

Please read the README and the relevant detector/tests before changing behaviour. If you are working on validation, also read [docs/gold-validation.md](docs/gold-validation.md).

## Development setup

```bash
git clone https://github.com/ThirumaranAsokan/Driftshield-mini.git
cd Driftshield-mini

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For external trajectory validation:

```bash
python -m pip install -e ".[external-validation]"
```

Install an integration extra only when you need it, for example:

```bash
python -m pip install -e ".[langchain]"
```

## Running the checks

Before opening a pull request, run the checks that apply to your change:

```bash
pytest -q
ruff check .
python -m build
```

For validation-tool changes, also run the relevant benchmark or analysis command and include the command and result in the pull request description.

GitHub Actions runs the main CI, integration, and external-validation workflows. A green workflow means the configured checks passed for that commit. It is not exhaustive proof of every code path and it is not a substitute for independent detector labelling.

## Making a code change

A useful contribution normally follows this flow:

1. Create a branch from `main`.
2. Make the smallest focused change.
3. Add or update tests for behaviour that changed.
4. Run Ruff and the relevant tests locally.
5. Explain what changed and why in the pull request.
6. Include any known limitations or follow-up work.
7. Do not add generated datasets, credentials, API keys, webhook URLs, or private agent traces.

For detector changes, please include regression cases for both positive and negative examples. A detector should not be considered correct just because it fires on a failure case; false positives matter too.

## Validation contributions

Validation is currently one of the highest-value areas for the project.

### Public trajectory review

The repository uses public trajectory corpora as evidence, not as automatic DriftShield ground truth. For the SWE-agent review set, the source task outcome is deliberately not used to create detector labels.

When reviewing a trajectory:

- **action_loop**: label true only when repetition reasonably indicates execution stagnation rather than a legitimate repeated operation.
- **goal_drift**: label true only when behaviour materially departs from the stated task and the departure is not a justified intermediate step.
- **resource_spike**: label true only when the trajectory contains independent evidence of unusually excessive resource use under the agreed review rule.

If the evidence is genuinely unclear, flag it for adjudication instead of guessing.

Do not look at DriftShield's prediction first and then choose the label to match it.

### Human gold validation

The intended validation path is:

```bash
python benchmarks/build_gold_review_set.py \
  --limit 300 \
  --seed 20261005 \
  --output validation/gold_review.json
```

Then review the records independently and freeze the labels. Split the set into 200 calibration records and 100 holdout records.

After labels are frozen:

```bash
python benchmarks/convert_gold_review.py validation/gold_holdout_review.json \
  --split holdout \
  --output validation/gold_holdout.json

python benchmarks/validate_trace_dataset.py validation/gold_holdout.json \
  --output validation/gold_holdout-results.json
```

For a credible accuracy claim, use two reviewers for the holdout when possible and document how disagreements were resolved.

The current repository does **not** claim that the assistant-reviewed preliminary labels are human gold truth.

### Finance Agent v2 real-agent pilot

The first controlled finance-agent execution uses the frozen question set at `validation/finance_agent_v2_pilot.txt`. Run the real Finance Agent v2 workload, preserve its raw `trajectory_atif.json` files, and convert them with the repository ingestion tools before review.

Keep the source workload's answer rubrics and outcomes separate from DriftShield detector labels. A reviewer must independently assess `action_loop`, `goal_drift`, and `resource_spike` without using DriftShield's prediction as the reason for a label. Freeze calibration and holdout records before reporting detector metrics.

The Finance Agent workflow is described in [docs/finance-agent-real-validation.md](docs/finance-agent-real-validation.md). The ingestion path is intended to preserve provider/runtime telemetry available in ATIF and must not invent missing tool timing data.

Do not publish finance accuracy numbers from the pilot until the labels are independently established and the evaluation set is frozen.

## External datasets

The project currently has analysis tooling for:

- **FinTrace** — 800 financial-agent trajectories.
- **Nebius SWE-agent trajectories** — 80,036 public trajectories.
- **tau2-bench** — published trajectory/result files across airline, retail, and telecom domains.

These sources help exercise the system against real agent behaviour. Their task outcomes and benchmark reference trajectories must not be relabelled as DriftShield detector truth.

## Pull request expectations

A good pull request should answer:

- What problem does this change solve?
- Which files or components changed?
- How was it tested?
- Did you add regression coverage?
- Are there known limitations?
- Does it change detector semantics or thresholds?
- If it changes validation, what is the source of the labels?

Please keep PRs focused. Large unrelated refactors make detector behaviour harder to review.

## Reporting bugs and detector misses

If DriftShield misses a case or produces an unexpected alert, open an issue with as much reproducible detail as you can safely share:

1. detector name
2. agent/framework used
3. expected behaviour
4. observed behaviour
5. relevant trace shape or a redacted trace
6. configuration/thresholds
7. Python version and dependency versions

Never include secrets or private customer traces.

## Concurrency and state

The monitor is designed to be used from applications that may handle multiple runs at once. Avoid adding shared mutable detector state unless it is protected or scoped to a run. New concurrency-sensitive behaviour should include a deterministic regression test.

Detector lifecycle is explicit: `DriftMonitor.end_run()` notifies detectors when a run is complete so per-run state can be released. Resource-spike counters are retained for all active runs rather than silently evicting an older active run; cleanup happens when that run ends. Changes to this lifecycle should include tests for both active-run isolation and end-of-run cleanup.

When adding a new dataset or validation script, run the normal test and Ruff checks before opening the PR. Dataset-specific code must not bypass the same quality gates used by the core package.


## Project direction

DriftShield is being developed as a practical monitoring and validation project for AI-agent applications. The immediate goal is to establish reliable detector behaviour and evidence before making stronger release or accuracy claims.

The current priorities are:

1. establish independently reviewed detector labels on representative agent traces
2. validate the detectors on finance and other realistic workloads without treating task outcomes as detector truth
3. strengthen integration lifecycle coverage, including concurrent and long-running runs
4. measure false positives, false negatives, latency, CPU, memory, and storage overhead
5. improve examples and integration documentation so other developers can reproduce results
6. keep package publication separate until the validation evidence is strong enough to support a release

Contributors are welcome to challenge assumptions and report negative results. A failed experiment or detector miss is useful evidence and should be documented rather than hidden.

## Maintainer review record

The current code review of this project has been carried out by **Thirumaran Asokan**, including review of detector state handling, concurrent-run behaviour, regression coverage, validation workflows, package build/install checks, integration checks, and the distinction between benchmark results and independently established detector labels.

This review record is not a substitute for independent external review. Contributors should still review changes critically and raise issues when implementation or validation evidence is incomplete.

## How to take ownership of a piece of the project

If you want to take the project further, choose one focused area and open an issue or pull request describing the intended result. Useful ownership areas include:

- **Detector engineering:** improve one detector while preserving explicit regression tests and documenting threshold/semantic changes.
- **Validation:** independently label trajectories, adjudicate disagreements, and publish reproducible evaluation methodology.
- **Finance validation:** test against realistic financial-agent traces and document what can and cannot be inferred from those traces.
- **Framework integrations:** test complete start/record/end lifecycles against supported agent frameworks, including concurrent runs.
- **Performance:** benchmark overhead on representative workloads and identify regressions before they reach release candidates.
- **Data quality:** add new datasets only with clear provenance, licensing, extraction rules, and a reproducible validation command.
- **Documentation:** improve examples, setup instructions, detector explanations, and reproducibility notes.

For any new validation dataset, keep three things separate: the source dataset's own outcome/reference labels, DriftShield's detector output, and any independently reviewed human labels. Do not silently convert one into another.

## A good first contribution

A first contribution does not need to be large. A well-reproduced detector miss, a false-positive example with a regression test, an integration lifecycle test, or a carefully documented validation result is valuable.

Before starting a large refactor, open an issue so the scope and expected evidence can be agreed first. This helps keep the project understandable as more contributors join.
## Code style

- Python 3.10+
- Ruff for linting
- Type hints for public APIs
- Prefer small, testable functions
- Keep external integrations isolated from the core detector logic

## Review checklist for contributors

Before asking for review:

- [ ] Tests pass
- [ ] Ruff passes
- [ ] New behaviour has regression coverage
- [ ] No secrets or private traces were added
- [ ] Documentation is updated if behaviour or configuration changed
- [ ] Validation claims are backed by the correct type of evidence
- [ ] Accuracy numbers are not presented unless detector labels are independently established
- [ ] Concurrency-sensitive state has a regression test when applicable
- [ ] Detector lifecycle state is released explicitly at run completion when applicable
- [ ] Validation output/artifacts were inspected, not just the workflow status

## Where help is most useful right now

The project would particularly benefit from contributors who can:

- independently label agent trajectories for the three detector families
- review goal-drift false positives/negatives
- improve resource-spike measurement using real provider telemetry
- test integrations against current framework releases
- add adversarial and edge-case tests
- measure CPU, memory, latency, and storage overhead on representative agents
- review the SQLite/concurrency behaviour under longer-running workloads
- improve documentation and examples

If you are unsure where to start, open an issue describing the area you want to work on and we can keep the change focused.
