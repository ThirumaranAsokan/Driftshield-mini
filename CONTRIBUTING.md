# Contributing to DriftShield

Thanks for taking a look at DriftShield. The project is still in alpha, so contributions that improve correctness, testing, documentation, integrations, and validation are especially useful.

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

GitHub Actions runs the main CI, integration, and external-validation workflows. A green workflow is evidence that the checked path passed; it is not a substitute for independent detector labelling.

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
