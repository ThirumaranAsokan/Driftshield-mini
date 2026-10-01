# Nebius SWE-agent external trajectory analysis

This benchmark adds a separate analysis path for the public Nebius SWE-agent trajectories dataset.

The dataset contains 80,036 SWE-agent trajectories and provides a target field for whether the issue was solved. It does not provide expected DriftShield detector labels. Therefore this tool reports behavioural associations with the dataset outcome and does not report detector precision, recall, F1, or false positive rate.

## Run a small sample

Install the optional dependency:

    python -m pip install -e ".[external-validation]"

Then run a 1,000-row streaming sample:

    python benchmarks/analyze_swe_agent_trajectories.py --limit 1000 --output swe_agent_analysis.json

The loader uses Hugging Face streaming mode, so the complete dataset does not need to be materialised before analysis.

## Full external analysis

For the complete training split:

    python benchmarks/analyze_swe_agent_trajectories.py --limit 80036 --output swe_agent_analysis.json

The default loop signal is four consecutive identical extracted actions. The default resource signal is an estimated 50,000 tokens of AI trajectory text.

These are deliberately conservative external-analysis signals. They are not claims that the dataset supplies DriftShield ground truth, and the token count is an estimate from logged text rather than measured API usage.

## Interpretation

Use the output to compare behavioural signals between:

- trajectories where target is true (issue solved), and
- trajectories where target is false (issue not solved).

The analysis is an external SWE-agent behavioural study. It is not a production-accuracy evaluation and should not replace a representative, independently labelled pilot dataset.

The dataset is tagged synthetic and is licensed CC-BY-4.0; repository-specific licenses and the dataset's stated Llama 3.1 license notice also apply.
