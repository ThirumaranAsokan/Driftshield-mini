# Finance pilot validation

This document defines the validation needed before DriftShield Mini is presented as tested on a real financial-services workload.

The aim is to measure how the existing detectors behave on a real agent workflow. This is not a regulatory certification exercise.

## 1. What counts as a finance validation trace

A useful pilot trace should come from an actual financial-services workflow or a controlled test environment that reproduces that workflow closely.

The trace should contain, where available:
- agent goal or task definition
- model/agent identifier
- tool calls and tool results
- action names and sequence
- timestamps or durations
- token counts
- run boundaries
- agent inputs and outputs needed for labelling
- detector labels created independently of DriftShield
- enough context to explain why a label was assigned

Do not upload customer records, account numbers, payment data, credentials, API keys, or other unnecessary confidential information to the repository.

Keep the raw trace inside the pilot environment. Create a redacted evaluation copy for DriftShield validation.

## 2. Start with one real workflow

Do not try to validate every financial use case at once.

Choose one real workflow where an agent already exists or can be tested under controlled conditions. Examples include:
- financial-document research and summarisation
- internal financial analysis
- customer-support assistance
- compliance or regulatory reporting support
- financial-crime investigation support
- reconciliation or operational processing

The selected workflow should have a clearly stated goal and a human owner who understands what constitutes normal and abnormal behaviour.

## 3. Independent labelling

The person or process assigning labels must not use DriftShield's detector result as the reason for the label.

For each run, record:
- action_loop
- goal_drift
- resource_spike

Each value should be based on the agreed labelling rules for the pilot.

A run can have more than one positive label.

Recommended label fields:

    {
      "expected_detectors": ["action_loop"],
      "label_reason": "The agent repeated the same search operation after receiving the same result.",
      "label_source": "reviewer",
      "review_status": "reviewed"
    }

Keep the label reason outside the detector implementation so the evaluation remains independent.

## 4. Calibration and evaluation split

Do not tune thresholds on the same runs used for the final metrics.

Use separate data for:
1. calibration
2. threshold selection
3. holdout evaluation

The holdout set should remain untouched until the detector configuration is fixed.

The existing validator already supports separate calibration runs and evaluation runs. The pilot should preserve this separation in the source data and in the evidence report.

## 5. What to measure

For each detector, report:
- TP
- FP
- FN
- TN
- precision
- recall
- F1
- false-positive rate
- first-detection event latency

Also record:
- number of runs
- number of events
- workload period
- model/agent versions
- tool versions
- detector configuration
- calibration size
- holdout size
- missing-data rate
- redaction method
- storage used
- monitoring overhead

Where wall-clock timestamps are available, also measure elapsed detection time. The current validator reports event-position latency; wall-clock latency should be measured separately from the raw trace.

## 6. Test conditions

The pilot should contain normal runs as well as deliberately introduced failure cases where it is safe to do so.

Useful controlled cases include:

### Action loops
- repeated identical tool call
- repeated short sequence
- tool retry that is actually legitimate
- long but valid repeated workflow
- loop caused by a tool returning an unexpected result

### Goal drift
- output clearly outside the stated task
- partially relevant output
- legitimate change of subtask
- legitimate additional research
- output containing financial terminology but unrelated to the actual goal

### Resource spikes
- unusually long run
- unusually high token consumption
- unusually many tool calls
- slow external tool
- legitimate large workload

The last cases are important because they create realistic negative examples.

## 7. False positives matter

For a finance deployment, do not focus only on catching abnormal runs.

Record examples where DriftShield raised an alert but the reviewer judged the behaviour legitimate.

For every false positive, capture:
- detector
- run identifier
- workload type
- reason the behaviour was legitimate
- threshold/configuration at the time
- whether the threshold should change

This gives the pilot a useful failure-analysis record rather than only a single headline metric.

## 8. Data handling

The validation dataset should have three forms:

    raw traces
        |
        +--> secure internal archive
        |
        +--> redaction
                 |
                 v
          labelled pilot dataset
                 |
                 v
           validation report

Never commit the raw financial traces to this repository.

The existing prepare_pilot_dataset.py utility removes common secret-bearing fields and secret-like values, but it does not guarantee anonymisation. A financial-services pilot should apply the organisation's own data-classification and redaction process before using the utility.

## 9. Evidence package

For each pilot, retain:
1. pilot description
2. detector configuration
3. calibration dataset description
4. holdout dataset description
5. independent labelling procedure
6. validation JSON
7. generated evidence report
8. false-positive/false-negative review
9. overhead measurements
10. known limitations

The final report should state exactly what was tested and what was not tested.

Do not describe the result as “finance validated” unless the evidence actually comes from a financial-services workload.

## 10. Acceptance criteria

Do not set arbitrary accuracy targets before seeing the workload.

First establish the measured baseline for each detector. Then agree target thresholds with the pilot owner based on the operational consequences of missed detections and false alerts.

A useful pilot outcome is therefore not simply a single F1 value.

It should answer:
- Which behaviours were detected reliably?
- Which legitimate behaviours generated alerts?
- Which abnormal behaviours were missed?
- How quickly were problems detected?
- How much monitoring overhead was introduced?
- Which thresholds were used?
- What changed after reviewing false positives?
- What remains untested?

## 11. External context

UK financial regulators are currently focusing on practical AI testing, monitoring, governance and risk management. The FCA's AI Live Testing work explicitly includes evaluation frameworks, live monitoring and risk management, while its 2026 AI guidance describes oversight, model testing and outcome monitoring as active industry questions.

This document does not turn DriftShield into a regulatory compliance tool. It uses those themes only to make the pilot evidence more useful to a prospective financial-services customer.

## 12. Current DriftShield position

The repository already provides:
- deterministic synthetic detector tests
- labelled-trace validation
- pilot-data redaction
- baseline robustness checks
- controlled local overhead measurement
- evidence-report generation
- external behavioural-trace analysis

The missing evidence is the important part: an independently labelled, representative financial-services workload.

Until that exists, keep the product description at alpha/source-release level and do not claim production financial-services detection accuracy.