# Evaluation datasets

Files in this directory are development artifacts and never enter a submission.

`v4_dev_50.*` is the original 50-position pipeline smoke test. It is retained
for audit history but rejected as training evidence: it has no validation split,
uses only shallow opening labels, and its fitted weights are not wired into an
agent. Schema-v2 tools intentionally refuse to load it.

A promotable fit must come from the residual-evaluation workflow in
`docs/EVALUATION_TUNING.md`, with immutable development, validation and holdout
splits and enough diverse positions to support every fitted feature.
