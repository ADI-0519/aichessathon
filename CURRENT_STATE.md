# CURRENT ENGINE STATE
Updated: 7 Sep 2026

## Deployable baseline

V7 continuous time

Source:
challengers/v7_continuous_time/

Submission artifact:
submission_v7.zip

V7 lineage:
V5 NNUE
→ V6 stable completed-depth timeout
→ V7 continuous time allocation

## IMPORTANT

Repository root agent.py/engine.py/search.py are historical V3.
DO NOT treat repository root as current engine.
DO NOT package repository root.

## Current evidence

V7 vs V6 development:
20W 12D 8L — 65.0%

V7 vs V6 validation:
48.75%

Interpretation:
No proven general Elo superiority.
V7 retained because it contains two causally verified reliability/time fixes.

## Rejected

HalfKP-256:
offline improved, 35% games → REJECT

HalfKP-128:
small-screen ~50% → NOT PROMOTED

Residual HCE:
offline improved, failed critical probes and games → REJECT

Ordering:
40% → REJECT

## Current task

Build V7 diagnostic laboratory.

Do not change deployable V7.

Classify:
- evaluation
- NNUE
- LMR
- NMP
- persistent state
- depth

## Current rules

Canonical:
https://aichessathon.com/docs

120s + 0.5
90s init
1 CPU
2 GB RAM
600 plies → draw
50 MB
10 uploads/day