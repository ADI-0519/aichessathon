# Phase-tapered learned-evaluation challenger

This isolated challenger starts from the selected V5-50 implementation. It
retains the team's learned evaluator and exact search, changing only how the
learned and handcrafted scores are blended in sparse positions.

The blend is intentionally controlled by explicit source constants in
`search.py`. This makes every arena result reproducible and prevents a hidden
environment setting from changing the submitted engine. `NNUE_BLEND` is the
full-material contribution. It tapers by material phase toward
`NNUE_ENDGAME_BLEND`, because the first ladder games showed that the simple
piece-square network was overconfident in sparse endings.

Do not promote this challenger until correctness checks, critical-position
probes, and paired games have all passed.

Run the implementation gate from the repository root:

```bash
./.venv/Scripts/python.exe -m tools.verify_v5_nnue \
  --candidate challengers/v5_phase
```
