# Current champion: V7 continuous time plus qsearch evaluation caching

This is the canonical deployable engine. Its frozen source ancestor is
`challengers/v7_continuous_time`; do not edit that archived challenger.

V7 builds on `challengers/v6_stable_timeout` and changes only clock allocation. It removes the
60-second and 10-second discontinuities, uses the FEN fullmove number for a bounded moves-to-go
estimate, and models only part of the fixed increment. A hard reserve and low-clock taper remain.

The schedule spends more of the clock in late middlegames without changing evaluation, search,
persistent state, or the stable-timeout policy. V7 scored 65.0% against V6 in its development
screen, 48.75% in independent validation, and 50.0% directly against submitted V5. It is retained
for two causal reliability fixes rather than a proven general Elo gain.

The champion also includes an exact 65,536-entry direct-mapped cache for blended static
evaluations reached by quiescence search. It preserves fixed-node search results exactly. Across
the clean six-position suite it improved median NPS by 7.50% at 100k nodes, 4.49% at 300k, and
5.49% at one million; its 20-pair timed screen scored 51.25% with zero technical failures.

Build the upload artifact from the repository root with:

```bash
./.venv/Scripts/python.exe -m harness.package --root current --out submission.zip
```
