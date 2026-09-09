# Current champion: KingNet75 plus qsearch evaluation caching

This is the canonical deployable engine. Its exact promotion candidate is retained at
`challengers/exp_kingnet75_qcache`; do not edit that evidence snapshot.

The search retains V7's completed-iteration timeout safety and continuous clock allocation. Its
learned evaluator is a team-trained, 16-bucket king-conditioned network with a 128-wide
accumulator and 32-wide hidden layer. It supplies 75% of the blended static score; the existing
handcrafted evaluator supplies 25%.

The champion also includes the exact 65,536-entry direct-mapped cache for blended static
evaluations reached by quiescence search. Fixed-node testing against raw V9 retained the same
search result. In 100 development pairs against raw V9, the combined engine scored 54.0% with no
technical failures.

Against the previous canonical V7/qcache champion, this exact combined candidate scored 71 wins,
35 draws, and 16 losses over 61 development pairs: 72.54%, approximately +169 Elo. The integrated
0-versus-20 Elo pentanomial SPRT accepted the positive hypothesis with zero failures.

The exact model-training provenance must remain available to the team before upload: dataset and
trainer identity, command/configuration, selected checkpoint, and model hash.

Build the upload artifact from the repository root with:

```bash
./.venv/Scripts/python.exe -m harness.package --root current --out submission.zip
```
