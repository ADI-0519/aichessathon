# KingNet V11-BIG challenger

This is an evaluator-development branch built from the frozen Search V10 code.
Its `nnue.py` accepts both the deployed format-2 KingNet model and the new,
self-describing format-3 V11-BIG export.

The checked-in weight is intentionally still the format-2 champion weight. That
makes this directory a runtime parity control; it is not evidence about V11-BIG
strength. A trained format-3 model must replace `weights/model.npz` in a copied
challenger before strength testing.

Format 3 validates and consumes all architecture metadata from the model:

- accumulator width and pairwise width;
- material `piece_head_map`;
- hidden/head dimensions;
- centipawn scale and dual output branches.

It also refreshes stale king-bucket accumulator rows before every incremental
update. This is necessary because a search or qeval-cache hit can skip evaluation
of the parent position.

Keep `current/` frozen while training and testing. Promote only after runtime
parity, critical-position, throughput, paired-development, independent-validation,
official-clock, and packaging gates described in `docs/KINGNET_v11.md`.
