# Experiment: dedicated qsearch tactical generation

This challenger is an isolated copy of the KingNet75/qcache champion. It changes only board move
generation used by non-check quiescence nodes:

- captures and promotions are generated directly in the champion's existing relative order;
- ordinary quiet moves and castling are not materialised;
- when no legal tactical move exists, a short-circuit legal-quiet probe preserves the distinction
  between stalemate and an ordinary quiet position.

This is intended to preserve the complete fixed-node search tree. Promotion requires exact
tactical-set and fixed-node parity, followed by repeatable single-process throughput evidence.
