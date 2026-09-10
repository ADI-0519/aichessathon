# Experiment: adaptive iterative-deepening time

This challenger is an isolated copy of the KingNet75/qcache champion. It retains the continuous
normal clock allocation and adds three reserve-safe deadlines:

- stable root move and score across completed iterations may stop at the soft deadline;
- ordinary positions stop at the normal deadline;
- move changes, large score changes, or aspiration failures may continue toward the hard deadline.

The hard timer remains authoritative, and interrupted iterations are never returned. The policy
also avoids starting a likely-unfinishable iteration based on the preceding iteration's cost.
Fixed-node callers that do not supply adaptive deadlines retain champion behaviour.
