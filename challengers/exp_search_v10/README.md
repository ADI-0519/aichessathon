# exp_search_v10_bundle

High-upside selective-search challenger built from the frozen KingNet75 + q-eval-cache champion.

Independent standard chess-search mechanisms (no third-party engine code):

- dynamic null-move reduction based on depth and eval margin, with deep verification;
- conservative reverse futility pruning;
- independently switchable shallow late-move and quiet-futility pruning;
- shallow capture SEE pruning;
- contextual LMR using depth, move index, PV status, history and killers.

Each V10 mechanism has a named source switch near the top of `search.py` and a compile-time
diagnostic profile. The `current` profile restores the frozen champion's fixed-R2 null move and
original LMR eligibility; it is an internal parity control, while `baseline` means the complete
V10 bundle. `current/` is untouched.
