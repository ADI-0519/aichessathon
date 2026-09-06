# V5 learned-evaluation challenger

This challenger starts from the exact submitted V4 search and replaces part of
its leaf evaluation with a small network trained by the team from labelled
positions. `nnue.py` validates and loads `weights/model.npz`, keeps its two sparse
piece-square accumulators incrementally through search, and runs the dense head
with Numba on one CPU core.

The blend is intentionally an explicit source constant in `search.py`. This
makes every arena result reproducible and prevents a hidden environment setting
from changing the submitted engine. A blend of 0 is exact V4 evaluation; 100 is
the learned evaluation alone.

Do not promote this challenger from its directory until feature parity,
inference throughput, fixed-node regressions, and paired games have all passed.

Run the implementation gate from the repository root:

```bash
./.venv/Scripts/python.exe -m tools.verify_v5_nnue \
  --candidate challengers/v5_nnue
```
