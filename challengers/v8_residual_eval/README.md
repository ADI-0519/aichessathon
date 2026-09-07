# V8 strategic residual evaluator

This isolated evaluator challenger starts from `v6_stable_timeout`. Search,
clock allocation, NNUE weights, and stable completed-iteration handling are
unchanged. It adds a 24-channel tapered linear correction trained against the
exact V5 50% blended static evaluator on the existing 20,000 teacher-labelled
positions.

The model covers mobility, king-zone pressure, hanging pieces, safe space,
rook activity, blocked passers, minor-piece outposts, and king-file exposure.
Its selected integer fit reduces validation residual RMSE from 267.6 cp to
204.0 cp. That offline result is only an implementation gate: runtime feature
parity, search throughput, critical positions, and paired games decide whether
the challenger survives.
