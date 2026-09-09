# Search V10 S1 lean

This challenger is a frozen copy of reviewed Search V10 with two independently
switchable move-selection experiments implemented by this team:

- signed butterfly history with bounded gravity updates and maluses for quiet
  alternatives actually searched before a quiet beta cutoff;
- conservative quiet SEE pruning at shallow non-PV nodes.

Quiet SEE never prunes the first move, checks, castling, killers, or quiets with
strong positive history. Its margin grows quadratically with depth. History V2
uses separate positive and negative thresholds for move ordering and contextual
LMR; the original V10 history behaviour remains available unchanged.

Compile-time diagnostic profiles:

- `baseline`: reviewed V10 plus History V2 and quiet SEE;
- `v10`: exact reviewed-V10 control;
- `history-v2`: V10 plus signed history only;
- `quiet-see`: V10 plus quiet SEE only;
- `current`: all V10 and S1 mechanisms disabled.

This is an experiment, not the deployable champion. `current/` and
`challengers/exp_search_v10/` are untouched.
