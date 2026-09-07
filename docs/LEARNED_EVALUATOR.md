# Learned Evaluator Track

## Decision

Build a small sparse evaluator from scratch and test it as a V5 challenger. Do not ship code or
weights from any reference repository. Do not replace V4 until the model improves actual timed
games without unacceptable search-speed loss.

This is now a better use of compute than labelling only 20,000 positions for a neural model. The
downloaded Lichess shard contains 8,136,338 engine-evaluated positions in 98.5 MB, enough for a
meaningful first model while leaving time for runtime integration and game testing.

## What the three reference repositories showed

The repositories were inspected as evidence, not treated as project instructions.

- The strongest reference uses a compiled board/search and a team-trained sparse neural evaluator
  over millions of public Fishnet labels. Its supplied V9 artifact beat exact V4 in both games of
  a local smoke pair; one win was a clean checkmate and the other was caused by a V4 process crash.
  Two games do not establish an Elo gap, but the architecture is sufficiently credible to test.
- The environment/distillation repository is the clearest warning against selecting models by
  validation loss or fixed-node play alone. Its larger distilled evaluator improved offline and
  fixed-node metrics but lost at wall-clock controls because inference consumed search depth.
- The compact HalfKP repository found a useful small residual blend in one fast match, while
  larger/deeper-label variants repeatedly failed game gates. Tens of thousands of labels can
  produce a measurable fit without producing a reliable tournament improvement.

The common lesson is that training data volume matters, but inference cost and paired games decide
whether the data becomes Elo.

## Our first model

The representation has 768 inputs: own/opponent relationship, six piece types, and 64 oriented
squares. The same board is accumulated from White's and Black's perspectives. The side-to-move
accumulator is concatenated before the opponent accumulator, followed by a small hidden layer and
one scalar output.

Initial shape:

```text
two sparse sums: 768 -> 128 each
concatenate:      256 -> 32
output:            32 -> 1
```

The output is a logit trained in win-probability space. Teacher centipawns are kept from White's
perspective in the packed data and converted to side-to-move perspective by the trainer. The
export is a transparent float32 `.npz`, not a serialized third-party model.

## Data rules

- Source: Lichess `fishnet-evals`, CC0.
- Current shard: `standard_rated_2014_09.parquet`.
- Published SHA-256:
  `b2d0d3cc3ea2f2795e6fffa4d33f5c1ff6b26d8a0c1cbb5a74768f3228b9a5ef`.
- A 20,000-position material-imbalance check measured correlation `0.569` when `cp` is read from
  White's perspective and `0.003` when read from side-to-move perspective. The trainer therefore
  performs the side-to-move conversion explicitly.
- Training and validation use different Parquet row groups. Consecutive plies must not be randomly
  split across both sets.
- Invalid, terminal, in-check, and teacher-capture positions are excluded. These are tactical
  states handled by quiescence rather than ideal static-evaluation targets.
- Early opening positions are excluded at first to reduce duplicate opening bias.
- Data manifests record the exact source and packed-data hashes.

## Promotion gates

1. Verify Python feature encoding against the eventual Numba runtime on random legal positions.
2. Verify exported inference against PyTorch to tight numerical tolerance.
3. Measure static-evaluation throughput and full search nodes per second on the pinned suite.
4. Compare pure neural evaluation and conservative blends with the V4 handcrafted evaluation.
5. Run at least 20 fast development pairs against exact V4; reject technical failures immediately.
6. Confirm a survivor on an untouched validation split and against the fixed Stockfish benchmark.
7. Run official-clock games, package the exact candidate, and test the extracted archive.

Only after those gates should a model become `submission_v5.zip`. The model's validation MAE is a
debugging metric, not the promotion criterion.

## V5 integration status (6 September 2026)

- Packed 4,000,000 training positions and 456,556 positions from a disjoint validation row group.
- Trained for eight epochs; validation probability MSE improved monotonically from `0.010185` to
  `0.008191`. The exported model is 428,848 bytes with SHA-256
  `76336cbb0e2b270ab5e626e2f88201afef43972d0fd1a9bd056db8486b2614ef`.
- `challengers/v5_nnue` starts from the exact V4 submission search. It updates both 128-value
  accumulators for ordinary moves, captures, en passant, promotions, castling, and null moves.
- Integer inference uses scale 2,048. Across 506 deterministic positions, incremental rebuild
  error was exactly zero and the worst difference from the float PyTorch graph was 3.17 cp.
- On the development machine the warmed dense head measured about 895,000 evaluations/second.
  A 200,000-node tactical-position search measured about 206,000 nodes/second versus 265,000 for
  exact V4, a roughly 23% throughput cost rather than the original float path's 57% cost.
- The initial 25% blend completed a four-game technical smoke against exact V4 with no crash,
  illegal move, or flag and scored 4/4. This is encouraging but far below the sample needed for a
  strength conclusion.
- A zip built through the standard packager contained all Python sources and
  `weights/model.npz` at 529,756 bytes uncompressed. Import warm-up took about 31.6 seconds and an
  extracted-archive move was legal, within the platform's init allowance.
- The first long blend command exposed a local orchestration failure before chess began: NumPy
  processes were starting roughly 23--26 native threads and both V4 and V5 intermittently crashed
  at ply zero. V5 now fixes all common numerical thread pools to one before importing NumPy, and
  the runbook exports the same limits so the frozen V4 opponent is constrained too. A clean match
  from the previously failing opening then completed normally. The contaminated `t0` journals
  remain diagnostic evidence and must not be resumed; corrected candidates and outputs use `t1`.

The fixed-blend development screen is complete. Against the exact extracted V4 submission over
20 opening pairs, the 25%, 50%, and 100% blends scored 75.0%, 73.75%, and 65.0% respectively, with
no technical failures. The small difference between 25% and 50% against V4 was not decisive, so
they played a direct 20-pair match on fresh development positions. The 50% blend won that match
62.5% to 37.5%, again without a technical failure, and is the selected V5 candidate. Pure NNUE is
rejected: the trained evaluator is useful in combination with the handcrafted evaluation but has
not earned the right to replace it.

The selected 50% build must still pass archive verification, official-clock smoke games, an
untouched validation sample, and the fixed Stockfish benchmark before it becomes the final locked
build. An expedited ladder upload may precede the longer confirmations because the selected build
has already completed both development gates cleanly; V4 remains the rollback artifact.

## Endgame-taper experiment

The submitted V5 artifact remains the fixed 50% blend. Its first ladder games exposed a separate
failure mode in sparse queen, rook, and minor-piece endings: the learned evaluator could become
more confident while deeper search followed strategically losing plans. The next isolated
challenger therefore keeps the 50% full-material blend but tapers it to 15% as non-pawn material
disappears. This preserves the measured middlegame contribution while giving V4's tapered
handcrafted evaluation more control in the positions least represented by the network's inputs.

This is only a hypothesis until it passes the critical-position probes and paired games. It must
not replace the fixed V5 candidate during its held-out validation run.

## Held-out V5 result

The selected V5-50 artifact completed its 20-pair validation match against exact V4 with 30 wins,
5 draws, 5 losses, and no technical failures: 81.25% score and an estimated +255 Elo. The paired
95% score interval was 59.8% to 92.7%, so the result is positive despite the modest sample.

The five losses did not expose one common crash or illegal-search defect. Fixed-node Stockfish
review found separate errors involving central tension, rook placement, king activity, and passed
pawn control. Their first major swings now live in `benchmarks/suites/v5_validation_losses.json`.
Because those positions have been inspected, they are diagnostic regressions and must never be
presented as untouched validation evidence again.

At 50,000 nodes, the phase taper fixed the `Kh7` ladder case and retained two other reference
moves, but failed to repair the queen-conversion sequence and made two choices worse. It remains
an isolated arena experiment rather than a justified replacement for V5-50.

## Strategic residual pilot

A reproducible 20,000-position pilot was labelled from the balanced Lichess suite with Stockfish
at 10,000 nodes per position. The completed manifest records 11,995 development, 3,941 validation,
and 4,064 untouched holdout positions; 997 mate scores were excluded from fitting. The holdout has
not been inspected. The fitted 24-channel tapered linear correction selected ridge 100 and reduced
validation residual RMSE from 236.3 to 209.6 cp (11.3%) and MAE from 165.9 to 152.0 cp (8.4%)
against the handcrafted evaluator. Integer rounding was effectively lossless.

That offline improvement is promising but is not a promotion result. Recomputing the target
against the actual V5-50 blended static score showed an even larger gap before correction (267.6
cp RMSE), and a refitted full feature set reduced it to 204.1 cp. This also shows why a correction
must be fitted against V5 directly rather than blindly added to the earlier V4 target: the NNUE and
the strategic features are not independent.

The feature cost matters. On the V4 target, an intercept-only model reached 232.2 cp validation
RMSE, hanging-piece channels reached 226.3, the twelve cheaper tactical/activity channels reached
219.5, all non-mobility channels reached 213.5, and the full set reached 209.6. Mobility therefore
adds measurable information, but only a timed throughput measurement can say whether it earns its
leaf cost.

The fit does not directly cure the known failures. At one ply it favoured the reference only for
`dxc4`, was neutral on `Ra5`, and favoured the played error in the other three validation cases.
At 200,000 search nodes, unmodified V5 found `Kg8`, `Ra5`, and `dxc4`, but still missed `h6` and
`Rac1`. The next learned-evaluation experiment must therefore be a separate V5-residual challenger
with exact Python/Numba feature parity and a measured nodes-per-second gate. It must not replace V5
unless it wins timed development pairs; the two unresolved positions remain diagnostic cases, not
training or validation evidence.

## Search challenger results

The isolated V6 countermove/history-maluses challenger completed 20 development pairs against V5
at 10 seconds plus 100 ms. It scored 12 wins, 8 draws, and 20 losses (40.0%, approximately -70
Elo) with no technical failures. Its pentanomial result was four lost pairs, three 0.5-point pairs,
ten split pairs, three 1.5-point pairs, and no won pairs. The paired interval is too wide to prove
the heuristic is universally harmful, but this candidate has no promotion evidence and is
rejected in its current form.

The first phase-taper arena attempt did not produce chess evidence: its first game was lost during
initialization on a slow Colab CPU. The same machine took about 88 seconds to initialize ordinary
V5, leaving almost no margin under the 90-second contract. The phase challenger no longer invokes
the development-only engine warm-up, which redundantly compiled public perft and convenience
wrappers before compiling the real root search. On the development machine, the revised import
completed in 35.0 seconds and the first move incurred no deferred compilation. NNUE parity over
506 positions, focused unit tests, lint, and configured strict type checking all still pass. The
phase hypothesis therefore needs a fresh timed match; the initialization failure is not a loss to
V5 and must not be included in its chess score.

## V6 king-conditioned track

The two remaining validation failures stayed wrong at one million nodes: V5 still missed `h6` in
the connected-pawn ending and still chose the wrong rook instead of `Rac1`. Search depth alone is
therefore not the general fix. The next high-upside evaluator conditions every oriented
piece-square on the friendly king square. Both kings remain in the sparse piece set, yielding
49,152 possible features and preserving king-to-king geometry.

The implementation starts from an exact lift of V5 into every king bucket. Its 256-wide
accumulator reserves the first 128 channels for V5. The new feature channels start with small
random values but their outgoing dense columns start at zero, so epoch zero is exactly V5 while
the extra capacity can still receive gradients once the dense head is unfrozen. Training retains
the V5 checkpoint if validation probability loss never improves. SparseAdam updates the large
embedding while AdamW updates only the small dense portion.

The runtime export contains int16/int32 weights rather than redundant float arrays. The untrained
256-wide scaffold is 25,202,450 bytes. Across 507 deterministic and random positions, Python and
Numba features matched exactly, incremental updates matched full rebuilds exactly, and the maximum
fixed-point/reference difference was 0.66 cp. The verifier covers castling, en passant, promotion,
ordinary captures, king moves, and king captures.

On the pinned 200,000-node position, exact V5 searched 201,309 nodes/second and the lifted V6
runtime searched 166,340 nodes/second, retaining 82.6% of throughput. Both completed depth seven
with `Qxc2` and a +19 cp score, confirming fixed-node equivalence before training. Import completed
in 33.8 seconds locally and the first move incurred no deferred compilation. These results clear
the implementation gate, not the strength gate: only the team-trained export and timed games can
promote the candidate.

### First full V6 training result

The first full run trained on 4,000,000 positions and evaluated on the disjoint 456,556-position
validation set using an RTX 3060. Relative to the exact V5 lift, the selected epoch two reduced
validation probability MSE from 0.00819139 to 0.00791476, a 3.38% improvement. Validation MAE fell
from 176.87 to 173.46 cp and RMSE from 347.81 to 343.03 cp. Epochs three through six continued to
improve centipawn MAE and RMSE but gradually worsened probability MSE, while training loss kept
falling. The protected checkpoint therefore correctly exported epoch two rather than the final
epoch. The selected integer model has SHA-256
`170c72639f2da8581bb5e8002f8e712fc3e414538d2a79810d4d2877fa0d686a`.

The trained artifact passed the 507-position runtime verifier with zero feature or incremental
errors and a maximum fixed-point/reference difference of 0.64 cp. At 100,000 nodes, V5 and V6 both
found the deeper `Kg8` and `Ra5` corrections. Neither found `h6` or `Rac1`; V6 also selected the
played `Rfc8` error where V5 selected `dxc4`. The targeted regression pack therefore does not show
a net tactical/strategic cure, despite the broad offline improvement.

A one-pair technical smoke against exact V5 finished with one V6 win and one draw, no crashes,
flags, or illegal moves. That result clears the technical gate but is far too small for a strength
claim. The next authority is the pinned 20-pair development match against V5. More epochs, more
data, or hard-position fine-tuning must wait for that result; otherwise training changes and model
strength would be confounded.

Because epoch two was selected while the dense head was still frozen, all outgoing connections
from accumulator channels 129-256 are exactly zero. Those channels cannot affect the selected
model even though their sparse feature values are non-zero. `tools.prune_halfkp` verified that
condition and produced an exact 128-wide graph with hash
`47697ccf3bae83e9ad8ad5389cb831c2ba87d8af0abfd6f06bf384322bacb4f7`, reducing the model from
25,202,450 to 12,602,642 bytes. This is lossless structural pruning, not approximate magnitude
pruning. Runtime and fixed-node throughput were measured only after the active 256-wide arena
completed, avoiding clock contention.

### V6 development result and rejection

The 256-wide trained V6 completed 20 development pairs against exact V5 at 10 seconds plus 100 ms.
It scored 7 wins, 14 draws, and 19 losses: 35.0%, approximately -108 Elo, with no technical
failures. By colour it scored 30.0% as White and 40.0% as Black. The pentanomial distribution was
three 0-point pairs, eight 0.5-point pairs, seven split pairs, two 1.5-point pairs, and no 2-point
pairs. Equivalently, V6 lost 11 non-split opening pairs and won two; a two-sided sign test on those
13 pairs is approximately 0.022. The build is rejected even though the backtest's deliberately
conservative 95% score interval includes 50%.

All 40 games ended normally: 26 checkmates, ten threefold repetitions, three insufficient-material
draws, and one fifty-move draw. The failure is chess strength, not initialization, legality, or
clock reliability. V6 differed from V5 on the first move in only seven of 20 openings, so the loss
cannot be explained solely by unlucky initial choices. In the three double-loss pairs, offline
Stockfish analysis found repeated medium and large positional errors rather than one common crash
motif; examples include `Bd3`, `...Be6`, `Ne4`, `...Qc7`, `g5`, and early `...Rc8`.

The corrected 128-wide exact-pruned candidate passes the same 507-position verifier with zero
feature and incremental errors, the same live checksum, and the same 0.64 cp fixed-point bound.
Dense evaluation rises from roughly 477,000 to 1,011,000 calls/second. Across the five 100,000-node
regression probes it reduced aggregate search time from 2.783 to 2.266 seconds, an approximately
22.8% throughput gain over 256-wide V6, while producing identical nodes, scores, and moves. It is
still about 14.7% slower than V5 on those probes. This candidate remains unproven and must be
tested independently; it does not erase the 256-wide rejection.

The 128-wide full-agent smoke completed two development pairs with zero technical failures. It
scored three draws and one loss (37.5%), reproducing the same pair scores as 256-wide V6 on those
two openings. This is sufficient for runtime safety but provides no strength evidence. The next
bounded test is positions three through ten, which completes a ten-pair screen without spending
another hour on a candidate derived from a clearly rejected parent.

That ten-pair screen is now complete. Across the smoke and positions three through ten, the
128-wide candidate scored five wins, ten draws, and five losses: exactly 50.0%, with zero technical
or opponent failures. It scored 60.0% as White and 40.0% as Black. The pair distribution was four
0.5-point pairs, three split pairs, two 1.5-point pairs, and one 2-point pair, with no double-loss
pairs. The conservative pair-level 95% score interval remains very wide at 23.7%-76.3% (about
-204 to +204 Elo), so parity is not established.

The 256-wide candidate scored 37.5% on these same first ten openings. The pruned build improved
five pair scores, worsened one, and left four unchanged, gaining 2.5 game points in total. Most
games began with the same candidate move, so the difference is consistent with the 128-wide
runtime reaching different later search frontiers rather than a changed evaluator: pruning is
mathematically exact. This rescues the compact candidate from immediate rejection, but it does not
override the full 256-wide 20-pair loss or fix the new round-46 regression. Positions 11-20 are the
next strength gate if spare compute is available; the isolated stable-timeout experiment remains
the higher-priority local test.

### Rated rounds 43-47

V5 won rounds 43-45 and lost rounds 46-47, but the three wins were not clean strength evidence.
At 100,000 Stockfish 18 nodes per move, round 44 included an approximately 114 cp `fxe5` error
before an opponent blunder reversed the game. Round 43 was converted after the opponent's major
errors, while the 104-move round 45 win repeatedly leaked evaluation in a winning position. The
common signal remains unstable positional play rather than a legality, clock, or initialization
failure.

The first major round-46 error is `20.Be3` instead of `20.Be2`, an approximately 109 cp loss that
allows `...Bxf3` and damages the king shelter. Fresh V5 searches still choose `Be3` at 25,000,
100,000, 300,000, and one million nodes. A faithful fixed-node game replay also chooses `Be3`, and
the trained 128-wide HalfKP candidate chooses it through 300,000 nodes. This is an evaluator or
search-selectivity blind spot; more time and the rejected HalfKP model do not fix it.

Round 47 has a different cause. The rated `11...Qd7` instead of `11...Nd4` loses approximately
150 cp. An exact wall-clock replay reproduced every V5 move through `...Qd7` using the PGN clocks:
the target search stopped after about 829,000 nodes with depth eight complete. Isolating state at
that position showed that full persistent TT plus quiet history returned `Qd7`, while TT-only and
history-only runs both returned `Nd4`; a fresh run returned `Nbd7`. Capping the persistent search
at its last completed depth also returned `Nd4` after about 301,000 nodes. V5 currently promotes a
root move that raised alpha in an incomplete deeper iteration when the timer fires. Persistent
ordering can therefore turn a sound completed result into an unstable partial result.

The next search challenger should change only this timeout policy: preserve the move from the last
fully completed iteration and discard partial-depth candidates. It should retain the TT and quiet
history initially, because neither table alone reproduced the blunder. The round-46 and round-47
positions are checked in as `benchmarks/suites/v5_round46_47_losses.json`, with their exact game
histories in `benchmarks/suites/v5_round46_47_replays.json`. This surgical challenger takes
priority over further HalfKP training.

`challengers/v6_stable_timeout` implements exactly that change. Its agent, board implementation,
NNUE runtime, and model artifact hash-identically match V5; only the two interrupted-root branches
in `search_position` differ. In an exact rated-clock replay it reproduced V5's first four round-47
moves (`O-O`, `Bg4`, `Bxf3`, and `Nb6`) and then returned `Nd4` rather than `Qd7`, retaining the
same 3,669 ms budget, completed depth eight, -37 cp internal score, and persistent game state.
Ruff, standalone strict mypy, compilation, and the replay all pass. This establishes mechanism and
correctness, not Elo; paired games against frozen V5 are still required.

The two-pair technical smoke against exact V5 completed normally with two wins, one draw, and one
loss (62.5%). There were no candidate or opponent failures; three games ended by checkmate and one
by threefold repetition. The first pair remained identical through White's `Qc2`, after which the
stable challenger selected `...Bd6` while V5 selected `...Be7`; the challenger eventually won as
Black. In the second pair both games remained identical through `...Ndf6`, after which stable
timeout selected `O-O` while V5 selected `Ngf3`; that pair split one win each. This clears the
technical smoke gate but four games carry essentially no Elo confidence.

The remaining 18 development pairs also completed with no technical failures: 12 wins, 14 draws,
and ten losses (52.8%, about +19 Elo). Combined with the smoke, stable timeout scored 14 wins, 15
draws, and 11 losses over 20 pairs: 53.75%, about +26 Elo. The combined pentanomial distribution
was one 0-point pair, four 0.5-point pairs, seven split pairs, seven 1.5-point pairs, and one 2-point
pair. The pair-level 95% score interval is still broad (roughly 33%-73%), so this is evidence of
non-regression rather than proof of an Elo gain. Together with the exact round-47 repair and zero
failures, it promotes stable timeout to the development baseline; it does not yet justify a rated
upload by itself.

### Rated round 48 and continuous time management

Round 48 exposes a separate failure. V5 entered move 49 with approximately 41.5 seconds and an
equal ending, but the discontinuous clock policy allocated only 1.24 seconds. It completed depth
ten and played `49...Kg8`; independent Stockfish 18 analysis at 100,000 nodes identifies
`49...Kh7`, and the played move loses by force. The stable-timeout build also chooses `Kg8` when
capped at completed depth ten, so discarding incomplete iterations cannot repair this case.

The unchanged search chooses `Kh7` at depth 12 when allowed one million nodes. A wall-clock probe
at the budget assigned by the new policy reached depth 12 after about 931,000 nodes and also chose
`Kh7`. The position is preserved in `benchmarks/suites/v5_round48_loss.json`.

`challengers/v7_continuous_time` changes only time allocation on top of stable timeout. It removes
the 60-second and 10-second cliffs, reserves a bounded amount of clock, estimates fewer remaining
decisions later in the game, and credits only part of the increment. At the round-48 position it
allocates 3.35 seconds instead of V5's 1.24 seconds. Simulations through 150 decisions retain a
positive reserve at both the official 120+0.5 control and the 10+0.1 development control. This is
a targeted attempt to convert the large unused clocks seen in recent rated losses into completed
search depth; paired testing against exact stable timeout is still required before promotion.
