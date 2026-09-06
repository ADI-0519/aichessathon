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
