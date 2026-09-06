# V6 ordering challenger

This challenger is copied from the exact V5-50 candidate and changes only
quiet-move ordering:

- a persistent counter-move table records the quiet reply that most recently
  caused a beta cutoff after each opponent move;
- quiet moves searched unsuccessfully before a quiet cutoff receive a history
  malus;
- TT moves, SEE-ranked captures, and killer moves retain higher priority.

The counter table is bounded at 32 KiB (`2 x 64 x 64` int32 entries). Both it
and quiet history persist for one game and are cleared when the agent detects a
new game.

This experiment must be tested directly against the immutable V5-50 candidate.
Do not combine it with the phase-taper experiment until each change has passed
independently.
