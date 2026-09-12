# V16 Search + BigNet champion

This is the canonical promoted copy of
`challengers/exp_release_v16_search_bignet`. It combines:

- the V15 search stack: two-slot TT, capture history, countermoves,
  conservative singular extensions and hardened selective search;
- the team-trained 200M-position format-3 BigNet with a 256-wide
  king-conditioned accumulator, pairwise features and eight material heads;
- the evidence-backed 75% learned / 25% handcrafted evaluation blend;
- exact qsearch evaluation caching and post-pruning accumulator updates; and
- persistent board/repetition state, legal emergency fallback and adaptive
  completed-iteration timing.

The executable behavior and model match the frozen V16 challenger; only import
grouping and descriptive package metadata differ. No third-party engine code,
network, opening book, tablebase or opponent-specific move is included.

The finals V18 candidates remain separate experiments. In particular, this
directory does not contain their long-horizon clock, SearchMax settings or
30-second asynchronous initialization adapter.
