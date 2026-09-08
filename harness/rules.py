"""Event constants. Canonical source: https://aichessathon.com/docs/rules.md"""

INIT_BUDGET_S = 90.0
BASE_MS = 120_000
INCREMENT_MS = 500
PLY_CAP = 600
STDOUT_CAP = 4096
STDERR_HEAD = 4096
STDERR_TAIL = 4096
MAX_UNZIPPED_BYTES = 50_000_000
WATCHDOG_GRACE_MS = 500
SMOKE_PLIES = 20

OPENINGS = (
    ("English Opening", "r1bqk2r/pp1pppbp/2n2np1/2p5/2P5/2N1PNP1/PP1P1PBP/R1BQK2R b KQkq - 0 6"),
    ("French Winawer", "r1b1k2r/pp2nppp/2n1p3/q1ppP3/P2P4/2P2N2/2PB1PPP/R2QKB1R b KQkq - 4 9"),
    ("Petroff Defence", "rnbqkb1r/pp3ppp/2p5/1B1p4/3Pn3/5N2/PPP2PPP/RNBQK2R w KQkq - 0 7"),
    ("Scotch Game", "r1bq1rk1/pppp1ppp/2n2n2/1Bb5/3NP3/2P5/PP3PPP/RNBQ1RK1 w - - 3 8"),
    ("Grunfeld Defence", "rnbqk2r/pp2ppbp/6p1/2p5/3PP3/2P1BN2/P4PPP/R2QKB1R b KQkq - 1 8"),
    ("French Classical", "r1bqk2r/pp1n1ppp/2n1p3/2bpP3/5P2/2NB4/PPP3PP/R1BQK1NR w KQkq - 0 8"),
    ("Sicilian Closed", "1rbqk1nr/pp2ppbp/2np2p1/2p5/P3P3/2NP2P1/1PP1NPBP/R1BQK2R b KQk - 0 7"),
    ("Sicilian Sveshnikov", "r1bqkb1r/pp3ppp/2np4/1N1Pp3/8/8/PPP2PPP/R1BQKB1R b KQkq - 0 8"),
)
