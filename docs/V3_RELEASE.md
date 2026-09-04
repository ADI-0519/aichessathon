# V3 compiled release

Built on 4 September 2026 from the promoted `challengers/numba_v1` engine.

## Submission artifact

- upload file: `submission_v3.zip` (also copied byte-for-byte to `submission.zip`);
- SHA-256: `913F850CCBC4336C8174CD35D6B921F99D2BA9768F375CB65CE870FED7203825`;
- compressed size: 16,621 bytes;
- expanded size: 81,217 bytes;
- root members: `agent.py`, `engine.py`, `search.py`;
- no weights, binaries, native extensions, or non-platform dependencies.

The previous Python artifact remains locally as `submission_v2_python.zip`, SHA-256
`9AA9D177E82EF144E35C9423F3CFF5BE4BF84459E5FE4166DE171A8D54EFC0A3`, and its readable source is
retained at `champions/python_v2/agent.py`.

## Evidence before promotion

- 100,000-position differential board campaign with exact legal moves, FEN transitions,
  incremental hashes, and undo restoration;
- 33 production and engine tests passing after root promotion;
- Ruff clean and strict mypy clean across the 19 relevant files;
- `+24 =6 -0` (90.0%) against the Python V2 champion over 30 paired games at 2,000+50 ms;
- `+3 =1 -0` (87.5%) against Python V2 over four paired games at the official 120,000+500 ms;
- `+9 =9 -12` (45.0%) against Stockfish 18 at 500 nodes/move over 30 paired games, compared with
  Python V2's `+3 =13 -14` (31.7%) on the same suite;
- all 34 retained PGNs reparsed with legal moves, expected starting FENs, matching results, and
  independently reproduced terminal conditions;
- the extracted V3 archive won smoke games as both White and Black through the wire-protocol
  harness, with no technical failure.

These local checks do not replace platform validation. The dashboard validation log is the final
authority after upload.
