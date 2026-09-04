from __future__ import annotations

import random
import unittest

import chess
import numpy as np

from challengers.numba_v1 import engine


class NumbaBoardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        engine.warmup()

    def test_starting_position_perft(self) -> None:
        position = engine.position_from_board(chess.Board())
        expected = (1, 20, 400, 8_902, 197_281, 4_865_609)
        for depth, nodes in enumerate(expected):
            with self.subTest(depth=depth):
                self.assertEqual(engine.perft(position, depth), nodes)

    def test_en_passant_reference_perft(self) -> None:
        # Chess Programming Wiki perft position 3 exercises checks, rook rays,
        # en passant legality, and king confinement.
        position = engine.position_from_fen(
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
        )
        expected = (1, 14, 191, 2_812, 43_238, 674_624)
        for depth, nodes in enumerate(expected):
            with self.subTest(depth=depth):
                self.assertEqual(engine.perft(position, depth), nodes)

    def test_special_position_move_sets_match_python_chess(self) -> None:
        fens = (
            chess.STARTING_FEN,
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
            "4k3/P6P/8/8/8/8/p6p/4K3 w - - 0 1",
            "4r1k1/8/8/8/8/8/4R3/4K3 w - - 0 1",
            "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/2pP4/1p2P3/2N2N2/PPQ1BPPP/R3K2R w KQkq - 0 1",
        )
        for fen in fens:
            with self.subTest(fen=fen):
                board = chess.Board(fen)
                expected = {move.uci() for move in board.legal_moves}
                actual = engine.legal_moves_uci(engine.position_from_board(board))
                self.assertEqual(actual, expected)

    def test_random_move_sets_match_python_chess(self) -> None:
        rng = random.Random(2026090401)
        board = chess.Board()
        checked = 0
        while checked < 1_000:
            if board.is_game_over() or len(board.move_stack) >= 180:
                board.reset()
            expected = {move.uci() for move in board.legal_moves}
            actual = engine.legal_moves_uci(engine.position_from_board(board))
            self.assertEqual(actual, expected, board.fen())
            checked += 1
            board.push(rng.choice(list(board.legal_moves)))

    def test_random_attack_maps_match_python_chess(self) -> None:
        rng = random.Random(2026090403)
        board = chess.Board()
        checked = 0
        while checked < 200:
            if board.is_game_over() or len(board.move_stack) >= 160:
                board.reset()
            position = engine.position_from_board(board)
            for square in range(64):
                self.assertEqual(
                    engine.is_square_attacked(position.pieces, square, engine.WHITE),
                    board.is_attacked_by(chess.WHITE, square),
                    (board.fen(), chess.square_name(square), "white"),
                )
                self.assertEqual(
                    engine.is_square_attacked(position.pieces, square, engine.BLACK),
                    board.is_attacked_by(chess.BLACK, square),
                    (board.fen(), chess.square_name(square), "black"),
                )
            checked += 1
            board.push(rng.choice(list(board.legal_moves)))

    def test_zobrist_uses_canonical_position_identity(self) -> None:
        base_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        phantom_ep = base_fen.replace(" - 0 1", " e3 17 42")
        base = engine.position_from_fen(base_fen)
        phantom = engine.position_from_fen(phantom_ep)
        self.assertFalse(engine.has_legal_en_passant(phantom.pieces, phantom.state))
        self.assertEqual(int(base.key[0]), int(phantom.key[0]))

        legal_ep = engine.position_from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
        no_ep = engine.position_from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - - 0 1")
        self.assertTrue(engine.has_legal_en_passant(legal_ep.pieces, legal_ep.state))
        self.assertNotEqual(int(legal_ep.key[0]), int(no_ep.key[0]))

        pinned_ep = engine.position_from_fen("k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
        pinned_no_ep = engine.position_from_fen("k3r3/8/8/3pP3/8/8/8/4K3 w - - 0 1")
        self.assertFalse(engine.has_legal_en_passant(pinned_ep.pieces, pinned_ep.state))
        self.assertEqual(int(pinned_ep.key[0]), int(pinned_no_ep.key[0]))

        black_to_move = engine.position_from_fen(base_fen.replace(" b ", " w "))
        no_castling = engine.position_from_fen(base_fen.replace(" KQkq ", " - "))
        self.assertNotEqual(int(base.key[0]), int(black_to_move.key[0]))
        self.assertNotEqual(int(base.key[0]), int(no_castling.key[0]))

    def test_en_passant_hash_presence_matches_python_chess(self) -> None:
        rng = random.Random(2026090405)
        board = chess.Board()
        for _ in range(1_000):
            if board.is_game_over() or len(board.move_stack) >= 180:
                board.reset()
            position = engine.position_from_board(board)
            self.assertEqual(
                engine.has_legal_en_passant(position.pieces, position.state),
                board.has_legal_en_passant(),
                board.fen(en_passant="fen"),
            )
            board.push(rng.choice(list(board.legal_moves)))

    def test_insufficient_material_matches_python_chess(self) -> None:
        fens = (
            "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            "4k3/8/8/8/8/8/8/2B1K3 w - - 0 1",
            "4k3/8/8/8/8/8/8/2N1K3 w - - 0 1",
            "4k3/8/8/8/8/8/8/1NN1K3 w - - 0 1",
            "4k3/8/8/8/8/8/2b5/2B1K3 w - - 0 1",
            "4k3/8/8/8/8/8/1b6/2B1K3 w - - 0 1",
            "4k3/8/8/8/8/8/2n5/2B1K3 w - - 0 1",
            "4k3/8/8/8/8/8/8/3RK3 w - - 0 1",
        )
        for fen in fens:
            with self.subTest(fen=fen):
                board = chess.Board(fen)
                position = engine.position_from_board(board)
                self.assertEqual(
                    engine.is_insufficient_material(position.pieces),
                    board.is_insufficient_material(),
                )

    def test_repetition_and_rule_fifty_boundaries(self) -> None:
        key = np.uint64(123)
        other = np.uint64(456)
        twice = np.array((key, other, key), dtype=np.uint64)
        three_times = np.array((key, other, key, other, key), dtype=np.uint64)
        self.assertFalse(engine.is_repetition_draw(key, twice, 3, 3, 8))
        self.assertTrue(engine.is_repetition_draw(key, three_times, 5, 5, 8))
        self.assertTrue(engine.is_repetition_draw(key, three_times, 5, 3, 8))
        self.assertFalse(engine.is_repetition_draw(key, three_times, 5, 5, 1))

        position = engine.position_from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 99 1")
        history = np.array((position.key[0],), dtype=np.uint64)
        self.assertFalse(
            engine.has_rule_draw(
                position.pieces, position.state, position.key[0], history, 1, 1
            )
        )
        position.state[engine.STATE_HALFMOVE] = 100
        self.assertTrue(
            engine.has_rule_draw(
                position.pieces, position.state, position.key[0], history, 1, 1
            )
        )

    def test_make_unmake_matches_python_chess_and_restores_every_field(self) -> None:
        rng = random.Random(2026090402)
        board = chess.Board()
        checked = 0
        while checked < 500:
            if board.is_game_over() or len(board.move_stack) >= 160:
                board.reset()
            position = engine.position_from_board(board)
            moves = engine.legal_moves(position)
            move = int(moves[rng.randrange(len(moves))])
            chess_move = chess.Move.from_uci(engine.move_to_uci(move))

            original_pieces = position.pieces.copy()
            original_state = position.state.copy()
            original_key = position.key.copy()
            undo = np.empty(engine.UNDO_SIZE, dtype=np.int64)
            undo_key = np.empty(1, dtype=np.uint64)
            self.assertTrue(
                engine.make_move(
                    position.pieces, position.state, position.key, move, undo, undo_key
                )
            )

            expected = board.copy(stack=False)
            expected.push(chess_move)
            actual = engine.board_from_position(position)
            self.assertEqual(
                actual.fen(en_passant="fen"),
                expected.fen(en_passant="fen"),
            )
            self.assertEqual(
                int(position.key[0]),
                int(engine.recompute_zobrist(position.pieces, position.state)),
            )

            engine.unmake_move(
                position.pieces, position.state, position.key, move, undo, undo_key
            )
            np.testing.assert_array_equal(position.pieces, original_pieces)
            np.testing.assert_array_equal(position.state, original_state)
            np.testing.assert_array_equal(position.key, original_key)

            board.push(chess_move)
            checked += 1

    def test_promotion_encodings_round_trip(self) -> None:
        board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
        moves = engine.legal_moves_uci(engine.position_from_board(board))
        self.assertTrue({"a7a8q", "a7a8r", "a7a8b", "a7a8n"}.issubset(moves))

    def test_special_moves_update_complete_state(self) -> None:
        cases = (
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 7 20", "e1g1"),
            ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"),
            ("1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7b8q"),
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "a1a8"),
        )
        for fen, uci in cases:
            with self.subTest(fen=fen, move=uci):
                board = chess.Board(fen)
                position = engine.position_from_board(board)
                packed = next(
                    int(move)
                    for move in engine.legal_moves(position)
                    if engine.move_to_uci(int(move)) == uci
                )
                undo = np.empty(engine.UNDO_SIZE, dtype=np.int64)
                undo_key = np.empty(1, dtype=np.uint64)
                original_pieces = position.pieces.copy()
                original_state = position.state.copy()
                original_key = position.key.copy()

                self.assertTrue(
                    engine.make_move(
                        position.pieces,
                        position.state,
                        position.key,
                        packed,
                        undo,
                        undo_key,
                    )
                )
                board.push_uci(uci)
                rebuilt = engine.board_from_position(position)
                self.assertEqual(
                    rebuilt.fen(en_passant="fen"),
                    board.fen(en_passant="fen"),
                )
                self.assertEqual(
                    int(position.key[0]),
                    int(engine.recompute_zobrist(position.pieces, position.state)),
                )

                engine.unmake_move(
                    position.pieces,
                    position.state,
                    position.key,
                    packed,
                    undo,
                    undo_key,
                )
                np.testing.assert_array_equal(position.pieces, original_pieces)
                np.testing.assert_array_equal(position.state, original_state)
                np.testing.assert_array_equal(position.key, original_key)


if __name__ == "__main__":
    unittest.main()
