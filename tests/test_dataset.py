from pathlib import Path

import chess

from me_at_chess.dataset import build, mirror_move, split_by_game, time_class

FIX = Path(__file__).parent / "fixtures"


def test_mirror_move():
    assert mirror_move("e7e5") == "e2e4"
    assert mirror_move("e8g8") == "e1g1"
    assert mirror_move("a2a1q") == "a7a8q"


def test_time_class():
    assert time_class("60") == "bullet"
    assert time_class("180+2") == "blitz"
    assert time_class("600") == "rapid"
    assert time_class("1800") == "classical"
    assert time_class("1/86400") == "daily"
    assert time_class("") == "classical"


def test_build_keeps_only_my_moves_and_skips_variants():
    df, stats = build(FIX)
    assert stats.games == 3
    assert stats.skipped_variant == 1
    assert stats.skipped_not_me == 1
    # game 1: white, 5 of my moves; game 2: black, 8; game 5: white, 9
    assert len(df) == 5 + 8 + 9
    assert set(df.color) == {"white", "black"}


def test_positions_are_white_to_move_and_moves_legal():
    df, _ = build(FIX)
    for fen, move in zip(df.fen, df.move):
        board = chess.Board(fen)
        assert board.turn == chess.WHITE
        assert chess.Move.from_uci(move) in board.legal_moves


def test_black_game_is_mirrored_and_scored():
    df, _ = build(FIX)
    g2 = df[df.game_id.str.endswith("/2")]
    assert (g2.color == "black").all()
    assert (g2.active_win == 1).all()  # I won with black
    assert (g2.elo_self == 1460).all() and (g2.elo_oppo == 1500).all()
    # my first move 1...d5 seen from my side is d2d4 on a mirrored board
    assert g2.iloc[0].move == "d2d4"


def test_split_is_chronological_by_game():
    df, _ = build(FIX)
    df = split_by_game(df, val_frac=0.34)
    val_games = set(df[df.split == "val"].game_id)
    assert val_games == {"https://www.chess.com/game/live/5"}
