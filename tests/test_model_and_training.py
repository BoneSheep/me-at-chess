"""Uses a tiny randomly-initialised Maia-2 so CI never downloads real weights."""

import io
from pathlib import Path

import chess

from me_at_chess.dataset import build, split_by_game
from me_at_chess.finetune import train, write_report
from me_at_chess.model import load_base, load_checkpoint, predict
from me_at_chess.uci import Engine

FIX = Path(__file__).parent / "fixtures"


def test_predict_returns_legal_moves_for_both_colours():
    model, _ = load_base("tiny")
    board = chess.Board()
    probs = predict(model, board, 1500, 1500)
    assert set(probs) == {m.uci() for m in board.legal_moves}
    assert abs(sum(probs.values()) - 1) < 1e-4
    board.push_uci("e2e4")
    probs = predict(model, board, 1500, 1500)
    assert set(probs) == {m.uci() for m in board.legal_moves}  # black moves, unmirrored


def test_finetune_learns_on_tiny_data_and_saves(tmp_path):
    df, _ = build(FIX)
    df = split_by_game(df, val_frac=0.34)
    df["split"] = "train"  # overfit check: evaluate on the training games themselves
    val = df.copy()
    val["split"] = "val"
    import pandas as pd
    both = pd.concat([df, val])
    out = tmp_path / "m.pt"
    result = train(both, "tiny", out, epochs=30, lr=3e-3, batch_size=8, patience=30, log=lambda *_: None)
    assert result["best"]["acc"] > result["base"]["acc"]
    write_report(result, tmp_path / "r.md")
    assert "move-match" in (tmp_path / "r.md").read_text()

    model, meta = load_checkpoint(out)
    assert meta["elo_default"] > 1000
    # after overfitting, 1.e4 (BoneSheep's opening in the fixtures) should rank high
    top3 = list(predict(model, chess.Board(), meta["elo_default"], 1450))[:3]
    assert "e2e4" in top3


def test_uci_protocol():
    model, _ = load_base("tiny")
    eng = Engine(model)
    out = io.StringIO()
    for line in ["uci", "isready", "setoption name Temperature value 0.7",
                 "position startpos moves e2e4 e7e5", "go wtime 1000 btime 1000"]:
        assert eng.handle(line, out)
    text = out.getvalue()
    assert "uciok" in text and "readyok" in text
    best = text.strip().splitlines()[-1].split()[1]
    board = chess.Board()
    board.push_uci("e2e4"); board.push_uci("e7e5")
    assert chess.Move.from_uci(best) in board.legal_moves
    assert eng.temperature == 0.7
    assert not eng.handle("quit", out)
