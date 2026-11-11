from pathlib import Path

import pandas as pd

from me_at_chess.dataset import build, split_by_game
from me_at_chess.detect import calibrate, check, verdict
from me_at_chess.finetune import train

FIX = Path(__file__).parent / "fixtures"


def test_verdict_thresholds():
    assert verdict(0.5, 20, 0.0) == "waarschijnlijk Ruben"
    assert verdict(-0.5, 20, 0.0) == "zeer waarschijnlijk niet Ruben"
    assert "te weinig" in verdict(1.0, 3, 0.0)


def test_check_and_calibrate_run_end_to_end(tmp_path):
    df, _ = build(FIX)
    df = split_by_game(df, val_frac=0.34)
    pos_csv = tmp_path / "positions.csv.gz"
    df.to_csv(pos_csv, index=False, compression="gzip")
    model = tmp_path / "m.pt"
    train(df, "tiny", model, epochs=2, lr=1e-3, batch_size=8, log=lambda *_: None)

    rows = check((FIX / "chesscom" / "2024-01.pgn").read_text(), model, tmp_path / "missing.json")
    assert {r["side"] for r in rows} == {"white", "black"}
    assert all("verdict" in r for r in rows)

    res = calibrate(FIX, pos_csv, model)
    # only one val game with 9 moves per side -> below MIN_MOVES, so nothing is scored
    assert res["ruben_sides"] == 0 or 0 <= res["accuracy"] <= 1
