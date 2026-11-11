"""Was this Ruben? Style-based player detection.

For every move a side played we compare how likely it is under Ruben's
fine-tuned model versus the generic Maia-2 base model:

    score = mean over moves of [ log p_ruben(move) - log p_base(move) ]

Moves Ruben would typically play get a positive score; moves typical for
"a generic player of that rating" get ~0 or negative. The decision
threshold is calibrated on held-out games (`mac-detect calibrate`), using
Ruben's own moves as positives and his opponents' moves as negatives.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import COLUMNS, iter_games, positions_from_game
from .finetune import PositionDataset, masked_logits
from .model import load_base, load_checkpoint

MIN_MOVES = 10


@torch.no_grad()
def move_logprobs(model, df: pd.DataFrame) -> np.ndarray:
    ds = PositionDataset(df)
    out = []
    model.eval()
    for boards, labels, es, eo, mask, _ in DataLoader(ds, batch_size=512, collate_fn=ds.collate):
        logits, _, _ = model(boards, es, eo)
        lp = F.log_softmax(masked_logits(logits, mask), dim=1)
        out.append(lp.gather(1, labels[:, None]).squeeze(1))
    return torch.cat(out).numpy() if out else np.zeros(0)


def side_positions(game: chess.pgn.Game, color: str) -> pd.DataFrame:
    name = game.headers.get("White" if color == "white" else "Black", "?")
    rows = positions_from_game(game, "check", aliases=(name,))
    rows = [r for r in rows if r["color"] == color]  # same name on both sides edge case
    return pd.DataFrame(rows, columns=COLUMNS)


def score_side(game, color: str, ruben, base) -> tuple[float, int]:
    df = side_positions(game, color)
    if len(df) == 0:
        return float("nan"), 0
    llr = move_logprobs(ruben, df) - move_logprobs(base, df)
    return float(llr.mean()), len(df)


def verdict(score: float, n: int, threshold: float) -> str:
    if n < MIN_MOVES or np.isnan(score):
        return "te weinig zetten om te beoordelen"
    margin = score - threshold
    if margin > 0.15:
        return "waarschijnlijk Ruben"
    if margin > 0:
        return "mogelijk Ruben"
    if margin > -0.15:
        return "waarschijnlijk niet Ruben"
    return "zeer waarschijnlijk niet Ruben"


def load_models(model_path: Path, maia_root: str = "maia2_models"):
    ruben, meta = load_checkpoint(model_path)
    base, _ = load_base(meta.get("base_model", "rapid"), maia_root)
    return ruben, base, meta


def check(pgn_text: str, model_path: Path, detector_path: Path, maia_root: str = "maia2_models") -> list[dict]:
    ruben, base, _ = load_models(model_path, maia_root)
    threshold = json.loads(detector_path.read_text())["threshold"] if detector_path.exists() else 0.0
    results = []
    fh = io.StringIO(pgn_text)
    while (game := chess.pgn.read_game(fh)) is not None:
        for color in ("white", "black"):
            s, n = score_side(game, color, ruben, base)
            results.append({
                "game": f"{game.headers.get('White', '?')} vs {game.headers.get('Black', '?')} ({game.headers.get('Date', '?')})",
                "side": color, "player": game.headers.get(color.capitalize(), "?"),
                "moves": n, "score": round(s, 3), "verdict": verdict(s, n, threshold),
            })
    return results


def calibrate(raw_dir: Path, positions_csv: Path, model_path: Path, maia_root: str = "maia2_models") -> dict:
    """Score Ruben's and his opponents' moves in the held-out games and pick the best threshold."""
    ruben, base, _ = load_models(model_path, maia_root)
    val_ids = set(pd.read_csv(positions_csv, usecols=["game_id", "split"]).query("split == 'val'").game_id)
    pos, neg = [], []
    for path in sorted(raw_dir.rglob("*.pgn")):
        for game in iter_games(path):
            mine = positions_from_game(game, "x")
            if not mine or mine[0]["game_id"] not in val_ids:
                continue
            my_color = mine[0]["color"]
            opp_color = "black" if my_color == "white" else "white"
            for color, bucket in ((my_color, pos), (opp_color, neg)):
                s, n = score_side(game, color, ruben, base)
                if n >= MIN_MOVES:
                    bucket.append(s)
    pos_a, neg_a = np.array(pos), np.array(neg)
    if len(pos_a) == 0 or len(neg_a) == 0:
        return {"threshold": 0.0, "accuracy": float("nan"), "auc": float("nan"),
                "ruben_sides": len(pos_a), "opponent_sides": len(neg_a)}
    candidates = np.unique(np.concatenate([pos_a, neg_a]))
    accs = [((pos_a > t).sum() + (neg_a <= t).sum()) / (len(pos_a) + len(neg_a)) for t in candidates]
    best = int(np.argmax(accs))
    # AUC: probability a random Ruben-side scores above a random opponent-side
    auc = float((pos_a[:, None] > neg_a[None, :]).mean() + 0.5 * (pos_a[:, None] == neg_a[None, :]).mean())
    return {
        "threshold": float(candidates[best]), "accuracy": float(accs[best]), "auc": auc,
        "ruben_sides": len(pos_a), "opponent_sides": len(neg_a),
        "ruben_mean": float(pos_a.mean()), "opponent_mean": float(neg_a.mean()),
        "recall_ruben": float((pos_a > candidates[best]).mean()),
        "false_positive_rate": float((neg_a > candidates[best]).mean()),
    }


def latest_chesscom_games(user: str, n: int) -> str:
    import requests

    from .fetch_chesscom import _get, archive_months

    session = requests.Session()
    games: list[str] = []
    for url in reversed(archive_months(session, user)):
        fh = io.StringIO(_get(session, url + "/pgn").text)
        month = []
        while (g := chess.pgn.read_game(fh)) is not None:
            month.append(str(g))
        games = month + games
        if len(games) >= n:
            break
    return "\n\n".join(games[-n:])


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="score one or more games (PGN file or '-' for stdin)")
    c.add_argument("pgn", nargs="?", default=None, help="PGN file, or '-' for stdin")
    c.add_argument("--chesscom-user", default=None, help="check this chess.com user's latest games instead")
    c.add_argument("--last", type=int, default=5)
    c.add_argument("--model", type=Path, default=Path("models/me_at_chess.pt"))
    c.add_argument("--detector", type=Path, default=Path("models/detector.json"))
    k = sub.add_parser("calibrate", help="pick a threshold on held-out games")
    k.add_argument("--raw", type=Path, default=Path("data/raw"))
    k.add_argument("--positions", type=Path, default=Path("data/processed/positions.csv.gz"))
    k.add_argument("--model", type=Path, default=Path("models/me_at_chess.pt"))
    k.add_argument("--out", type=Path, default=Path("models/detector.json"))
    args = p.parse_args(argv)

    if args.cmd == "check":
        if args.chesscom_user:
            text = latest_chesscom_games(args.chesscom_user, args.last)
        elif args.pgn in (None, "-"):
            text = sys.stdin.read()
        else:
            text = Path(args.pgn).read_text(encoding="utf-8")
        rows = check(text, args.model, args.detector)
        print("| partij | kant | speler | zetten | score | oordeel |\n|---|---|---|---|---|---|")
        for r in rows:
            print(f"| {r['game']} | {r['side']} | {r['player']} | {r['moves']} | {r['score']} | **{r['verdict']}** |")
    else:
        res = calibrate(args.raw, args.positions, args.model)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
