"""Turn Ruben's PGNs into (position, move) training examples for Maia-2.

Only positions where *Ruben* is to move are kept: that's the behaviour we
want the bot to copy. Positions are stored the way Maia-2 consumes them:
always from the side-to-move's perspective (black positions are mirrored),
with the move in UCI notation, also mirrored.

Unlike Maia-2's own pipeline we keep the opening moves: your opening
repertoire is a big part of "playing like you".
"""

from __future__ import annotations

import argparse
import gzip
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn
import pandas as pd

from . import PLAYER_ALIASES

COLUMNS = [
    "game_id", "source", "date", "time_class", "color", "ply",
    "fen", "move", "elo_self", "elo_oppo", "active_win",
]


def mirror_move(uci: str) -> str:
    """Mirror a UCI move vertically (e2e4 -> e7e5), keeping promotion piece."""
    def sq(s: str) -> str:
        return s[0] + str(9 - int(s[1]))
    return sq(uci[0:2]) + sq(uci[2:4]) + uci[4:]


def time_class(tc: str) -> str:
    """Classify a PGN TimeControl the way chess.com does (est. = base + 40*inc)."""
    if not tc or tc in ("-", "?"):
        return "classical"  # OTB games usually have no TimeControl
    if "/" in tc:
        return "daily"
    try:
        base, _, inc = tc.partition("+")
        est = int(base) + 40 * int(inc or 0)
    except ValueError:
        return "unknown"
    if est < 180:
        return "bullet"
    if est < 600:
        return "blitz"
    if est < 1800:
        return "rapid"
    return "classical"


def _is_me(name: str, aliases=PLAYER_ALIASES) -> bool:
    n = name.strip().lower()
    return any(n == a.lower() for a in aliases)


def _elo(value: str | None, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


@dataclass
class GameStats:
    games: int = 0
    skipped_variant: int = 0
    skipped_not_me: int = 0


def positions_from_game(game: chess.pgn.Game, source: str, default_elo: int = 1500,
                        aliases=PLAYER_ALIASES) -> list[dict]:
    h = game.headers
    if _is_me(h.get("White", ""), aliases):
        color = chess.WHITE
    elif _is_me(h.get("Black", ""), aliases):
        color = chess.BLACK
    else:
        return []

    result = h.get("Result", "*")
    white_score = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}.get(result)
    if white_score is None:
        return []
    my_score = white_score if color == chess.WHITE else -white_score

    white_elo = _elo(h.get("WhiteElo"), default_elo)
    black_elo = _elo(h.get("BlackElo"), default_elo)
    elo_self, elo_oppo = (white_elo, black_elo) if color == chess.WHITE else (black_elo, white_elo)

    game_id = h.get("Link") or h.get("Site", "") + "|" + h.get("Date", "") + "|" + h.get("White", "") + "|" + h.get("Black", "") + "|" + h.get("Round", "")
    date = h.get("UTCDate") or h.get("Date", "????.??.??")
    tclass = time_class(h.get("TimeControl", ""))

    rows = []
    board = game.board()
    for ply, move in enumerate(game.mainline_moves()):
        if board.turn == color:
            if color == chess.WHITE:
                fen, uci = board.fen(), move.uci()
            else:
                fen, uci = board.mirror().fen(), mirror_move(move.uci())
            rows.append({
                "game_id": game_id, "source": source, "date": date,
                "time_class": tclass, "color": "white" if color else "black",
                "ply": ply, "fen": fen, "move": uci,
                "elo_self": elo_self, "elo_oppo": elo_oppo, "active_win": my_score,
            })
        board.push(move)
    return rows


def iter_games(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        while True:
            game = chess.pgn.read_game(fh)
            if game is None:
                return
            yield game


def build(raw_dir: Path, default_elo: int = 1500, aliases=PLAYER_ALIASES) -> tuple[pd.DataFrame, GameStats]:
    stats = GameStats()
    rows: list[dict] = []
    files = sorted(list(raw_dir.rglob("*.pgn")) + list(raw_dir.rglob("*.pgn.gz")))
    for path in files:
        source = path.parent.name  # e.g. "chesscom" or "otb"
        for game in iter_games(path):
            variant = game.headers.get("Variant", "Standard")
            if variant.lower() not in ("standard", "") or game.headers.get("SetUp") == "1":
                stats.skipped_variant += 1
                continue
            new = positions_from_game(game, source, default_elo, aliases)
            if not new:
                stats.skipped_not_me += 1
                continue
            stats.games += 1
            rows.extend(new)
    return pd.DataFrame(rows, columns=COLUMNS), stats


def split_by_game(df: pd.DataFrame, val_frac: float = 0.1) -> pd.DataFrame:
    """Chronological split: the most recent `val_frac` of games go to validation.

    Splitting by game (not by position) keeps near-duplicate positions of the
    same game out of both sides; splitting by time tests whether the model
    predicts how you play *now*.
    """
    df = df.copy()
    order = df.groupby("game_id")["date"].first().sort_values(kind="stable")
    n_val = max(1, int(round(len(order) * val_frac))) if len(order) > 1 else 0
    val_ids = set(order.index[len(order) - n_val:])
    df["split"] = ["val" if g in val_ids else "train" for g in df["game_id"]]
    return df


def summary(df: pd.DataFrame, stats: GameStats) -> str:
    if df.empty:
        return "No games found for Ruben."
    by_tc = df.groupby("time_class")["game_id"].nunique().sort_values(ascending=False)
    by_src = df.groupby("source")["game_id"].nunique()
    lines = [
        f"games used: {stats.games}  (skipped variants: {stats.skipped_variant}, not Ruben: {stats.skipped_not_me})",
        f"positions: {len(df)}  (train {int((df.split == 'train').sum())}, val {int((df.split == 'val').sum())})",
        "games by source: " + ", ".join(f"{k}={v}" for k, v in by_src.items()),
        "games by time class: " + ", ".join(f"{k}={v}" for k, v in by_tc.items()),
        f"rating range: {df.elo_self.min()}–{df.elo_self.max()} (median {int(df.elo_self.median())})",
        f"date range: {df.date.min()} → {df.date.max()}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--out", type=Path, default=Path("data/processed/positions.csv.gz"))
    p.add_argument("--default-elo", type=int, default=1500, help="rating used when a PGN has none (OTB games)")
    p.add_argument("--val-frac", type=float, default=0.1)
    args = p.parse_args(argv)

    df, stats = build(args.raw, args.default_elo)
    df = split_by_game(df, args.val_frac)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, compression="gzip")
    text = summary(df, stats)
    print(text)
    (args.out.parent / "summary.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
