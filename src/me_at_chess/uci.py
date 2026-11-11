"""UCI engine that plays like Ruben.

Plug it into any UCI GUI (Cute Chess, Arena, En Croissant) or into
lichess-bot as a "homemade"/UCI engine:

    mac-uci --model models/me_at_chess.pt

Options:
  EloSelf / EloOppo  rating buckets the model conditions on
  Temperature        0 = always play the most "Ruben" move, >0 = sample (more varied, more human)
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import chess

from .model import load_base, load_checkpoint, predict

NAME = "me-at-chess (plays like BoneSheep)"


class Engine:
    def __init__(self, model, elo_self: int = 1500, elo_oppo: int = 1500, temperature: float = 0.0, seed: int | None = None):
        self.model = model
        self.elo_self = elo_self
        self.elo_oppo = elo_oppo
        self.temperature = temperature
        self.rng = random.Random(seed)
        self.board = chess.Board()

    def choose(self, board: chess.Board) -> chess.Move | None:
        probs = predict(self.model, board, self.elo_self, self.elo_oppo)
        if not probs:
            return None
        moves, ps = list(probs), list(probs.values())
        if self.temperature <= 0:
            return chess.Move.from_uci(moves[0])
        weights = [p ** (1.0 / self.temperature) for p in ps]
        return chess.Move.from_uci(self.rng.choices(moves, weights=weights, k=1)[0])

    def set_position(self, tokens: list[str]) -> None:
        if not tokens:
            return
        if tokens[0] == "startpos":
            self.board = chess.Board()
            rest = tokens[1:]
        elif tokens[0] == "fen":
            fen_parts = []
            rest = tokens[1:]
            while rest and rest[0] != "moves":
                fen_parts.append(rest.pop(0))
            self.board = chess.Board(" ".join(fen_parts))
        else:
            return
        if rest and rest[0] == "moves":
            for uci in rest[1:]:
                self.board.push_uci(uci)

    def handle(self, line: str, out=sys.stdout) -> bool:
        """Process one UCI command. Returns False on 'quit'."""
        tokens = line.strip().split()
        if not tokens:
            return True
        cmd = tokens[0]

        def say(s: str) -> None:
            print(s, file=out, flush=True)

        if cmd == "uci":
            say(f"id name {NAME}")
            say("id author Ruben Thienpont")
            say(f"option name EloSelf type spin default {self.elo_self} min 600 max 3000")
            say(f"option name EloOppo type spin default {self.elo_oppo} min 600 max 3000")
            say(f"option name Temperature type string default {self.temperature}")
            say("uciok")
        elif cmd == "isready":
            say("readyok")
        elif cmd == "ucinewgame":
            self.board = chess.Board()
        elif cmd == "setoption" and "name" in tokens and "value" in tokens:
            name = " ".join(tokens[tokens.index("name") + 1:tokens.index("value")]).lower()
            value = " ".join(tokens[tokens.index("value") + 1:])
            if name == "eloself":
                self.elo_self = int(value)
            elif name == "elooppo":
                self.elo_oppo = int(value)
            elif name == "temperature":
                self.temperature = float(value)
        elif cmd == "position":
            self.set_position(tokens[1:])
        elif cmd == "go":
            move = self.choose(self.board)
            say(f"bestmove {move.uci() if move else '0000'}")
        elif cmd == "quit":
            return False
        return True


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path("models/me_at_chess.pt"))
    p.add_argument("--base", default=None, help="use an un-finetuned base instead (rapid/blitz/tiny)")
    p.add_argument("--elo-self", type=int, default=None, help="default: your median rating in training data")
    p.add_argument("--elo-oppo", type=int, default=1500)
    p.add_argument("--temperature", type=float, default=0.0)
    args = p.parse_args(argv)

    if args.base:
        model, _ = load_base(args.base)
        meta = {}
    else:
        model, meta = load_checkpoint(args.model)
    elo = args.elo_self or meta.get("elo_default", 1500)
    engine = Engine(model, elo, args.elo_oppo, args.temperature)
    for line in sys.stdin:
        if not engine.handle(line):
            break


if __name__ == "__main__":
    main()
