"""Play against the bot in your terminal, or ask what "you" would play in a position.

    mac-play                              # play a game (you are white)
    mac-play --color black
    mac-play --fen "<FEN>"                # show the bot's top moves for a position
"""

from __future__ import annotations

import argparse
from pathlib import Path

import chess

from .model import load_base, load_checkpoint, predict
from .uci import Engine


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path("models/me_at_chess.pt"))
    p.add_argument("--base", default=None)
    p.add_argument("--fen", default=None)
    p.add_argument("--color", choices=["white", "black"], default="white", help="your colour")
    p.add_argument("--elo-self", type=int, default=1500)
    p.add_argument("--elo-oppo", type=int, default=1500)
    p.add_argument("--temperature", type=float, default=0.5)
    args = p.parse_args(argv)

    model = load_base(args.base)[0] if args.base else load_checkpoint(args.model)[0]

    if args.fen:
        board = chess.Board(args.fen)
        for uci, prob in list(predict(model, board, args.elo_self, args.elo_oppo).items())[:5]:
            print(f"{board.san(chess.Move.from_uci(uci)):8s} {prob:6.1%}")
        return

    engine = Engine(model, args.elo_self, args.elo_oppo, args.temperature)
    board = chess.Board()
    human = chess.WHITE if args.color == "white" else chess.BLACK
    while not board.is_game_over():
        print(board, "\n")
        if board.turn == human:
            raw = input("your move (SAN or UCI, 'q' to quit): ").strip()
            if raw == "q":
                return
            try:
                board.push_san(raw)
            except ValueError:
                try:
                    board.push_uci(raw)
                except ValueError:
                    print("illegal move\n")
        else:
            move = engine.choose(board)
            print(f"bot plays {board.san(move)}\n")
            board.push(move)
    print(board, "\nresult:", board.result())


if __name__ == "__main__":
    main()
