"""Thin wrapper around Maia-2: loading base / fine-tuned weights, encoding, predicting."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import chess
import torch
from maia2.main import MAIA2Model
from maia2.utils import board_to_tensor, create_elo_dict, get_all_possible_moves, map_to_category

from .dataset import mirror_move

ALL_MOVES = get_all_possible_moves()
MOVE_INDEX = {m: i for i, m in enumerate(ALL_MOVES)}
ELO_DICT = create_elo_dict()

# Small config used in tests (and as a smoke-test base) so CI never needs the real weights.
TINY_CFG = dict(input_channels=18, dim_cnn=8, num_blocks_cnn=1, vit_length=4,
                dim_vit=16, num_blocks_vit=1, elo_dim=8)


def build_model(cfg: dict) -> MAIA2Model:
    return MAIA2Model(len(ALL_MOVES), ELO_DICT, SimpleNamespace(**cfg))


def load_base(kind: str, save_root: str = "maia2_models") -> tuple[MAIA2Model, dict]:
    """Return (model, cfg_dict). kind: 'rapid', 'blitz' or 'tiny' (random, for tests)."""
    if kind == "tiny":
        torch.manual_seed(0)
        return build_model(TINY_CFG), dict(TINY_CFG)
    from maia2 import model as m2model
    from maia2.utils import parse_args

    model = m2model.from_pretrained(type=kind, device="cpu", save_root=save_root)
    cfg = vars(parse_args(str(Path(save_root) / "config.yaml")))
    return model, cfg


def save_checkpoint(path: Path, model: torch.nn.Module, cfg: dict, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "cfg": cfg, "meta": meta}, path)


def load_checkpoint(path: Path) -> tuple[MAIA2Model, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["cfg"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt.get("meta", {})


def legal_indices(board: chess.Board) -> list[int]:
    """Indices of legal moves for a board that is *already* white-to-move."""
    return [MOVE_INDEX[m.uci()] for m in board.legal_moves]


def encode_board(fen_white_to_move: str) -> tuple[torch.Tensor, list[int]]:
    board = chess.Board(fen_white_to_move)
    return board_to_tensor(board), legal_indices(board)


def elo_category(elo: int) -> int:
    return map_to_category(int(elo), ELO_DICT)


@torch.no_grad()
def predict(model: MAIA2Model, board: chess.Board, elo_self: int, elo_oppo: int) -> dict[str, float]:
    """Move probabilities for the side to move, in real (unmirrored) UCI, highest first."""
    black = board.turn == chess.BLACK
    view = board.mirror() if black else board
    x = board_to_tensor(view).unsqueeze(0)
    idx = legal_indices(view)
    if not idx:
        return {}
    model.eval()
    logits, _, _ = model(x, torch.tensor([elo_category(elo_self)]), torch.tensor([elo_category(elo_oppo)]))
    probs = torch.softmax(logits[0, idx], dim=0).tolist()
    moves = [ALL_MOVES[i] for i in idx]
    if black:
        moves = [mirror_move(m) for m in moves]
    return dict(sorted(zip(moves, probs), key=lambda kv: kv[1], reverse=True))
