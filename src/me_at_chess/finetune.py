"""Fine-tune Maia-2 on Ruben's own moves.

Objective: predict the move Ruben actually played (cross-entropy over legal
moves), plus a small value-head term on the game result. The headline metric
is *move-matching accuracy* on held-out recent games: how often the model's
top move is the move Ruben played. We report it for the untouched base model
too, so the gain from personalisation is visible.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .model import MOVE_INDEX, elo_category, encode_board, load_base, save_checkpoint


class PositionDataset(Dataset):
    """Pre-encodes every position once (board_to_tensor is slow in Python)."""

    def __init__(self, df: pd.DataFrame):
        boards, legal, labels = [], [], []
        for fen, move in zip(df.fen, df.move):
            t, idx = encode_board(fen)
            boards.append(t.to(torch.uint8))
            legal.append(np.asarray(idx, dtype=np.int16))
            labels.append(MOVE_INDEX[move])
        self.boards = torch.stack(boards) if boards else torch.zeros((0, 18, 8, 8), dtype=torch.uint8)
        self.legal = legal
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.elo_self = torch.tensor([elo_category(e) for e in df.elo_self], dtype=torch.long)
        self.elo_oppo = torch.tensor([elo_category(e) for e in df.elo_oppo], dtype=torch.long)
        self.win = torch.tensor(df.active_win.values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i):
        return i

    def collate(self, idxs: list[int]):
        ix = torch.tensor(idxs)
        mask = torch.zeros((len(idxs), len(MOVE_INDEX)), dtype=torch.bool)
        for row, i in enumerate(idxs):
            mask[row, torch.from_numpy(self.legal[i].astype(np.int64))] = True
        return (self.boards[ix].float(), self.labels[ix], self.elo_self[ix],
                self.elo_oppo[ix], mask, self.win[ix])


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~mask, -1e9)


@torch.no_grad()
def evaluate(model, loader) -> dict:
    model.eval()
    n, correct, top3, nll = 0, 0, 0, 0.0
    for boards, labels, es, eo, mask, _ in loader:
        logits, _, _ = model(boards, es, eo)
        logits = masked_logits(logits, mask)
        nll += F.cross_entropy(logits, labels, reduction="sum").item()
        top = logits.topk(3, dim=1).indices
        correct += (top[:, 0] == labels).sum().item()
        top3 += (top == labels[:, None]).any(dim=1).sum().item()
        n += len(labels)
    if n == 0:
        return {"n": 0, "acc": float("nan"), "top3": float("nan"), "nll": float("nan")}
    return {"n": n, "acc": correct / n, "top3": top3 / n, "nll": nll / n}


def train(df: pd.DataFrame, base: str, out: Path, epochs: int = 8, lr: float = 3e-5,
          wd: float = 1e-4, batch_size: int = 256, value_coef: float = 0.1,
          patience: int = 2, max_minutes: float = 300.0, time_classes: list[str] | None = None,
          maia_root: str = "maia2_models", log=print) -> dict:
    t0 = time.time()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    torch.manual_seed(0)

    if time_classes:
        df = df[df.time_class.isin(time_classes)]
    tr_df, va_df = df[df.split == "train"], df[df.split == "val"]
    log(f"train positions: {len(tr_df)}  val positions: {len(va_df)}")

    tr, va = PositionDataset(tr_df), PositionDataset(va_df)
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True, collate_fn=tr.collate)
    va_loader = DataLoader(va, batch_size=batch_size * 2, shuffle=False, collate_fn=va.collate)
    log(f"encoded in {time.time() - t0:.0f}s")

    model, cfg = load_base(base, maia_root)
    base_metrics = evaluate(model, va_loader)
    log(f"base Maia-2 ({base}) on your val games: acc={base_metrics['acc']:.4f} top3={base_metrics['top3']:.4f} nll={base_metrics['nll']:.4f}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps_total = max(1, epochs * math.ceil(len(tr) / batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps_total, pct_start=0.1)

    history = [{"epoch": 0, **base_metrics}]
    best = dict(base_metrics, epoch=0)
    meta_common = {"base_model": base, "lr": lr, "wd": wd, "batch_size": batch_size,
                   "time_classes": time_classes, "train_positions": len(tr), "val_positions": len(va),
                   "elo_default": int(df.elo_self.median()) if len(df) else 1500}
    save_checkpoint(out, model, cfg, {**meta_common, "epoch": 0, "val": base_metrics})
    bad_epochs, stopped = 0, "completed"

    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for boards, labels, es, eo, mask, win in tr_loader:
            logits, _, value = model(boards, es, eo)
            loss = F.cross_entropy(masked_logits(logits, mask), labels)
            if value_coef:
                loss = loss + value_coef * F.mse_loss(value, win)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item() * len(labels)
            seen += len(labels)
            if (time.time() - t0) / 60 > max_minutes:
                break
        m = evaluate(model, va_loader)
        m.update(epoch=epoch, train_loss=running / max(seen, 1))
        history.append(m)
        log(f"epoch {epoch}: train_loss={m['train_loss']:.4f} val_acc={m['acc']:.4f} top3={m['top3']:.4f} nll={m['nll']:.4f} [{(time.time() - t0) / 60:.1f} min]")

        if m["nll"] < best["nll"]:
            best = dict(m)
            bad_epochs = 0
            save_checkpoint(out, model, cfg, {**meta_common, "epoch": epoch, "val": m})
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                stopped = f"early stop after epoch {epoch}"
                break
        if (time.time() - t0) / 60 > max_minutes:
            stopped = f"time budget reached after epoch {epoch}"
            break

    result = {"base": base_metrics, "best": best, "history": history, "stopped": stopped,
              "minutes": (time.time() - t0) / 60, **meta_common}
    log(f"best epoch {best['epoch']}: acc {base_metrics['acc']:.4f} -> {best['acc']:.4f} ({stopped})")
    return result


def write_report(result: dict, path: Path) -> None:
    b, f = result["base"], result["best"]
    lines = [
        "# Fine-tuning report",
        "",
        f"Base model: Maia-2 **{result['base_model']}** · train positions {result['train_positions']} · val positions {result['val_positions']}",
        f"Time classes: {', '.join(result['time_classes']) if result['time_classes'] else 'all'} · lr {result['lr']} · {result['stopped']} · {result['minutes']:.0f} min",
        "",
        "| model | move-match acc | top-3 acc | NLL |",
        "|---|---|---|---|",
        f"| Maia-2 base | {b['acc']:.1%} | {b['top3']:.1%} | {b['nll']:.3f} |",
        f"| fine-tuned (epoch {f['epoch']}) | {f['acc']:.1%} | {f['top3']:.1%} | {f['nll']:.3f} |",
        "",
        "Validation = your most recent games, never seen in training.",
        "",
        "| epoch | train loss | val acc | val top-3 | val NLL |",
        "|---|---|---|---|---|",
    ]
    for h in result["history"]:
        tl = f"{h['train_loss']:.4f}" if "train_loss" in h else "–"
        lines.append(f"| {h['epoch']} | {tl} | {h['acc']:.4f} | {h['top3']:.4f} | {h['nll']:.4f} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/processed/positions.csv.gz"))
    p.add_argument("--base", default="rapid", choices=["rapid", "blitz", "tiny"])
    p.add_argument("--out", type=Path, default=Path("models/me_at_chess.pt"))
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--value-coef", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--max-minutes", type=float, default=300.0)
    p.add_argument("--time-classes", default="", help="comma list, e.g. blitz,rapid (default: all)")
    p.add_argument("--max-positions", type=int, default=0, help="subsample for quick runs")
    p.add_argument("--report", type=Path, default=Path("reports/latest.md"))
    args = p.parse_args(argv)

    df = pd.read_csv(args.data)
    if args.max_positions and len(df) > args.max_positions:
        df = df.sample(args.max_positions, random_state=0)
    tcs = [t for t in args.time_classes.split(",") if t] or None
    result = train(df, args.base, args.out, args.epochs, args.lr, args.wd, args.batch_size,
                   args.value_coef, args.patience, args.max_minutes, tcs)
    write_report(result, args.report)
    args.report.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
