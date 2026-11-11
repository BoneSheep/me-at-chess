# me-at-chess ♟️

A chess bot that plays like **Ruben (BoneSheep)**, built by fine-tuning
[Maia-2](https://github.com/CSSLab/maia2), a human-like chess model from
the University of Toronto, on Ruben's own games.

## How it works

1. **Fetch.** `mac-fetch` downloads every BoneSheep game from the public chess.com API,
   one PGN per month, into `data/raw/chesscom/`. OTB games go in `data/raw/otb/`.
2. **Dataset.** `mac-dataset` keeps only the positions where *Ruben* is to move, including the
   opening, since your repertoire is part of your style. Positions are mirrored to the
   side-to-move's view (as Maia-2 expects). The most recent 10% of games are held out for validation.
3. **Fine-tune.** `mac-finetune` starts from Maia-2 (rapid or blitz weights) and trains it to
   predict Ruben's actual moves. It reports **move-match accuracy** (how often the bot's top
   move is the move you played) for the base model and the fine-tuned model.
4. **Play.** `mac-uci` is a UCI engine (use it in Cute Chess, En Croissant or lichess-bot);
   `mac-play` lets you play it in a terminal.

Everything heavy runs on GitHub Actions:

| Workflow | When | What |
|---|---|---|
| **CI** | every push / PR | unit tests with a tiny random Maia-2 (no downloads) |
| **Fetch games & train** | weekly, or *Run workflow* | fetch new games → dataset → fine-tune → publish the model as a GitHub Release |

The latest results are in [`reports/latest.md`](reports/latest.md).

## Use the model locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
pip install --no-deps maia2==0.10.0 && pip install -e ".[dev]"

gh release download --pattern me_at_chess.pt -D models     # latest trained model
mac-play                                                   # play against yourself
mac-play --fen "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
mac-uci --temperature 0.3                                  # UCI engine
```

`--temperature 0` always plays the move "you" are most likely to play. Around `0.3–0.7`
it samples, which feels more human and varies from game to game.

## Layout

```
src/me_at_chess/
  fetch_chesscom.py   chess.com API → monthly PGNs
  dataset.py          PGNs → positions where Ruben moves
  model.py            Maia-2 loading / encoding / prediction
  finetune.py         training loop + report
  uci.py, play.py     the bot
data/raw/             games (committed)
reports/              training reports (committed by the workflow)
```
