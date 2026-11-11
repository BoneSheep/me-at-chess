# Latest results (2026-11-11)

Validation: Ruben's 203 most recent games (6,671 positions), never seen in training.
"Move-match" = how often the model's top move is the move Ruben actually played.

| run | base | lr | best epoch | move-match | top-3 | NLL | release |
|---|---|---|---|---|---|---|---|
| Maia-2 out of the box | rapid | – | – | 47.8% | 75.9% | 1.612 | – |
| **fine-tuned ⭐** | rapid | 3e-5 | 2 | **56.1%** | 81.4% | 1.383 | `model-20261111-run2-rapid-lr3e-5` |
| fine-tuned | rapid | 1e-4 | 2 | 55.9% | 81.6% | 1.382 | `model-20261111-run3-rapid-lr1e-4` |
| fine-tuned | blitz | 1e-4 | 1 | 55.6% | 82.0% | 1.378 | `model-20261111-run4-blitz-lr1e-4` |

All three fine-tunes land within 0.5 pp of each other: the gain (+8 pp) comes from the
personal data, not the hyper-parameters. Validation loss bottoms out after 1–2 epochs,
so more epochs don't help with ~2k games; more games (or OTB games) would.

Default model (`gh release download` → latest): **model-20261111-run2-rapid-lr3e-5**.
Per-run details: [`reports/runs/`](runs/).
