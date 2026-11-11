# Fine-tuning report

Base model: Maia-2 **rapid** · train positions 62164 · val positions 6671
Time classes: all · lr 0.0001 · early stop after epoch 4 · 43 min

| model | move-match acc | top-3 acc | NLL |
|---|---|---|---|
| Maia-2 base | 47.8% | 75.9% | 1.612 |
| fine-tuned (epoch 2) | 55.9% | 81.6% | 1.382 |

Validation = your most recent games, never seen in training.

| epoch | train loss | val acc | val top-3 | val NLL |
|---|---|---|---|---|
| 0 | – | 0.4780 | 0.7588 | 1.6115 |
| 1 | 1.4581 | 0.5545 | 0.8123 | 1.3878 |
| 2 | 1.3033 | 0.5585 | 0.8162 | 1.3818 |
| 3 | 1.1749 | 0.5527 | 0.8185 | 1.4198 |
| 4 | 1.0577 | 0.5401 | 0.8078 | 1.5034 |
