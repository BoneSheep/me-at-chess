# Fine-tuning report

Base model: Maia-2 **rapid** · train positions 18050 · val positions 1950
Time classes: all · lr 3e-05 · completed · 8 min

| model | move-match acc | top-3 acc | NLL |
|---|---|---|---|
| Maia-2 base | 47.0% | 76.5% | 1.600 |
| fine-tuned (epoch 2) | 53.8% | 81.5% | 1.405 |

Validation = your most recent games, never seen in training.

| epoch | train loss | val acc | val top-3 | val NLL |
|---|---|---|---|---|
| 0 | – | 0.4697 | 0.7651 | 1.6005 |
| 1 | 1.5280 | 0.5390 | 0.8154 | 1.4173 |
| 2 | 1.4229 | 0.5385 | 0.8154 | 1.4048 |
