# Fine-tuning report

Base model: Maia-2 **rapid** · train positions 62164 · val positions 6671
Time classes: all · lr 3e-05 · early stop after epoch 4 · 44 min

| model | move-match acc | top-3 acc | NLL |
|---|---|---|---|
| Maia-2 base | 47.8% | 75.9% | 1.612 |
| fine-tuned (epoch 2) | 56.1% | 81.4% | 1.383 |

Validation = your most recent games, never seen in training.

| epoch | train loss | val acc | val top-3 | val NLL |
|---|---|---|---|---|
| 0 | – | 0.4780 | 0.7588 | 1.6115 |
| 1 | 1.5020 | 0.5500 | 0.8114 | 1.4079 |
| 2 | 1.3698 | 0.5606 | 0.8140 | 1.3832 |
| 3 | 1.3082 | 0.5578 | 0.8138 | 1.3837 |
| 4 | 1.2640 | 0.5602 | 0.8146 | 1.3912 |
