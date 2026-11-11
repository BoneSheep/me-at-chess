# Fine-tuning report

Base model: Maia-2 **blitz** · train positions 62164 · val positions 6671
Time classes: all · lr 0.0001 · early stop after epoch 3 · 51 min

| model | move-match acc | top-3 acc | NLL |
|---|---|---|---|
| Maia-2 base | 47.2% | 75.2% | 1.606 |
| fine-tuned (epoch 1) | 55.6% | 82.0% | 1.378 |

Validation = your most recent games, never seen in training.

| epoch | train loss | val acc | val top-3 | val NLL |
|---|---|---|---|---|
| 0 | – | 0.4723 | 0.7524 | 1.6058 |
| 1 | 1.4528 | 0.5564 | 0.8201 | 1.3781 |
| 2 | 1.3009 | 0.5590 | 0.8209 | 1.3802 |
| 3 | 1.1702 | 0.5551 | 0.8183 | 1.4249 |
