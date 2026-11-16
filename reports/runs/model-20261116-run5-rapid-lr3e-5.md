# Fine-tuning report

Base model: Maia-2 **rapid** · train positions 62164 · val positions 6671
Time classes: all · lr 3e-05 · early stop after epoch 4 · 43 min

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

## Detector calibration
```
{
  "threshold": 0.09474358707666397,
  "accuracy": 0.8776041666666666,
  "auc": 0.9457450560182297,
  "ruben_sides": 193,
  "opponent_sides": 191,
  "ruben_mean": 0.2609666513636168,
  "opponent_mean": -0.055576189933710426,
  "recall_ruben": 0.8860103626943006,
  "false_positive_rate": 0.13089005235602094
}```
