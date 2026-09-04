# Experiment Notes

| Experiment | Architecture/settings | Accuracy | Macro F1 | Iterations | Final loss | Observation |
|---|---|---:|---:|---:|---:|---|
| 1 | Majority baseline | 0.100 | 0.018 | N/A | N/A | Predicting one class is useless for balanced ten-class data. |
| 2 | Scaled pixels, hidden layer (64,), alpha=0.0001 | 0.958 | 0.958 | 53 | 0.01589 | Strong first neural network; only 15 of 360 test images were incorrect. |
| 3 | Unscaled pixels, hidden layer (64,), alpha=0.0001 | 0.947 | 0.947 | 54 | 0.00802 | Four additional test errors; scaling improved generalization despite the unscaled run's lower training loss. |
| 4 | Scaled pixels, hidden layers (128, 64), alpha=0.0001 | 0.969 | 0.969 | 31 | 0.00487 | Current best: only 11 test errors and better performance than the single hidden layer. |
| 5 | Scaled pixels, hidden layers (128, 64), alpha=0.01 | 0.969 | 0.969 | 31 | 0.02439 | Same test predictions as alpha=0.0001; stronger regularization made no measurable difference on this split. |

## Error analysis

- Most confused digit pair: actual 8 predicted as 1 (3 cases); 6 to 8 and 9 to 7 occurred twice each.
- A likely visual reason: the 8 x 8 resolution makes weak loops and short strokes disappear, so narrow 8s can resemble 1s and open 9s can resemble 7s.
- Lowest-recall digit: 8, with recall 0.886 (31 of 35 correctly identified).
- What accuracy did not reveal: errors are uneven across classes; digit 2 had perfect recall while digit 8 was substantially harder.

The loss curve decreased smoothly from above 2.0 to 0.01589. Early stopping ended
training after 53 iterations instead of using the maximum 300 iterations.

The unscaled experiment was deterministic across repeated runs because the data
split and MLP both use `random_state=42`. Its lower training loss did not imply a
better model: held-out accuracy and macro F1 both fell by 0.011, and digit 8
recall fell from 0.886 to 0.829.

The deeper `(128, 64)` network improved accuracy and macro F1 by 0.011 over the
scaled `(64,)` network and reduced errors from 15 to 11. It also reached early
stopping in fewer iterations, though each iteration performs more computation.
Digit 8 remained the hardest class, with recall 0.857.

Increasing `alpha` to 0.01 produced the same 11 mistakes and identical per-class
metrics. The larger final loss is not evidence of a worse fit because the MLP
loss includes the L2 penalty controlled by `alpha`; losses from different
regularization strengths are not directly comparable. The saved final model uses
the stronger `alpha=0.01` setting, but this experiment does not establish that it
will generalize better than `alpha=0.0001`.
