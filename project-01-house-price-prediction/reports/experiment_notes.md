# Experiment Notes

Use one row for every experiment. Change only one major setting at a time.

| Experiment | Model/settings | MAE | RMSE | R-squared | Training time | Observation |
|---|---|---:|---:|---:|---:|---|
| 1 | Median baseline | $59,568 | $88,667 | -0.025 | Not recorded | Predicting one median value performs slightly worse than predicting the test-set mean. |
| 2 | Random forest, 200 trees | $17,483 | $28,590 | 0.893 | 6.05 s | Large improvement over the baseline; expensive outliers still increase RMSE. |
| 3 | Random forest, 400 trees | $17,394 | $28,585 | 0.893 | 14.31 s | MAE improved by only $89 while fit time more than doubled; 200 trees is the better efficiency tradeoff. |
| 4 | Random forest, 200 trees, max depth 15 | $17,536 | $28,646 | 0.893 | 6.07 s | Both error metrics became slightly worse and runtime barely changed; depth 15 did not help. |

## Questions

- Which model performed best, and how do you know?
- What does your best model's MAE mean in plain language?
- Where might this model make especially poor predictions?

## Experiment 1 interpretation

- The random forest performed best because it had much lower MAE and RMSE and
  much higher R-squared than the baseline.
- Its MAE means its predictions were off by about $17,483 on average for the 292
  unseen test houses.
- RMSE is higher than MAE because RMSE penalizes a small number of large errors
  more heavily. Unusual luxury houses are a likely source of those errors.
