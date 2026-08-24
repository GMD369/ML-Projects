# Experiment Notes

| Experiment | Model/settings | Accuracy | Precision | Recall | F1 | ROC-AUC | Observation |
|---|---|---:|---:|---:|---:|---:|---|
| 1 | Majority baseline | 0.735 | 0.000 | 0.000 | 0.000 | 0.500 | Predicts that everyone stays, so it misses every churner. |
| 2 | Logistic regression | 0.806 | 0.657 | 0.559 | 0.604 | 0.842 | Useful first model, but it still misses about 44% of churners at the default threshold. |
| 3 | Logistic regression, balanced class weights | 0.738 | 0.504 | 0.783 | 0.614 | 0.841 | Catches many more churners at the cost of more false alarms; appropriate when outreach is cheap. |

## Threshold experiment (balanced logistic regression)

| Threshold | Precision | Recall | F1 | Interpretation |
|---:|---:|---:|---:|---|
| 0.30 | 0.429 | 0.928 | 0.587 | Best coverage: catches nearly 93% of churners, but produces many false alarms. |
| 0.50 | 0.504 | 0.783 | 0.614 | Best F1 of the tested thresholds and a balanced operating point. |
| 0.70 | 0.603 | 0.602 | 0.602 | More selective predictions, but misses about 40% of churners. |

Decision: choose 0.30 if customer outreach is cheap relative to losing a
customer. Choose 0.50 when false positives and false negatives need a more even
balance.

## Business interpretation

- A false positive means: the model flags a customer who would actually stay, so the company may make an unnecessary retention offer.
- A false negative means: the model predicts that a customer will stay, but the customer actually churns, so the company misses a chance to intervene.
- The metric I would prioritize is: recall, if contacting customers is cheap.
- My reason is: higher recall catches more actual churners, and the additional false-positive contacts have a relatively low cost.
