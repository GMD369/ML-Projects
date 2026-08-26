# Experiment Notes

| Experiment | Vectorizer/model settings | Accuracy | Spam precision | Spam recall | Spam F1 | ROC-AUC | Vocabulary size |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Majority baseline | 0.873 | 0.000 | 0.000 | 0.000 | 0.500 | N/A |
| 2 | TF-IDF unigrams, min_df=2 + Multinomial NB alpha=1.0 | 0.969 | 1.000 | 0.756 | 0.861 | 0.989 | 3,322 |
| 3 | TF-IDF unigrams+bigrams, min_df=2 + Multinomial NB alpha=1.0 | 0.963 | 1.000 | 0.710 | 0.830 | 0.982 | 10,188 |
| 4 | TF-IDF unigrams, min_df=1 + Multinomial NB alpha=1.0 | 0.954 | 1.000 | 0.634 | 0.776 | 0.984 | 7,680 |
| 5 | Final: TF-IDF unigrams, min_df=2 + Multinomial NB alpha=0.1 | 0.981 | 0.951 | 0.893 | 0.921 | 0.993 | 3,322 |

Bigrams increased vocabulary size by 6,866 features (about 207%) but reduced
recall, F1, and ROC-AUC. With this small dataset, many bigrams are too sparse to
provide reliable evidence, so the unigram model remains stronger.

Allowing one-off tokens increased the vocabulary by 4,358 features (about 131%)
relative to `min_df=2`, but increased errors from 32 to 48. Rare tokens added
noise and diluted the useful evidence available to Naive Bayes.

## Alpha cross-validation

Five-fold cross-validation used only the training partition for selection:

| Alpha | Mean CV F1 | CV standard deviation |
|---:|---:|---:|
| 0.05 | 0.935 | 0.011 |
| 0.10 | 0.938 | 0.011 |
| 0.25 | 0.924 | 0.016 |
| 0.50 | 0.908 | 0.028 |
| 0.75 | 0.889 | 0.027 |
| 1.00 | 0.856 | 0.031 |
| 1.50 | 0.797 | 0.035 |
| 2.00 | 0.742 | 0.039 |

Selected `alpha=0.1`. With unigrams and `min_df=2`, its held-out precision was
0.951, recall was 0.893, and F1 was 0.921. Lower smoothing allows strong spam
tokens to influence predictions more, improving recall while introducing a small
number of false positives.

## Error analysis

- Data preparation: removed 403 exact duplicate rows before the train/test split
  to prevent identical messages leaking across both partitions.
- False positive patterns: none in the first test run; all 903 legitimate messages were preserved.
- False negative patterns: promotional messages disguised as service notices or personal messages, unusual abbreviations, changing phone numbers, adult-content offers, and rare phrases removed by `min_df=2`.
- The more costly error is: usually a false positive.
- Reason: hiding a legitimate message can cause direct harm, while a missed spam message is normally an inconvenience. The first model appropriately favors precision, though recall can still be improved.
