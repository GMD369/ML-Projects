# Experiment Notes

## Selecting k

| k | Inertia | Silhouette score | Observation |
|---:|---:|---:|---|
| 2 | 269.691 | 0.321 | Too broad; weak separation. |
| 3 | 157.704 | 0.467 | Major improvement, but distinct spending groups remain combined. |
| 4 | 108.921 | 0.494 | Better separation, though the elbow has not fully flattened. |
| 5 | 65.568 | 0.555 | Best silhouette score and a clear elbow; selected solution. |
| 6 | 55.057 | 0.540 | Lower inertia, but slightly weaker separation than k=5. |
| 7 | 44.865 | 0.528 | Additional complexity without improved silhouette. |
| 8 | 37.148 | 0.457 | Noticeable decline in cluster separation. |
| 9 | 32.392 | 0.457 | Little benefit over k=8. |
| 10 | 29.686 | 0.436 | Lowest inertia but weakest separation among the larger solutions. |

## Cluster interpretation

| Cluster | Business name | Typical income | Typical spending score | Possible action |
|---:|---|---:|---:|---|
| 0 | Mainstream customers | $55.30k | 49.52 | Use broad campaigns, cross-selling, and loyalty nudges. |
| 1 | High-value VIPs | $86.54k | 82.13 | Retain with premium service, exclusivity, and rewards. |
| 2 | Enthusiastic budget shoppers | $25.73k | 79.36 | Offer affordable promotions and spending-based rewards. |
| 3 | High-income cautious shoppers | $88.20k | 17.11 | Use personalized discovery and premium-product activation campaigns. |
| 4 | Low-engagement budget shoppers | $26.30k | 20.91 | Use low-cost campaigns; avoid expensive acquisition incentives. |

Cluster labels are arbitrary IDs, not grades or rankings.

## Decision

Selected `k=5`. It has the highest tested silhouette score (0.555), and the
inertia curve shows a strong elbow at five clusters. Beyond five, inertia must
continue decreasing, but the smaller reductions and declining silhouette scores
do not justify the added complexity.

## Age feature experiment

Adding `Age` produced the following notable results:

| Features | Best k | Best silhouette | Result |
|---|---:|---:|---|
| Income + spending score | 5 | 0.555 | Stronger, clearer segmentation; keep as final model. |
| Income + spending score + age | 6 | 0.427 | Weaker separation despite using more information. |

The three-feature run still trained `k=5` because `selected_k = 5` is a separate,
fixed training choice. Its `k=5` silhouette was 0.417. Adding a feature creates a
new distance geometry; extra information does not guarantee better clusters.
