---
{"dg-publish":true,"permalink":"/30-knowledge/stats/04-supervised-learning/discounted-cumulative-gain-dcg/","tags":["machine-learning","supervised","ranking","evaluation","information-retrieval"]}
---

## Definition

> [!abstract] Core Statement
> **Discounted Cumulative Gain (DCG)** is a metric for evaluating ==ranking quality==. It measures the usefulness (gain) of search results based on their position, giving higher weight to results at the top of the list.

---

> [!tip] Intuition (ELI5): The Netflix Recommendation Score
> When Netflix shows you 10 movies, the first one matters most. If #1 is perfect (5 stars) and #10 is perfect, you're happier with #1 being good. DCG captures this: early positions count more.

---

## Purpose

1. **Evaluate search engines** and recommendation systems
2. **Compare ranking algorithms** fairly
3. **Account for graded relevance** (not just binary relevant/not)

---

## Theoretical Background

### Cumulative Gain (CG)

Simple sum of relevance scores (ignores position):
$$
CG_p = \sum_{i=1}^{p} rel_i
$$

### Discounted Cumulative Gain (DCG)

Adds position discount:
$$
DCG_p = \sum_{i=1}^{p} \frac{rel_i}{\log_2(i + 1)}
$$

Or alternative formulation (emphasizes high relevance):
$$
DCG_p = \sum_{i=1}^{p} \frac{2^{rel_i} - 1}{\log_2(i + 1)}
$$

### Normalized DCG (nDCG)

Normalize by ideal ranking:
$$
nDCG_p = \frac{DCG_p}{IDCG_p}
$$

where $IDCG$ = DCG of perfect ranking.

---

## Worked Example

**Query results with relevance scores:**

| Position | Document | Relevance |
|----------|----------|-----------|
| 1 | Doc A | 3 |
| 2 | Doc B | 2 |
| 3 | Doc C | 3 |
| 4 | Doc D | 0 |
| 5 | Doc E | 1 |

**Calculate DCG@5:**
$$
DCG_5 = \frac{3}{\log_2(2)} + \frac{2}{\log_2(3)} + \frac{3}{\log_2(4)} + \frac{0}{\log_2(5)} + \frac{1}{\log_2(6)}
$$
$$
= \frac{3}{1} + \frac{2}{1.585} + \frac{3}{2} + \frac{0}{2.322} + \frac{1}{2.585}
$$
$$
= 3 + 1.26 + 1.5 + 0 + 0.39 = 6.15
$$

**Ideal ranking:** [3, 3, 2, 1, 0]
$$
IDCG_5 = \frac{3}{1} + \frac{3}{1.585} + \frac{2}{2} + \frac{1}{2.322} + \frac{0}{2.585} = 6.32
$$

$$
nDCG_5 = \frac{6.15}{6.32} = 0.973
$$

---

## Python Implementation

```python
import numpy as np

def dcg_at_k(relevances, k):
    """Calculate DCG@k"""
    relevances = np.asarray(relevances)[:k]
    if len(relevances) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    return np.sum(relevances / discounts)

def ndcg_at_k(relevances, k):
    """Calculate nDCG@k"""
    dcg = dcg_at_k(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg

# Example
relevances = [3, 2, 3, 0, 1]

print(f"DCG@5: {dcg_at_k(relevances, 5):.3f}")
print(f"nDCG@5: {ndcg_at_k(relevances, 5):.3f}")

# Using sklearn
from sklearn.metrics import ndcg_score
import numpy as np

y_true = np.array([[3, 2, 3, 0, 1]])
y_score = np.array([[0.9, 0.8, 0.7, 0.6, 0.5]])  # Model scores

print(f"sklearn nDCG: {ndcg_score(y_true, y_score):.3f}")
```

**Expected Output:**
```
DCG@5: 6.149
nDCG@5: 0.973
sklearn nDCG: 0.973
```

---

## R Implementation

```r
# DCG function
dcg_at_k <- function(relevances, k) {
  relevances <- relevances[1:min(k, length(relevances))]
  positions <- 1:length(relevances)
  sum(relevances / log2(positions + 1))
}

# nDCG function
ndcg_at_k <- function(relevances, k) {
  dcg <- dcg_at_k(relevances, k)
  ideal <- sort(relevances, decreasing = TRUE)
  idcg <- dcg_at_k(ideal, k)
  if (idcg == 0) return(0)
  dcg / idcg
}

# Example
rel <- c(3, 2, 3, 0, 1)
print(paste("DCG@5:", round(dcg_at_k(rel, 5), 3)))
print(paste("nDCG@5:", round(ndcg_at_k(rel, 5), 3)))
```

---

## Interpretation Guide

| nDCG Score | Meaning |
|------------|---------|
| 1.0 | Perfect ranking |
| 0.8-1.0 | Excellent |
| 0.6-0.8 | Good |
| 0.4-0.6 | Fair |
| < 0.4 | Poor |

---

## Limitations

> [!warning] Pitfalls
> 1. **Requires relevance labels:** Need ground truth relevance scores
> 2. **Position sensitivity:** Very sensitive to top positions
> 3. **Not intuitive:** Hard to explain to non-technical stakeholders
> 4. **Scale dependent:** IDCG varies by query, making cross-query comparison tricky

---

## Related Concepts

- Precision and Recall - Binary relevance metrics
- Mean Average Precision (MAP) - Alternative ranking metric
- [[30_Knowledge/Stats/04_Supervised_Learning/Learning to Rank\|Learning to Rank]] - ML for ranking
- [[30_Knowledge/Stats/04_Supervised_Learning/Information Retrieval\|Information Retrieval]] - Broader field

---

## When to Use

> [!success] Use Discounted Cumulative Gain (DCG) When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Dataset is too small for training
> - Interpretability is more important than accuracy

---

## References

1. Järvelin, K., & Kekäläinen, J. (2002). Cumulated Gain-Based Evaluation of IR Techniques. *ACM TOIS*. [ACM](https://dl.acm.org/doi/10.1145/582415.582418)

2. Wang, Y., et al. (2013). A Theoretical Analysis of NDCG Ranking Measures. *COLT*. [Paper](http://proceedings.mlr.press/v30/Wang13.html)
