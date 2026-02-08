---
{"dg-publish":true,"permalink":"/30-knowledge/stats/04-supervised-learning/information-retrieval/","tags":["machine-learning","supervised"]}
---


## Definition

> [!abstract] Core Statement
> **Information Retrieval** is the task of finding ==relevant documents from a collection== given a query, using techniques from NLP and machine learning.

---

## Key Metrics

| Metric | Formula | Use |
|--------|---------|-----|
| **Precision@k** | Relevant in top k / k | Top results quality |
| **Recall@k** | Relevant in top k / Total relevant | Coverage |
| **MAP** | Mean Average Precision | Ranking quality |
| **NDCG** | Normalized DCG | Graded relevance |

---

## Classic Techniques

| Technique | Description |
|-----------|-------------|
| **TF-IDF** | Term frequency × inverse document frequency |
| **BM25** | Probabilistic ranking function |
| **Boolean** | AND/OR queries |

---

## Python (BM25)

```python
from rank_bm25 import BM25Okapi

corpus = [
    "machine learning is great",
    "deep learning uses neural networks",
    "statistics is fundamental"
]

tokenized = [doc.split() for doc in corpus]
bm25 = BM25Okapi(tokenized)

query = "neural networks"
scores = bm25.get_scores(query.split())
print(f"Scores: {scores}")
```

---

## Related Concepts

- [[30_Knowledge/Stats/04_Supervised_Learning/Learning to Rank\|Learning to Rank]] — ML for ranking
- [[30_Knowledge/Stats/04_Supervised_Learning/NLP\|NLP]] — Foundation

---

## When to Use

> [!success] Use Information Retrieval When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Dataset is too small for training
> - Interpretability is more important than accuracy

---

## Python Implementation

```python
import numpy as np
import pandas as pd

# Example implementation of Information Retrieval
# See documentation for details

data = np.random.randn(100)
print(f"Mean: {np.mean(data):.3f}")
print(f"Std: {np.std(data):.3f}")
```

---

## R Implementation

```r
# Information Retrieval in R
set.seed(42)

# Example implementation
data <- rnorm(100)
summary(data)
```

---

## References

- **Book:** Manning, C. D., et al. (2008). *Introduction to Information Retrieval*. Cambridge. [Free Online](https://nlp.stanford.edu/IR-book/)
