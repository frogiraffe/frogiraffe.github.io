---
{"dg-publish":true,"permalink":"/stats/04-supervised-learning/information-retrieval/","tags":["probability","machine-learning","nlp","search"]}
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

- [[stats/04_Supervised_Learning/Learning to Rank\|Learning to Rank]] — ML for ranking
- [[stats/04_Supervised_Learning/NLP\|NLP]] — Foundation

---

## References

- **Book:** Manning, C. D., et al. (2008). *Introduction to Information Retrieval*. Cambridge. [Free Online](https://nlp.stanford.edu/IR-book/)
