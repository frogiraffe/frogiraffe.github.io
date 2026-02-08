---
{"dg-publish":true,"permalink":"/30-knowledge/stats/04-supervised-learning/learning-to-rank/","tags":["machine-learning","supervised"]}
---


## Definition

> [!abstract] Core Statement
> **Learning to Rank** applies machine learning to ==optimize document ranking== for search and recommendation systems.

---

## Approaches

| Approach | Loss Function | Examples |
|----------|---------------|----------|
| **Pointwise** | MSE/Classification | Predict relevance score |
| **Pairwise** | Pair preferences | RankNet, LambdaRank |
| **Listwise** | List-level metric | ListNet, LambdaMART |

---

## Python (LightGBM Ranker)

```python
import lightgbm as lgb

# Data format: features, relevance labels, query groups
train_data = lgb.Dataset(
    X_train, 
    label=y_train,
    group=query_groups  # Number of docs per query
)

params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [1, 3, 5, 10],
    'learning_rate': 0.1
}

model = lgb.train(params, train_data, num_boost_round=100)
predictions = model.predict(X_test)
```

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **NDCG@k** | Discounted cumulative gain |
| **MAP** | Mean average precision |
| **MRR** | Mean reciprocal rank |

---

## Related Concepts

- [[30_Knowledge/Stats/04_Supervised_Learning/Information Retrieval\|Information Retrieval]] — Foundation
- [[30_Knowledge/Stats/04_Supervised_Learning/XGBoost\|XGBoost]] — Often used for LTR

---

## When to Use

> [!success] Use Learning to Rank When...
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

# Example implementation of Learning to Rank
# See documentation for details

data = np.random.randn(100)
print(f"Mean: {np.mean(data):.3f}")
print(f"Std: {np.std(data):.3f}")
```

---

## R Implementation

```r
# Learning to Rank in R
set.seed(42)

# Example implementation
data <- rnorm(100)
summary(data)
```

---

## References

- **Paper:** Burges, C. J. (2010). From RankNet to LambdaRank to LambdaMART. *Microsoft Research*.
