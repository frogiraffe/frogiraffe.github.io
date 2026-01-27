---
{"dg-publish":true,"permalink":"/stats/01-foundations/categorical-distribution/","tags":["Distributions","Discrete","Multivariate"]}
---


## Definition

> [!abstract] Core Statement
> The **Categorical Distribution** models a ==single trial with K possible outcomes==, each with probability $p_k$.

$$P(X = k) = p_k \quad \text{where } \sum_{k=1}^{K} p_k = 1$$

---

## Relationship to Other Distributions

| Distribution | Relationship |
|--------------|--------------|
| **Bernoulli** | Categorical with K=2 |
| **Multinomial** | n independent Categorical trials |
| **Dirichlet** | Conjugate prior for p |

---

## One-Hot Encoding

Often represented as vector $\mathbf{x} = (0, \dots, 1, \dots, 0)$ with 1 in position k.

---

## Python Implementation

```python
import numpy as np

probs = [0.3, 0.5, 0.2]  # 3 categories
samples = np.random.choice([0, 1, 2], size=1000, p=probs)
print("Counts:", np.bincount(samples))
```

---

## R Implementation

```r
probs <- c(0.3, 0.5, 0.2)
sample(1:3, 1000, replace = TRUE, prob = probs)
```

---

## Applications

- Classification models (softmax output)
- Survey responses with multiple choices
- [[stats/04_Machine_Learning/Naive Bayes\|Naive Bayes]] - Feature likelihoods

---

## References

- **Book:** Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press. [MIT Press](https://mitpress.mit.edu/9780262017091/machine-learning/)
