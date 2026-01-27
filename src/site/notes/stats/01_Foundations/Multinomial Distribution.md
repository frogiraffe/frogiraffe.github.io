---
{"dg-publish":true,"permalink":"/stats/01-foundations/multinomial-distribution/","tags":["Probability","Distributions","Categorical"]}
---


## Definition

> [!abstract] Core Statement
> The **Multinomial Distribution** generalizes the binomial to ==k > 2 categories==. It models the counts of outcomes when sampling n times from k categories with fixed probabilities.

$$
P(X_1=x_1, \ldots, X_k=x_k) = \frac{n!}{x_1! \cdots x_k!} \prod_{i=1}^k p_i^{x_i}
$$

Where $\sum x_i = n$ and $\sum p_i = 1$.

---

## Properties

| Property | Formula |
|----------|---------|
| **E[X_i]** | $np_i$ |
| **Var[X_i]** | $np_i(1-p_i)$ |
| **Cov[X_i, X_j]** | $-np_ip_j$ (i ≠ j) |

---

## Python Implementation

```python
import numpy as np
from scipy import stats

# ========== SAMPLING ==========
# 100 trials, 4 categories with probabilities [0.2, 0.3, 0.35, 0.15]
n = 100
probs = [0.2, 0.3, 0.35, 0.15]

samples = np.random.multinomial(n, probs, size=1000)
print(f"Sample means: {samples.mean(axis=0)}")
print(f"Expected: {np.array(probs) * n}")

# ========== PMF ==========
# P(X = [20, 30, 35, 15]) given n=100, probs as above
from scipy.special import factorial

def multinomial_pmf(x, n, p):
    coef = factorial(n) / np.prod(factorial(x))
    return coef * np.prod(p ** x)

x = np.array([20, 30, 35, 15])
prob = multinomial_pmf(x, n, probs)
print(f"P(X = {x}) = {prob:.6e}")
```

---

## R Implementation

```r
# Sampling
rmultinom(1, size = 100, prob = c(0.2, 0.3, 0.35, 0.15))

# PMF
dmultinom(c(20, 30, 35, 15), prob = c(0.2, 0.3, 0.35, 0.15))
```

---

## Applications

| Application | Categories |
|-------------|------------|
| **Dice rolls** | 6 outcomes |
| **Survey responses** | Multiple choice |
| **Topic modeling** | Document-word counts |
| **Genetics** | Genotype frequencies |

---

## Related Concepts

- [[stats/01_Foundations/Binomial Distribution\|Binomial Distribution]] — Special case (k=2)
- [[stats/01_Foundations/Dirichlet Distribution\|Dirichlet Distribution]] — Conjugate prior
- [[stats/02_Statistical_Inference/Chi-Square Test\|Chi-Square Test]] — Goodness-of-fit for multinomial

---

## References

- **Book:** Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. Section 2.2.
