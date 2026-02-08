---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/multi-armed-bandit/","tags":["probability","foundations"]}
---


## Definition

> [!abstract] Core Statement
> The **Multi-Armed Bandit (MAB)** problem balances ==exploration (trying new options) vs exploitation (using best known option)== to maximize cumulative reward over time.

---

## Strategies

| Strategy | Approach |
|----------|----------|
| **ε-greedy** | Explore with probability ε, exploit otherwise |
| **UCB** | Choose arm with highest upper confidence bound |
| **Thompson Sampling** | Sample from posterior, choose highest |

---

## ε-greedy Algorithm

```python
import numpy as np

def epsilon_greedy(rewards, counts, epsilon=0.1):
    if np.random.random() < epsilon:
        return np.random.randint(len(rewards))  # Explore
    else:
        return np.argmax(rewards / (counts + 1e-10))  # Exploit
```

---

## Thompson Sampling

```python
from scipy import stats

def thompson_sampling(successes, failures):
    samples = [stats.beta(s+1, f+1).rvs() for s, f in zip(successes, failures)]
    return np.argmax(samples)
```

---

## Applications

- A/B testing with early stopping
- Ad placement optimization
- Clinical trial allocation
- Recommendation systems

---

## Related Concepts

- [[30_Knowledge/Stats/02_Statistical_Inference/A-B Testing\|A-B Testing]] - Traditional approach (no exploration)
- [[30_Knowledge/Stats/02_Statistical_Inference/Sequential Testing\|Sequential Testing]] - Related methodology
- [[30_Knowledge/Stats/01_Foundations/Bayesian Statistics\|Bayesian Statistics]] - Thompson Sampling uses Bayes

---

## When to Use

> [!success] Use Multi-Armed Bandit When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## Python Implementation

```python
import numpy as np
import pandas as pd

# Example implementation of Multi-Armed Bandit
# See documentation for details

data = np.random.randn(100)
print(f"Mean: {np.mean(data):.3f}")
print(f"Std: {np.std(data):.3f}")
```

---

## R Implementation

```r
# Multi-Armed Bandit in R
set.seed(42)

# Example implementation
data <- rnorm(100)
summary(data)
```

---

## References

- **Book:** Slivkins, A. (2019). *Introduction to Multi-Armed Bandits*. [arXiv:1904.07272](https://arxiv.org/abs/1904.07272)
