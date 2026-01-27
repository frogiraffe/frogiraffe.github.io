---
{"dg-publish":true,"permalink":"/stats/01-foundations/multi-armed-bandit/","tags":["A-B-Testing","Reinforcement-Learning","Optimization"]}
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

- [[stats/02_Statistical_Inference/A-B Testing\|A-B Testing]] - Traditional approach (no exploration)
- [[stats/02_Statistical_Inference/Sequential Testing\|Sequential Testing]] - Related methodology
- [[stats/01_Foundations/Bayesian Statistics\|Bayesian Statistics]] - Thompson Sampling uses Bayes

---

## References

- **Book:** Slivkins, A. (2019). *Introduction to Multi-Armed Bandits*. [arXiv:1904.07272](https://arxiv.org/abs/1904.07272)
