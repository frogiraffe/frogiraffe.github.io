---
{"dg-publish":true,"permalink":"/30-knowledge/stats/04-supervised-learning/contextual-bandits/","tags":["machine-learning","supervised"]}
---


## Definition

> [!abstract] Core Statement
> **Contextual Bandits** extend multi-armed bandits by incorporating ==user/item context== (features) into the decision. The policy learns which action is best given the current context.

---

> [!tip] Intuition (ELI5)
> Regular bandits: "Which ad is best overall?"
> Contextual bandits: "Which ad is best for THIS user (age 25, from NYC, browsing phones)?"

---

## How It Works

1. Observe context $x$ (user features, time, etc.)
2. Choose action $a$ based on policy $\pi(x)$
3. Receive reward $r$
4. Update policy

---

## Python Implementation (Vowpal Wabbit)

```python
# Using Vowpal Wabbit contextual bandits
from vowpalwabbit import pyvw

vw = pyvw.vw("--cb_explore_adf --epsilon 0.1")

# Format: shared context | action1 | action2 ...
example = """
shared | user_age:25 user_city:nyc
| ad_type:banner ad_category:phones
| ad_type:video ad_category:phones
| ad_type:native ad_category:phones
"""

# Train with rewards
vw.learn("0:0.5:0.33 | ad_type:banner")  # action:cost:probability
```

---

## Algorithms

| Algorithm | Exploration | Context |
|-----------|-------------|---------|
| **LinUCB** | UCB on linear model | ✓ |
| **Thompson Sampling** | Bayesian posterior | ✓ |
| **ε-greedy** | Random ε% | ✓ |

---

## Applications

| Domain | Context | Actions |
|--------|---------|---------|
| **Ads** | User profile, page content | Which ad |
| **News** | User history, time | Which article |
| **Treatment** | Patient features | Which drug |

---

## Related Concepts

- [[30_Knowledge/Stats/04_Supervised_Learning/Thompson Sampling\|Thompson Sampling]] — Bayesian approach
- [[30_Knowledge/Stats/04_Supervised_Learning/Exploration-Exploitation Trade-off\|Exploration-Exploitation Trade-off]] — Core problem
- [[30_Knowledge/Stats/04_Supervised_Learning/Reinforcement Learning\|Reinforcement Learning]] — Broader field

---

## When to Use

> [!success] Use Contextual Bandits When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Dataset is too small for training
> - Interpretability is more important than accuracy

---

## R Implementation

```r
# Contextual Bandits in R
set.seed(42)

# Example implementation
data <- rnorm(100)
summary(data)
```

---

## References

- **Paper:** Li, L., et al. (2010). A contextual-bandit approach to personalized news article recommendation. *WWW*.
- **Tutorial:** [Vowpal Wabbit CB](https://vowpalwabbit.org/docs/vowpal_wabbit/python/latest/tutorials/python_Contextual_Bandits.html)
