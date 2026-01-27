---
{"dg-publish":true,"permalink":"/stats/01-foundations/bernoulli-distribution/","tags":["Probability-Theory","Distributions","Discrete"]}
---

## Definition

> [!abstract] Core Statement
> The **Bernoulli Distribution** is the simplest discrete probability distribution. It models a **single trial** with exactly two possible outcomes: Success ($k=1$) with probability $p$, and Failure ($k=0$) with probability $q = 1-p$.

---

> [!tip] Intuition (ELI5): The Switch
> The Bernoulli distribution is the simplest "On/Off" switch of probability. It represents exactly one coin flip or one "Yes/No" question. Everything else in probability is just a bunch of these switches connected together.

![Bernoulli Distribution PMF](https://upload.wikimedia.org/wikipedia/commons/b/b6/PMF_and_CDF_of_a_bernouli_distribution.png)

$$ P(X=k) = p^k (1-p)^{1-k} \quad \text{for } k \in \{0, 1\} $$

---

## Purpose

1.  **Building Block:** It is the fundamental atom of probability.
    -   $n$ Bernoulli trials $\to$ **[[stats/01_Foundations/Binomial Distribution\|Binomial Distribution]]**.
    -   Trials until success $\to$ **[[stats/01_Foundations/Geometric Distribution\|Geometric Distribution]]**.
2.  **Binary Classification:** Modeling Yes/No outcomes (Logistic Regression).

---

## Key Moments

| Statistic | Formula | Intuition |
|-----------|---------|-----------|
| **Mean** ($E[X]$) | $p$ | If $p=0.8$, average outcome represents 80% success. |
| **Variance** ($\sigma^2$) | $p(1-p)$ | Max variance at $p=0.5$ (Most uncertainty). Min variance at $p=0$ or $p=1$ (Certainty). |

---

## Worked Example: Weighted Coin

> [!example] Problem
> A weighted coin lands Heads (1) 70% of the time ($p=0.7$).
> 
> **1. Probability of Heads:** $0.7$.
> **2. Variance:**
> $$ \sigma^2 = 0.7 \times (1 - 0.7) = 0.7 \times 0.3 = 0.21 $$
> 
> *Contrast:* If fair coin ($p=0.5$): $\sigma^2 = 0.5 \times 0.5 = 0.25$. (Fair coin is more unpredictable).

---

## Assumptions

- [ ] **Binary Outcome:** Exactly 0 or 1. No "Edge" or "Maybe".
- [ ] **Single Trial:** Only happens once.

---

## Python Implementation

```python
from scipy.stats import bernoulli
import matplotlib.pyplot as plt

p = 0.3
rv = bernoulli(p)

# Moments
mean, var = rv.stats(moments='mv')
print(f"Mean: {mean}, Variance: {var}")

# PMF
print(f"Prob of Success: {rv.pmf(1)}")
print(f"Prob of Failure: {rv.pmf(0)}")
```

---

## R Implementation

```r
# Simulate Bernoulli Trials
p <- 0.7
n <- 10

# Generate 10 random variates
flips <- rbinom(n, size=1, prob=p)

print(flips)
# [1] 1 1 0 1 1 0 1 1 1 0
```

---

## Related Concepts

- [[stats/01_Foundations/Binomial Distribution\|Binomial Distribution]] - Sum of $n$ Bernoullis.
- [[stats/03_Regression_Analysis/Logistic Regression\|Logistic Regression]] - Modeling outcomes using Bernoulli likelihood.
- [[stats/01_Foundations/Categorical Distribution\|Categorical Distribution]] - Generalization to >2 outcomes (Dice role).

---

## References

- **Historical:** Bernoulli, J. (1713). *Ars Conjectandi*. [Archive.org](https://archive.org/details/arsconjectandiop00bern)
- **Book:** Ross, S. M. (2014). *Introduction to Probability Models* (11th ed.). Academic Press. [Elsevier Link](https://www.elsevier.com/books/introduction-to-probability-models/ross/978-0-12-407948-9)
- **Book:** Wackerly, D., Mendenhall, W., & Scheaffer, R. L. (2008). *Mathematical Statistics with Applications* (7th ed.). Thomson Brooks/Cole. [Cengage Link](https://www.cengage.com/c/mathematical-statistics-with-applications-7e-wackerly/)
