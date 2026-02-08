---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/bernoulli-distribution/","tags":["probability","foundations"]}
---

## Definition

> [!abstract] Core Statement
> The **Bernoulli Distribution** is the simplest discrete probability distribution, modeling a ==single trial with two possible outcomes==: success (1) or failure (0). It is the building block for the [[30_Knowledge/Stats/01_Foundations/Binomial Distribution\|Binomial Distribution]].

---

> [!tip] Intuition (ELI5): The Single Coin Flip
> A Bernoulli trial is like flipping a coin once. You either get heads (success, $X=1$) or tails (failure, $X=0$). That's it—no second chances, just one flip.

---

## Purpose

1. **Building block** for more complex distributions ([[30_Knowledge/Stats/01_Foundations/Binomial Distribution\|Binomial Distribution]], [[30_Knowledge/Stats/01_Foundations/Geometric Distribution\|Geometric Distribution]])
2. Model **binary outcomes** (yes/no, pass/fail, click/no-click)
3. Foundation for **logistic regression** and classification

---

## When to Use

> [!success] Use Bernoulli Distribution When...
> - You have a **single trial** with two outcomes
> - Outcome is **binary** (success/failure)
> - You know or want to estimate the probability of success $p$

---

## When NOT to Use

> [!danger] Do NOT Use Bernoulli Distribution When...
> - **Multiple trials:** Use [[30_Knowledge/Stats/01_Foundations/Binomial Distribution\|Binomial Distribution]] for $n > 1$ trials
> - **More than 2 outcomes:** Use [[30_Knowledge/Stats/01_Foundations/Categorical Distribution\|Categorical Distribution]] or [[30_Knowledge/Stats/01_Foundations/Multinomial Distribution\|Multinomial Distribution]]
> - **Continuous outcomes:** Use appropriate continuous distribution

---

## Theoretical Background

### Notation

$$
X \sim \text{Bernoulli}(p)
$$

where $p \in [0, 1]$ is the probability of success.

### Probability Mass Function (PMF)

$$
P(X = k) = p^k (1-p)^{1-k} \quad \text{for } k \in \{0, 1\}
$$

Or equivalently:
$$
P(X = 1) = p, \quad P(X = 0) = 1 - p = q
$$

### Properties

| Property | Formula |
|----------|---------|
| **Mean** | $E[X] = p$ |
| **Variance** | $\text{Var}(X) = p(1-p) = pq$ |
| **Standard Deviation** | $\sigma = \sqrt{p(1-p)}$ |
| **Skewness** | $\frac{1-2p}{\sqrt{p(1-p)}}$ |
| **Support** | $X \in \{0, 1\}$ |

### Maximum Variance

Variance is maximized when $p = 0.5$:
$$
\text{Var}_{max} = 0.5 \times 0.5 = 0.25
$$

---

## Worked Example: Email Click Rate

> [!example] Problem
> You send a marketing email. Historical data shows **20% click rate** ($p = 0.2$).
> 
> **Questions:**
> 1. What is the probability a random recipient **clicks**?
> 2. What is the probability they **don't click**?
> 3. What is the expected value and variance?

**Solution:**

**1. P(Click):** $P(X=1) = p = 0.2$ → **20%**

**2. P(No Click):** $P(X=0) = 1 - p = 0.8$ → **80%**

**3. Expected Value & Variance:**
- $E[X] = p = 0.2$
- $\text{Var}(X) = p(1-p) = 0.2 \times 0.8 = 0.16$

---

## Assumptions

- [ ] **Single trial:** Only one experiment/observation
- [ ] **Two outcomes:** Mutually exclusive and exhaustive
- [ ] **Fixed probability:** $p$ is known and constant

---

## Limitations

> [!warning] Pitfalls
> 1. **Too simple:** Real-world problems often involve multiple trials → use Binomial
> 2. **Independence assumption:** When modeling multiple Bernoulli trials, they must be independent
> 3. **Fixed p:** If probability changes over trials, Bernoulli model is invalid

---

## Python Implementation

```python
from scipy.stats import bernoulli
import numpy as np
import matplotlib.pyplot as plt

# Bernoulli(p=0.3)
p = 0.3
dist = bernoulli(p)

# PMF
print(f"P(X=0): {dist.pmf(0):.2f}")  # 0.70
print(f"P(X=1): {dist.pmf(1):.2f}")  # 0.30

# Mean and Variance
print(f"Mean: {dist.mean():.2f}")    # 0.30
print(f"Variance: {dist.var():.2f}") # 0.21

# Random samples (10 trials)
samples = dist.rvs(size=10)
print(f"Samples: {samples}")  # e.g., [0, 1, 0, 0, 1, 0, 0, 0, 1, 0]

# Visualize
x = [0, 1]
plt.bar(x, [dist.pmf(0), dist.pmf(1)], alpha=0.7, edgecolor='black')
plt.xticks(x, ['Failure (0)', 'Success (1)'])
plt.ylabel('Probability')
plt.title(f'Bernoulli Distribution (p={p})')
plt.ylim(0, 1)
plt.show()
```

**Expected Output:**
```
P(X=0): 0.70
P(X=1): 0.30
Mean: 0.30
Variance: 0.21
```

---

## R Implementation

```r
# Bernoulli(p=0.3)
p <- 0.3

# P(X=1)
dbinom(1, size=1, prob=p)  # 0.3

# P(X=0)
dbinom(0, size=1, prob=p)  # 0.7

# Random samples
rbinom(10, size=1, prob=p)

# Mean and Variance
p           # Mean = 0.3
p * (1-p)   # Variance = 0.21

# Visualize
barplot(c(1-p, p), names.arg=c("0", "1"),
        main=paste("Bernoulli(p =", p, ")"),
        ylab="Probability", col="steelblue")
```

---

## Relationship to Other Distributions

| Distribution | Relationship |
|--------------|--------------|
| [[30_Knowledge/Stats/01_Foundations/Binomial Distribution\|Binomial Distribution]] | Sum of $n$ i.i.d. Bernoulli trials: $\sum_{i=1}^n X_i \sim \text{Binomial}(n, p)$ |
| [[30_Knowledge/Stats/01_Foundations/Geometric Distribution\|Geometric Distribution]] | Number of Bernoulli trials until first success |
| [[30_Knowledge/Stats/01_Foundations/Beta Distribution\|Beta Distribution]] | Conjugate prior for Bernoulli parameter $p$ |

---

## Related Concepts

### Directly Related
- [[30_Knowledge/Stats/01_Foundations/Binomial Distribution\|Binomial Distribution]] - Multiple Bernoulli trials
- [[30_Knowledge/Stats/01_Foundations/Geometric Distribution\|Geometric Distribution]] - Trials until first success
- [[30_Knowledge/Stats/01_Foundations/Beta Distribution\|Beta Distribution]] - Bayesian prior for $p$
- [[30_Knowledge/Stats/03_Regression_Analysis/Logistic Regression\|Logistic Regression]] - Models Bernoulli probabilities

### Applications
- A/B Testing - Each user response is Bernoulli
- Classification - Binary classification outputs

---

## References

1. Ross, S. M. (2014). *A First Course in Probability* (9th ed.). Pearson. Chapter 4. [Available online](https://www.pearson.com/us/higher-education/program/Ross-A-First-Course-in-Probability-9th-Edition/PGM220165.html)

2. Casella, G., & Berger, R. L. (2002). *Statistical Inference* (2nd ed.). Duxbury. [Available online](https://www.routledge.com/Statistical-Inference/Casella-Berger/p/book/9781032593036)
