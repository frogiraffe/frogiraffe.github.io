---
{"dg-publish":true,"permalink":"/stats/01-foundations/bernoulli-distribution/","tags":["probability","distributions","discrete","foundations"]}
---

## Definition

> [!abstract] Core Statement
> The **Bernoulli Distribution** is the simplest discrete probability distribution. It models a **single trial** with exactly two possible outcomes: Success ($k=1$) with probability $p$, and Failure ($k=0$) with probability $q = 1-p$.

---

> [!tip] Intuition (ELI5): The Switch
> The Bernoulli distribution is the simplest "On/Off" switch of probability. It represents exactly one coin flip or one "Yes/No" question. Everything else in probability is just a bunch of these switches connected together.

![Bernoulli Distribution showing PMF and CDF|500](https://upload.wikimedia.org/wikipedia/commons/b/b6/PMF_and_CDF_of_a_bernouli_distribution.png)
*Figure 1: PMF and CDF of a Bernoulli distribution for different values of p.*

---

## Purpose

1.  **Building Block:** It is the fundamental atom of probability.
    -   $n$ Bernoulli trials → **[[stats/01_Foundations/Binomial Distribution\|Binomial Distribution]]**.
    -   Trials until success → **[[stats/01_Foundations/Geometric Distribution\|Geometric Distribution]]**.
2.  **Binary Classification:** Modeling Yes/No outcomes ([[stats/03_Regression_Analysis/Logistic Regression\|Logistic Regression]]).

---

## When to Use

> [!success] Use Bernoulli Distribution When...
> - Modeling a **single trial** with exactly two outcomes
> - Each trial is independent
> - Probability of success is known and constant

---

## When NOT to Use

> [!danger] Do NOT Use Bernoulli Distribution When...
> - **Multiple trials:** Use [[stats/01_Foundations/Binomial Distribution\|Binomial Distribution]] for $n$ trials.
> - **More than two outcomes:** Use [[stats/01_Foundations/Categorical Distribution\|Categorical Distribution]] or [[stats/01_Foundations/Multinomial Distribution\|Multinomial Distribution]].
> - **Counting until success:** Use [[stats/01_Foundations/Geometric Distribution\|Geometric Distribution]].
> - **Continuous outcomes:** Bernoulli is strictly for discrete binary outcomes.

---

## Theoretical Background

### Notation

$$
X \sim \text{Bernoulli}(p)
$$

where $p$ is the probability of success ($0 \le p \le 1$).

### Probability Mass Function (PMF)

$$ P(X=k) = p^k (1-p)^{1-k} \quad \text{for } k \in \{0, 1\} $$

**Understanding the Formula:**
- When $k=1$ (success): $P(X=1) = p^1(1-p)^0 = p$
- When $k=0$ (failure): $P(X=0) = p^0(1-p)^1 = 1-p$

### Properties

| Property | Formula | Intuition |
|----------|---------|-----------|
| **Mean** ($E[X]$) | $p$ | If $p=0.8$, average outcome represents 80% success. |
| **Variance** ($\sigma^2$) | $p(1-p)$ | Max variance at $p=0.5$ (most uncertainty). Min at $p=0$ or $p=1$ (certainty). |
| **Standard Deviation** | $\sqrt{p(1-p)}$ | Spread around the mean |
| **Skewness** | $\frac{1-2p}{\sqrt{p(1-p)}}$ | Symmetric only when $p=0.5$ |

---

## Worked Example: Weighted Coin

> [!example] Problem
> A weighted coin lands Heads (1) 70% of the time ($p=0.7$).
> 
> **Questions:**
> 1. What is the probability of Heads?
> 2. What is the variance?

**Solution:**

**1. Probability of Heads:** $P(X=1) = p = 0.7$

**2. Variance:**
$$ \sigma^2 = 0.7 \times (1 - 0.7) = 0.7 \times 0.3 = 0.21 $$

*Contrast:* If fair coin ($p=0.5$): $\sigma^2 = 0.5 \times 0.5 = 0.25$. (Fair coin is more unpredictable).

**Verification with Code:**
```python
from scipy.stats import bernoulli

p = 0.7
dist = bernoulli(p)

print(f"P(X=1): {dist.pmf(1):.4f}")  # 0.7000
print(f"Variance: {dist.var():.4f}")  # 0.2100
```

---

## Assumptions

- [ ] **Binary Outcome:** Exactly 0 or 1. No "Edge" or "Maybe".
  - *Example:* Coin flip (H/T) ✓ vs Die roll (1-6) ✗
  
- [ ] **Single Trial:** Only happens once.
  - *Example:* One exam (Pass/Fail) ✓ vs Multiple exams ✗
  
- [ ] **Known Probability:** The value of $p$ is known or estimated.
  - *Example:* Fair coin ($p=0.5$) ✓ vs Unknown biased coin ✗

---

## Limitations

> [!warning] Pitfalls
> 1. **Oversimplification:** Real-world outcomes often have more than two possibilities or degrees of success.
> 2. **Independence Assumption:** Sequential trials may be correlated (e.g., fatigue affects performance).
> 3. **Fixed Probability:** The success probability may change over time or conditions.

---

## Python Implementation

```python
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
import numpy as np

p = 0.3
rv = bernoulli(p)

# Moments
mean, var = rv.stats(moments='mv')
print(f"Mean: {float(mean):.4f}, Variance: {float(var):.4f}")

# PMF
print(f"P(Success): {rv.pmf(1):.4f}")
print(f"P(Failure): {rv.pmf(0):.4f}")

# Simulate 1000 trials
samples = rv.rvs(1000)
print(f"Observed proportion: {samples.mean():.3f}")
```

**Expected Output:**
```
Mean: 0.3000, Variance: 0.2100
P(Success): 0.3000
P(Failure): 0.7000
Observed proportion: ~0.300
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

# Calculate sample mean
mean(flips)
# Should be close to 0.7
```

---

## Interpretation Guide

| Scenario | Interpretation |
|----------|----------------|
| **$p = 0.5$** | Maximum uncertainty. Fair coin. Variance = 0.25 (maximum). |
| **$p = 0.9$** | Highly biased toward success. Low variance = 0.09. |
| **$p = 0.01$** | Rare event (like winning lottery). Almost always 0. |
| **Sample mean ≠ $p$** | Sampling variability; converges to $p$ as $n \to \infty$ ([[stats/01_Foundations/Law of Large Numbers\|Law of Large Numbers]]). |

---

## Related Concepts

### Directly Related
- [[stats/01_Foundations/Binomial Distribution\|Binomial Distribution]] - Sum of $n$ Bernoullis
- [[stats/01_Foundations/Geometric Distribution\|Geometric Distribution]] - Trials until first success
- [[stats/01_Foundations/Categorical Distribution\|Categorical Distribution]] - Generalization to >2 outcomes

### Applications
- [[stats/03_Regression_Analysis/Logistic Regression\|Logistic Regression]] - Modeling binary outcomes
- [[stats/03_Regression_Analysis/Binary Logistic Regression\|Binary Logistic Regression]] - Bernoulli likelihood
- [[stats/02_Statistical_Inference/A-B Testing\|A-B Testing]] - Comparing conversion rates

### Other Related Topics
- [[stats/01_Foundations/Categorical Distribution\|Categorical Distribution]]
- [[stats/01_Foundations/Discrete Uniform Distribution\|Discrete Uniform Distribution]]
- [[stats/01_Foundations/Hypergeometric Distribution\|Hypergeometric Distribution]]
- [[stats/01_Foundations/Multinomial Distribution\|Multinomial Distribution]]
- [[stats/01_Foundations/Poisson Distribution\|Poisson Distribution]]

{ .block-language-dataview}

---

## References

1. Bernoulli, J. (1713). *Ars Conjectandi*. [Available online](https://archive.org/details/arsconjectandiop00bern)

2. Ross, S. M. (2014). *Introduction to Probability Models* (11th ed.). Academic Press. [Available online](https://www.elsevier.com/books/introduction-to-probability-models/ross/978-0-12-407948-9)

3. Wackerly, D., Mendenhall, W., & Scheaffer, R. L. (2008). *Mathematical Statistics with Applications* (7th ed.). Thomson Brooks/Cole. [Available online](https://www.cengage.com/c/mathematical-statistics-with-applications-7e-wackerly/)
