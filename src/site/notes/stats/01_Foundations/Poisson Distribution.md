---
{"dg-publish":true,"permalink":"/stats/01-foundations/poisson-distribution/","tags":["Probability-Theory","Distributions","Discrete"]}
---

## Definition

> [!abstract] Core Statement
> The **Poisson Distribution** models the probability of a given number of ==events occurring in a fixed interval== of time or space, assuming these events occur with a known constant mean rate and independently of the time since the last event.

![Poisson Distribution PMF](https://upload.wikimedia.org/wikipedia/commons/1/16/Poisson_pmf.svg)

---

> [!tip] Intuition (ELI5): The Midnight Inbox
> Imagine you get an average of 3 rare emails per night. Some nights you get 0, some nights you get 7. The Poisson Distribution is the math that predicts how many emails will be in your inbox when you wake up, based on that average rate.

---

## Purpose

1.  Model **counts** of rare events (e.g., car accidents, emails per hour, typos per page).
2.  Baseline model for **Count Regression** ([[stats/03_Regression_Analysis/Poisson Regression\|Poisson Regression]]).
3.  Approximation for Binomial Distribution when $n$ is large and $p$ is small.

---

## When to Use

> [!success] Use Poisson When...
> - Counting discrete events ($k = 0, 1, 2, \dots$).
> - Events are **independent**.
> - The average rate ($\lambda$) is **constant**.
> - Two events cannot occur at the exact same instant.

---

## Theoretical Background

### Notation

$$ X \sim \text{Poisson}(\lambda) $$

where $\lambda$ (lambda) is the average number of events per interval ($\lambda > 0$).

### Probability Mass Function (PMF)

$$ P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!} $$

- $e \approx 2.71828$ (Euler's number).
- $k!$ is the factorial of $k$.

### Properties

| Property | Formula |
|----------|---------|
| **Mean** | $E[X] = \lambda$ |
| **Variance** | $\text{Var}(X) = \lambda$ |
| **Standard Deviation** | $\sigma = \sqrt{\lambda}$ |
| **Mode** | $\lfloor \lambda \rfloor$ |

> [!important] Equidispersion
> A key property of Poisson is that **Mean = Variance** ($\lambda$). If Variance > Mean, data is **Overdispersed** (use Negative Binomial).

---

## Worked Example: Fast Food Drive-Thru

> [!example] Problem
> A fast food drive-thru gets an average of **$\lambda = 5$ cars per minute** during lunch.
> 
> **Questions:**
> 1. What is the probability of exactly **3 cars** arriving in a minute?
> 2. What is the probability of **0 cars** (a quiet minute)?

**Solution:**

**1. Exactly 3 cars:**
$$ P(X=3) = \frac{5^3 e^{-5}}{3!} = \frac{125 \times 0.0067}{6} \approx 0.140 $$
**Result:** ~14% chance.

**2. Zero cars:**
$$ P(X=0) = \frac{5^0 e^{-5}}{0!} = \frac{1 \times 0.0067}{1} \approx 0.0067 $$
**Result:** ~0.67% chance (very rare to be empty).

---

## Assumptions

- [ ] **Independence:** Arrival of one event doesn't affecting probability of another.
- [ ] **Homogeneity:** Rate $\lambda$ is constant over the interval.
- [ ] **No Simultaneous Events:** Events happen one at a time.

---

## Limitations

> [!warning] Pitfalls
> 1.  **Overdispersion:** Real data often has variance > mean (clumping). Using Poisson here yields falsely small standard errors.
> 2.  **Zero-Inflation:** If you have many more zeros than predicted (e.g., store is closed), use **Zero-Inflated Poisson (ZIP)**.
> 3.  **Variable Rate:** If rate changes over time (rush hour vs night), use non-homogeneous Poisson process.

---

## Python Implementation

```python
from scipy.stats import poisson
import numpy as np
import matplotlib.pyplot as plt

# Lambda = 5
lam = 5
dist = poisson(mu=lam)

# P(X = 3)
p_3 = dist.pmf(3)
print(f"P(X=3): {p_3:.4f}")

# Visualize
x = np.arange(0, 15)
plt.bar(x, dist.pmf(x), alpha=0.7)
plt.title(f"Poisson Distribution (λ={lam})")
plt.xlabel("Number of Events")
plt.ylabel("Probability")
plt.show()
```

---

## R Implementation

```r
# Lambda = 5
lam <- 5

# P(X = 3)
dpois(3, lambda = lam)

# P(X <= 2)
ppois(2, lambda = lam)

# Random Sample
rpois(10, lambda = lam)
```

---

## Interpretation Guide

| Scenario | Interpretation |
|----------|----------------|
| **$\lambda = 1$** | Rare events. Distribution is right-skewed. Mode at 0 or 1. |
| **$\lambda = 20$** | Frequent events. Distribution looks symmetric (Normal). |
| **Var > Mean** | **Overdispersion.** Model assumption violated. |

---

## Related Concepts

- [[stats/03_Regression_Analysis/Poisson Regression\|Poisson Regression]] - Predicting counts with covariates.
- [[stats/01_Foundations/Exponential Distribution\|Exponential Distribution]] - Time *between* Poisson events.
- [[stats/03_Regression_Analysis/Negative Binomial Regression\|Negative Binomial Regression]] - For overdispersed counts.
- [[stats/01_Foundations/Binomial Distribution\|Binomial Distribution]] - Poisson is limit as $n \to \infty, p \to 0$.

---

## References

- **Book:** Ross, S. M. (2014). *A First Course in Probability* (9th ed.). Pearson. [Pearson Link](https://www.pearson.com/en-us/subject-catalog/p/first-course-in-probability-a/P200000006198/)
- **Book:** Casella, G., & Berger, R. L. (2002). *Statistical Inference* (2nd ed.). Duxbury. [Cengage](https://www.cengage.com/c/statistical-inference-2e-casella/9780534243128/)
- **Historical:** Poisson, S. D. (1837). *Recherches sur la Probabilité des Jugements en Matière Criminelle et en Matière Civile*. [Gallica](https://gallica.bnf.fr/ark:/12148/bpt6k1102148)
