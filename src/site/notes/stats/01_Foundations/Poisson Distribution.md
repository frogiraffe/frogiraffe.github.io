---
{"dg-publish":true,"permalink":"/stats/01-foundations/poisson-distribution/","tags":["probability","distributions","discrete","foundations"]}
---

## Definition

> [!abstract] Core Statement
> The **Poisson Distribution** models the probability of a given number of ==events occurring in a fixed interval== of time or space, assuming these events occur with a known constant mean rate and independently of the time since the last event.

![Poisson Distribution showing PMF for different lambda values|500](https://upload.wikimedia.org/wikipedia/commons/1/16/Poisson_pmf.svg)
*Figure 1: Poisson PMF for various values of λ. Notice how the distribution becomes more symmetric as λ increases.*

---

> [!tip] Intuition (ELI5): The Midnight Inbox
> Imagine you get an average of 3 rare emails per night. Some nights you get 0, some nights you get 7. The Poisson Distribution is the math that predicts how many emails will be in your inbox when you wake up, based on that average rate.

---

## Purpose

1.  Model **counts** of rare events (e.g., car accidents, emails per hour, typos per page).
2.  Baseline model for **Count Regression** ([[stats/03_Regression_Analysis/Poisson Regression\|Poisson Regression]]).
3.  Approximation for [[stats/01_Foundations/Binomial Distribution\|Binomial Distribution]] when $n$ is large and $p$ is small.

---

## When to Use

> [!success] Use Poisson Distribution When...
> - Counting discrete events ($k = 0, 1, 2, \dots$)
> - Events are **independent**
> - The average rate ($\lambda$) is **constant**
> - Two events cannot occur at the exact same instant

---

## When NOT to Use

> [!danger] Do NOT Use Poisson Distribution When...
> - **Overdispersion:** Variance > Mean. Use [[stats/03_Regression_Analysis/Negative Binomial Regression\|Negative Binomial Regression]] instead.
> - **Excess zeros:** Too many zeros in data. Use [[stats/03_Regression_Analysis/Zero-Inflated Models\|Zero-Inflated Models]].
> - **Variable rate:** Rate changes over time (rush hour vs night). Use non-homogeneous Poisson.
> - **Events can cluster:** Events happen in bursts. Consider mixture models.
> - **Underdispersion:** Variance < Mean (rare). Consider binomial-based models.

---

## Theoretical Background

### Notation

$$ X \sim \text{Poisson}(\lambda) $$

where $\lambda$ (lambda) is the average number of events per interval ($\lambda > 0$).

### Probability Mass Function (PMF)

$$ P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!} $$

**Understanding the Formula:**
- $\lambda^k$: Expected contribution from $k$ events at rate $\lambda$
- $e^{-\lambda}$: Probability of no events (baseline)
- $k!$: Accounts for ordering of events
- $e \approx 2.71828$ (Euler's number)

### Properties

| Property | Formula |
|----------|---------|
| **Mean** | $E[X] = \lambda$ |
| **Variance** | $\text{Var}(X) = \lambda$ |
| **Standard Deviation** | $\sigma = \sqrt{\lambda}$ |
| **Mode** | $\lfloor \lambda \rfloor$ |
| **Skewness** | $1/\sqrt{\lambda}$ |

> [!important] Equidispersion
> A key property of Poisson is that **Mean = Variance** ($\lambda$). If Variance > Mean, data is **overdispersed** (use [[stats/03_Regression_Analysis/Negative Binomial Regression\|Negative Binomial Regression]]). If Variance < Mean, data is **underdispersed**.

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

**Verification with Code:**
```python
from scipy.stats import poisson

lam = 5
dist = poisson(mu=lam)

print(f"P(X=3): {dist.pmf(3):.4f}")  # 0.1404
print(f"P(X=0): {dist.pmf(0):.4f}")  # 0.0067

# Check mean = variance (equidispersion)
print(f"Mean: {dist.mean()}, Variance: {dist.var()}")  # Both = 5
```

---

## Assumptions

- [ ] **Independence:** Arrival of one event doesn't affect probability of another.
  - *Example:* Random customer arrivals ✓ vs Coordinated group arrivals ✗
  
- [ ] **Homogeneity:** Rate $\lambda$ is constant over the interval.
  - *Example:* Steady traffic ✓ vs Rush hour spikes ✗
  
- [ ] **No Simultaneous Events:** Events happen one at a time.
  - *Example:* Single-lane toll booth ✓ vs Multi-lane highway ✗

---

## Limitations

> [!warning] Pitfalls
> 1.  **Overdispersion:** Real data often has variance > mean (clumping). Using Poisson here yields falsely small standard errors.
> 2.  **Zero-Inflation:** If you have many more zeros than predicted (e.g., store is closed), use [[stats/03_Regression_Analysis/Zero-Inflated Models\|Zero-Inflated Poisson (ZIP)]].
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

# P(X <= 2) - Cumulative
p_le_2 = dist.cdf(2)
print(f"P(X<=2): {p_le_2:.4f}")

# Visualize
x = np.arange(0, 15)
plt.bar(x, dist.pmf(x), alpha=0.7, edgecolor='black')
plt.title(f"Poisson Distribution (λ={lam})")
plt.xlabel("Number of Events (k)")
plt.ylabel("P(X = k)")
plt.grid(axis='y', alpha=0.3)
plt.show()
```

**Expected Output:**
```
P(X=3): 0.1404
P(X<=2): 0.1247
```

*The plot shows a right-skewed distribution with mode at 4 or 5.*

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

# Visualize
x <- 0:15
barplot(dpois(x, lambda = lam), names.arg = x,
        main = paste("Poisson(λ =", lam, ")"),
        xlab = "Events", ylab = "Probability")
```

---

## Interpretation Guide

| Scenario | Interpretation |
|----------|----------------|
| **$\lambda = 1$** | Rare events. Distribution is right-skewed. Mode at 0 or 1. |
| **$\lambda = 20$** | Frequent events. Distribution looks symmetric (approaches Normal). |
| **Var > Mean** | **Overdispersion.** Model assumption violated. Use Negative Binomial. |
| **Excess zeros** | **Zero-inflation.** Use ZIP or hurdle models. |

---

## Relationship to Other Distributions

| Distribution | Relationship |
|--------------|--------------|
| [[stats/01_Foundations/Binomial Distribution\|Binomial Distribution]] | Poisson is limit as $n \to \infty$, $p \to 0$, $np = \lambda$ |
| [[stats/01_Foundations/Exponential Distribution\|Exponential Distribution]] | Time *between* Poisson events follows Exponential |
| [[stats/01_Foundations/Gamma Distribution\|Gamma Distribution]] | Time for $k$ Poisson events follows Gamma |
| [[stats/01_Foundations/Normal Distribution\|Normal Distribution]] | Poisson approaches Normal for large $\lambda$ |

---

## Related Concepts

### Directly Related
- [[stats/03_Regression_Analysis/Poisson Regression\|Poisson Regression]] - Predicting counts with covariates
- [[stats/01_Foundations/Exponential Distribution\|Exponential Distribution]] - Time between Poisson events
- [[stats/03_Regression_Analysis/Negative Binomial Regression\|Negative Binomial Regression]] - For overdispersed counts
- [[stats/01_Foundations/Binomial Distribution\|Binomial Distribution]] - Poisson is limit as $n \to \infty$, $p \to 0$

### Applications
- [[stats/03_Regression_Analysis/Zero-Inflated Models\|Zero-Inflated Models]] - Handling excess zeros
- [[stats/01_Foundations/Hurdle Models\|Hurdle Models]] - Two-part count models
- [[stats/01_Foundations/Overdispersion\|Overdispersion]] - When Poisson assumptions fail

### Other Related Topics
- [[stats/01_Foundations/Bernoulli Distribution\|Bernoulli Distribution]]
- [[stats/01_Foundations/Categorical Distribution\|Categorical Distribution]]
- [[stats/01_Foundations/Discrete Uniform Distribution\|Discrete Uniform Distribution]]
- [[stats/01_Foundations/Geometric Distribution\|Geometric Distribution]]
- [[stats/01_Foundations/Hypergeometric Distribution\|Hypergeometric Distribution]]

{ .block-language-dataview}

---

## References

1. Poisson, S. D. (1837). *Recherches sur la Probabilité des Jugements en Matière Criminelle et en Matière Civile*. [Available online](https://gallica.bnf.fr/ark:/12148/bpt6k1102148)

2. Ross, S. M. (2014). *A First Course in Probability* (9th ed.). Pearson. Chapter 4. [Available online](https://www.pearson.com/en-us/subject-catalog/p/first-course-in-probability-a/P200000006198/)

3. Casella, G., & Berger, R. L. (2002). *Statistical Inference* (2nd ed.). Duxbury. Chapter 3. [Available online](https://www.cengage.com/c/statistical-inference-2e-casella/)

### Additional Resources
- [Poisson Distribution Interactive Visualization](https://seeing-theory.brown.edu/probability-distributions/index.html#section2)
- [Khan Academy: Poisson Distribution](https://www.khanacademy.org/math/statistics-probability/random-variables-stats-library/poisson-distribution)
