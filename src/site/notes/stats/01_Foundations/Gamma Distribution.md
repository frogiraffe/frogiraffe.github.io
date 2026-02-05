---
{"dg-publish":true,"permalink":"/stats/01-foundations/gamma-distribution/","tags":["probability","distributions","continuous","reliability"]}
---

## Definition

> [!abstract] Core Statement
> The **Gamma Distribution** models the time until **$k$ events** occur in a Poisson process. It is a generalization of the [[stats/01_Foundations/Exponential Distribution\|Exponential Distribution]] (which models time until *1* event). It is defined by shape parameter $\alpha$ (or $k$) and scale parameter $\theta$ (or rate $\beta = 1/\theta$).

![Gamma Distribution showing PDF for different parameters|500](https://upload.wikimedia.org/wikipedia/commons/e/e6/Gamma_distribution_pdf.svg)
*Figure 1: Gamma distribution PDF for various shape and scale parameters.*

---

> [!tip] Intuition (ELI5): Waiting for Multiple Buses
> If an Exponential distribution is "time until *one* bus arrives," then Gamma is "time until *three* buses arrive." You're waiting for multiple events, and the total time is the sum of individual waiting times.

---

## Purpose

1. **Waiting Times:** How long until 5 customers arrive? How long until the 3rd component fails?
2. **Reliability Engineering:** Modeling fatigue life where damage accumulates
3. **Bayesian Statistics:** Conjugate prior for the rate parameter of Poisson or Exponential
4. **Financial Modeling:** Asset sizes or insurance claims (skewed, positive data)

---

## When to Use

> [!success] Use Gamma Distribution When...
> - Modeling **waiting time until $k$ events** in a Poisson process
> - Data is **positive, continuous, and right-skewed**
> - Need a flexible distribution for positive-valued outcomes

---

## When NOT to Use

> [!danger] Do NOT Use Gamma Distribution When...
> - **Single event:** Use [[stats/01_Foundations/Exponential Distribution\|Exponential Distribution]] (Gamma with $\alpha=1$)
> - **Discrete counts:** Use [[stats/01_Foundations/Poisson Distribution\|Poisson Distribution]]
> - **Bounded data:** Gamma is unbounded above; use [[stats/01_Foundations/Beta Distribution\|Beta Distribution]] for [0,1]
> - **Symmetric data:** Gamma is always right-skewed

---

## Theoretical Background

### Notation

$$
X \sim \text{Gamma}(\alpha, \beta) \quad \text{or} \quad X \sim \text{Gamma}(k, \theta)
$$

**Warning:** Different fields use different parametrizations!

| Parametrization | Mean | Common Usage |
|-----------------|------|--------------|
| Shape-Rate ($\alpha, \beta$) | $\alpha / \beta$ | Bayesian, Mathematics |
| Shape-Scale ($k, \theta$) | $k \cdot \theta$ | Engineering, scipy |

Note: $\theta = 1/\beta$

### Probability Density Function (PDF)

$$
f(x | \alpha, \beta) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}, \quad x > 0
$$

where $\Gamma(\alpha)$ is the Gamma function.

### Properties

| Property | Formula (Shape-Rate) | Formula (Shape-Scale) |
|----------|---------------------|----------------------|
| **Mean** | $\alpha / \beta$ | $k \cdot \theta$ |
| **Variance** | $\alpha / \beta^2$ | $k \cdot \theta^2$ |
| **Mode** | $(\alpha-1)/\beta$ for $\alpha \ge 1$ | $(k-1)\theta$ |
| **Skewness** | $2/\sqrt{\alpha}$ | $2/\sqrt{k}$ |

### The Gamma Family Tree

| Distribution | Relationship |
|--------------|--------------|
| [[stats/01_Foundations/Exponential Distribution\|Exponential Distribution]] | Gamma with $\alpha = 1$ |
| [[stats/01_Foundations/Chi-Square Distribution\|Chi-Square Distribution]] | Gamma with $\alpha = \nu/2$, $\beta = 1/2$ |
| Erlang | Gamma where $\alpha$ is a positive integer |

---

## Worked Example: Multiple Server Failures

> [!example] Problem
> A server crashes on average once every 2 days ($\lambda = 0.5$ per day).
> **Question:** What is the probability that the **3rd crash** happens within 10 days?

**Setup:**
- Events to wait for ($k$ or $\alpha$): 3
- Rate ($\lambda$ or $\beta$): 0.5
- Scale ($\theta = 1/\beta$): 2
- This is **Gamma(3, 0.5)** in shape-rate or **Gamma(3, 2)** in shape-scale

**Verification with Code:**
```python
from scipy.stats import gamma

# Shape-scale parametrization (scipy default)
alpha = 3  # shape = number of events
scale = 2  # scale = 1/rate = mean time per event

dist = gamma(a=alpha, scale=scale)

# P(X <= 10)
prob = dist.cdf(10)
print(f"P(3rd crash within 10 days): {prob:.4f}")  # ~0.8753

# Mean time until 3rd crash
print(f"E[X]: {dist.mean():.1f} days")  # 6.0 days
```

**Interpretation:** Since average time per crash is 2 days, 3 crashes take ~6 days on average. So 10 days is plenty of time—87.5% chance.

---

## Assumptions

- [ ] **Independent events:** Each event occurs independently
  - *Example:* Random server crashes ✓ vs Cascading failures ✗
  
- [ ] **Constant rate:** Event rate doesn't change over time
  - *Example:* Steady traffic ✓ vs Rush hour patterns ✗
  
- [ ] **Positive, continuous:** Outcome must be > 0
  - *Example:* Waiting time ✓ vs Bounded percentage ✗

---

## Limitations

> [!warning] Pitfalls
> 1. **Parametrization confusion:** Always clarify if using shape-rate or shape-scale.
> 2. **Right-skew only:** Cannot model symmetric or left-skewed data.
> 3. **Unbounded:** Cannot model data with an upper limit.

---

## Python Implementation

```python
from scipy.stats import gamma
import numpy as np
import matplotlib.pyplot as plt

# Parameters (using 'a' as shape alpha)
alpha = 3  # Wait for 3 events
scale = 2  # Mean time per event (theta = 1/beta)

dist = gamma(a=alpha, scale=scale)

# Probability wait < 10
prob = dist.cdf(10)
print(f"P(X < 10): {prob:.4f}")

# Mean and variance
print(f"Mean: {dist.mean():.2f}")
print(f"Variance: {dist.var():.2f}")

# Plot different shapes
x = np.linspace(0, 20, 100)
plt.figure(figsize=(10, 6))
for (a, s) in [(1, 2), (2, 2), (3, 2), (5, 1)]:
    plt.plot(x, gamma.pdf(x, a=a, scale=s), label=f'α={a}, θ={s}')

plt.xlabel('x')
plt.ylabel('Density')
plt.title('Gamma Distribution Family')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

**Expected Output:**
```
P(X < 10): 0.8753
Mean: 6.00
Variance: 12.00
```

---

## R Implementation

```r
# Gamma Distribution (R uses shape-rate)
k <- 3     # shape
rate <- 0.5  # rate (not scale!)

# Mean = shape/rate
k / rate  # 6

# P(X < 10)
pgamma(10, shape = k, rate = rate)  # 0.8753

# Sample
samples <- rgamma(1000, shape = k, rate = rate)

# Plot Density
hist(samples, freq=FALSE, main="Gamma(3, 0.5)", col="lightgreen")
curve(dgamma(x, shape=k, rate=rate), add=TRUE, col="darkgreen", lwd=2)
```

---

## Interpretation Guide

| Scenario | Interpretation |
|----------|----------------|
| **α = 1** | Exponential distribution (waiting for 1 event) |
| **Large α** | Distribution becomes more symmetric, approaches Normal |
| **High skewness** | Most events happen quickly, but some take very long |
| **Sum of Gammas** | Gamma(α₁, β) + Gamma(α₂, β) = Gamma(α₁+α₂, β) |

---

## Related Concepts

### Directly Related
- [[stats/01_Foundations/Exponential Distribution\|Exponential Distribution]] - Special case ($\alpha=1$)
- [[stats/01_Foundations/Poisson Distribution\|Poisson Distribution]] - Counts events in fixed time (dual concept)
- [[stats/01_Foundations/Chi-Square Distribution\|Chi-Square Distribution]] - Special case ($\alpha=\nu/2$, $\beta=1/2$)
- [[stats/01_Foundations/Beta Distribution\|Beta Distribution]] - Associated conjugate prior

### Applications
- [[stats/07_Causal_Inference/Survival Analysis\|Survival Analysis]] - Time-to-event modeling
- [[stats/01_Foundations/Bayesian Statistics\|Bayesian Statistics]] - Prior for rate parameters

### Other Related Topics
- [[stats/01_Foundations/Beta Distribution\|Beta Distribution]]
- [[stats/01_Foundations/Chi-Square Distribution\|Chi-Square Distribution]]
- [[stats/01_Foundations/Dirichlet Distribution\|Dirichlet Distribution]]
- [[stats/01_Foundations/Exponential Distribution\|Exponential Distribution]]
- [[stats/01_Foundations/F-Distribution\|F-Distribution]]

{ .block-language-dataview}

---

## References

1. Hogg, R. V., & Tanis, E. A. (2010). *Probability and Statistical Inference* (8th ed.). Pearson. [Available online](https://www.pearson.com/en-us/subject-catalog/p/probability-and-statistical-inference/P200000003540/9780137981502)

2. Casella, G., & Berger, R. L. (2002). *Statistical Inference* (2nd ed.). Duxbury. [Available online](https://www.routledge.com/Statistical-Inference/Casella-Berger/p/book/9781032593036)

3. Johnson, N. L., Kotz, S., & Balakrishnan, N. (1994). *Continuous Univariate Distributions, Vol. 1* (2nd ed.). Wiley. [Available online](https://www.wiley.com/en-us/Continuous+Univariate+Distributions%2C+Volume+1%2C+2nd+Edition-p-9780471584957)
