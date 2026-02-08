---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/weibull-distribution/","tags":["probability","foundations"]}
---

## Definition

> [!abstract] Core Statement
> The **Weibull Distribution** models ==time-to-event data== with flexible hazard functions. It generalizes the [[30_Knowledge/Stats/01_Foundations/Exponential Distribution\|Exponential Distribution]] by allowing the hazard rate to increase, decrease, or stay constant over time.

![Weibull Distribution showing PDF for different shape parameters|500](https://upload.wikimedia.org/wikipedia/commons/5/58/Weibull_PDF.svg)
*Figure 1: Weibull PDF for various shape parameters. Shape k controls whether hazard increases or decreases with time.*

---

> [!tip] Intuition (ELI5): The Aging Machine
> Exponential distribution assumes things "don't age"—a brand new lightbulb has the same failure probability as one that's been on for years. Weibull is more realistic: it can model "infant mortality" (k < 1: failures decrease over time as weak units die early) or "wear-out" (k > 1: failures increase as the machine ages).

---

## Purpose

1. **Reliability Engineering:** Modeling component failure times
2. **Survival Analysis:** Parametric models for time-to-event data
3. **Wind Speed Modeling:** Weather and energy applications
4. **Manufacturing:** Quality control and lifetime testing

---

## When to Use

> [!success] Use Weibull Distribution When...
> - Modeling **time-to-failure** data
> - Hazard rate **changes over time** (increases or decreases)
> - Need flexibility beyond [[30_Knowledge/Stats/01_Foundations/Exponential Distribution\|Exponential Distribution]]

---

## When NOT to Use

> [!danger] Do NOT Use Weibull Distribution When...
> - **Constant hazard:** Use [[30_Knowledge/Stats/01_Foundations/Exponential Distribution\|Exponential Distribution]] (simpler, Weibull with k=1)
> - **Multiple failure modes:** Consider mixture models
> - **Non-monotonic hazard:** Weibull only handles monotonic hazards
> - **Complex survival patterns:** Use [[30_Knowledge/Stats/02_Statistical_Inference/Cox Proportional Hazards\|Cox Proportional Hazards]] for flexibility

---

## Theoretical Background

### Notation

$$
X \sim \text{Weibull}(k, \lambda)
$$

where:
- $k$ (or $\beta$): **Shape parameter** (controls hazard behavior)
- $\lambda$ (or $\eta$): **Scale parameter** (characteristic lifetime)

### Probability Density Function (PDF)

$$
f(x; k, \lambda) = \frac{k}{\lambda}\left(\frac{x}{\lambda}\right)^{k-1} e^{-(x/\lambda)^k}, \quad x \ge 0
$$

### Cumulative Distribution Function (CDF)

$$
F(x; k, \lambda) = 1 - e^{-(x/\lambda)^k}
$$

### Survival Function

$$
S(x) = P(X > x) = e^{-(x/\lambda)^k}
$$

### Hazard Function

$$
h(x) = \frac{k}{\lambda}\left(\frac{x}{\lambda}\right)^{k-1}
$$

### Parameters and Their Meaning

| Shape (k) | Hazard Behavior | Interpretation |
|-----------|-----------------|----------------|
| **k < 1** | **Decreasing** hazard | "Infant mortality" — weak units fail early, survivors are strong |
| **k = 1** | **Constant** hazard | Reduces to [[30_Knowledge/Stats/01_Foundations/Exponential Distribution\|Exponential Distribution]] — no aging |
| **k > 1** | **Increasing** hazard | "Wear-out" — older units more likely to fail |
| **k ≈ 3.6** | Approximately **Normal** | Bell-shaped reliability curve |

### Properties

| Property | Formula |
|----------|---------|
| **Mean** | $\lambda \cdot \Gamma(1 + 1/k)$ |
| **Variance** | $\lambda^2 \left[\Gamma(1+2/k) - \Gamma(1+1/k)^2\right]$ |
| **Median** | $\lambda \cdot (\ln 2)^{1/k}$ |
| **Mode** | $\lambda \cdot \left(\frac{k-1}{k}\right)^{1/k}$ for $k > 1$ |

---

## Worked Example: Component Lifetime

> [!example] Problem
> A type of capacitor has been modeled with a **Weibull(2, 1000)** distribution (k=2, λ=1000 hours).
> 
> **Questions:**
> 1. What is the probability a capacitor lasts more than **500 hours**?
> 2. What is the **mean lifetime**?

**Solution:**

**1. Survival probability P(X > 500):**
$$ S(500) = e^{-(500/1000)^2} = e^{-0.25} \approx 0.7788 $$
**Result:** ~77.9% chance of surviving 500 hours.

**2. Mean lifetime:**
$$ E[X] = \lambda \cdot \Gamma(1 + 1/k) = 1000 \cdot \Gamma(1.5) = 1000 \cdot 0.886 \approx 886.2 \text{ hours} $$

**Verification with Code:**
```python
from scipy.stats import weibull_min
from scipy.special import gamma

k, lam = 2, 1000
dist = weibull_min(c=k, scale=lam)

# P(X > 500)
print(f"P(X > 500): {1 - dist.cdf(500):.4f}")  # 0.7788

# Mean
print(f"Mean: {dist.mean():.1f} hours")  # 886.2

# Verify with formula
print(f"Mean (formula): {lam * gamma(1 + 1/k):.1f}")
```

---

## Assumptions

- [ ] **Single failure mode:** One primary mechanism of failure.
  - *Example:* Wear-out ✓ vs Multiple competing causes ✗
  
- [ ] **Monotonic hazard:** Hazard only increases or decreases.
  - *Example:* Aging ✓ vs Bathtub curve (initial + wear-out) ✗
  
- [ ] **Independence:** Failure of one unit doesn't affect others.
  - *Example:* Individual components ✓ vs Cascading failures ✗

---

## Limitations

> [!warning] Pitfalls
> 1. **Non-monotonic hazards:** Weibull can't model bathtub curves (use mixture models).
> 2. **Parameter estimation:** Poor data can lead to unstable k estimates.
> 3. **Over-simplification:** Real systems often have multiple failure modes.

---

## Python Implementation

```python
from scipy.stats import weibull_min
import numpy as np
import matplotlib.pyplot as plt

# Weibull with k=1.5, λ=10
k, lam = 1.5, 10
dist = weibull_min(c=k, scale=lam)

# Statistics
print(f"Mean: {dist.mean():.2f}")
print(f"Median: {dist.median():.2f}")
print(f"Std: {dist.std():.2f}")

# Sample data
samples = dist.rvs(1000)

# Plot different shapes
x = np.linspace(0, 30, 300)
plt.figure(figsize=(10, 6))
for k_val in [0.5, 1, 1.5, 3]:
    dist_k = weibull_min(c=k_val, scale=10)
    plt.plot(x, dist_k.pdf(x), label=f'k={k_val}')

plt.xlabel('x')
plt.ylabel('Density')
plt.title('Weibull Distribution (λ=10)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

**Expected Output:**
```
Mean: 9.03
Median: 8.33
Std: 6.17
```

---

## R Implementation

```r
# Weibull with k=1.5, λ=10 (R calls them shape and scale)
k <- 1.5
lam <- 10

# Statistics
mean_val <- lam * gamma(1 + 1/k)
print(paste("Mean:", round(mean_val, 2)))

# Sample data
samples <- rweibull(1000, shape = k, scale = lam)
print(paste("Sample mean:", round(mean(samples), 2)))

# Plot different shapes
curve(dweibull(x, shape = 0.5, scale = 10), from = 0, to = 30, 
      col = "red", lwd = 2, ylab = "Density", main = "Weibull Distributions")
curve(dweibull(x, shape = 1, scale = 10), add = TRUE, col = "blue", lwd = 2)
curve(dweibull(x, shape = 1.5, scale = 10), add = TRUE, col = "green", lwd = 2)
curve(dweibull(x, shape = 3, scale = 10), add = TRUE, col = "purple", lwd = 2)
legend("topright", legend = c("k=0.5", "k=1", "k=1.5", "k=3"),
       col = c("red", "blue", "green", "purple"), lwd = 2)
```

---

## Interpretation Guide

| Scenario | Interpretation |
|----------|----------------|
| **k = 1** | Exponential distribution; constant failure rate |
| **k < 1** | "Burn-in period"—failures decrease over time |
| **k > 1** | "Aging"—failures increase over time |
| **λ** | Scale; ~63% of units fail by time λ |

---

## Related Concepts

### Directly Related
- [[30_Knowledge/Stats/01_Foundations/Exponential Distribution\|Exponential Distribution]] - Special case (k=1)
- [[30_Knowledge/Stats/01_Foundations/Gamma Distribution\|Gamma Distribution]] - Alternative for sum of waiting times
- [[30_Knowledge/Stats/07_Causal_Inference/Survival Analysis\|Survival Analysis]] - Primary application

### Applications
- [[30_Knowledge/Stats/02_Statistical_Inference/Hazard Ratio\|Hazard Ratio]] - Hazard function comparisons
- [[30_Knowledge/Stats/02_Statistical_Inference/Kaplan-Meier Curves\|Kaplan-Meier Curves]] - Non-parametric alternative
- [[30_Knowledge/Stats/02_Statistical_Inference/Cox Proportional Hazards\|Cox Proportional Hazards]] - Semi-parametric alternative

### Other Related Topics
- [[30_Knowledge/Stats/07_Causal_Inference/Cox Proportional Hazards Model\|Cox Proportional Hazards Model]]
- [[30_Knowledge/Stats/07_Causal_Inference/Kaplan-Meier Estimator\|Kaplan-Meier Estimator]]

{ .block-language-dataview}

---

## References

1. Meeker, W. Q., & Escobar, L. A. (1998). *Statistical Methods for Reliability Data*. Wiley. [Available online](https://www.wiley.com/en-us/Statistical+Methods+for+Reliability+Data-p-9780471143390)

2. Lawless, J. F. (2003). *Statistical Models and Methods for Lifetime Data* (2nd ed.). Wiley. [Available online](https://www.wiley.com/en-us/Statistical+Models+and+Methods+for+Lifetime+Data%2C+2nd+Edition-p-9780471372158)

3. Klein, J. P., & Moeschberger, M. L. (2003). *Survival Analysis: Techniques for Censored and Truncated Data* (2nd ed.). Springer. [Available online](https://link.springer.com/book/10.1007/b97377)
