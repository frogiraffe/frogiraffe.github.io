---
{"dg-publish":true,"permalink":"/stats/01-foundations/continuous-uniform-distribution/","tags":["Probability","Distributions","Statistics"]}
---


## Definition

> [!abstract] Core Statement
> The **Continuous Uniform Distribution** assigns ==equal probability density== to all values in an interval $[a, b]$. Every value in the range is equally likely.

---

## Probability Density Function

$$
f(x) = \begin{cases} 
\frac{1}{b-a} & \text{for } a \leq x \leq b \\
0 & \text{otherwise}
\end{cases}
$$

---

## Properties

| Property | Formula |
|----------|---------|
| **Mean** | $\mu = \frac{a + b}{2}$ |
| **Variance** | $\sigma^2 = \frac{(b-a)^2}{12}$ |
| **Standard Deviation** | $\sigma = \frac{b-a}{\sqrt{12}}$ |
| **CDF** | $F(x) = \frac{x-a}{b-a}$ for $a \leq x \leq b$ |

---

## Python Implementation

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# ========== DEFINE DISTRIBUTION ==========
a, b = 0, 10
uniform = stats.uniform(loc=a, scale=b-a)

# ========== STATISTICS ==========
print(f"Mean: {uniform.mean()}")
print(f"Variance: {uniform.var()}")
print(f"Std: {uniform.std()}")

# ========== PROBABILITIES ==========
print(f"P(X < 3): {uniform.cdf(3):.4f}")
print(f"P(2 < X < 7): {uniform.cdf(7) - uniform.cdf(2):.4f}")

# ========== RANDOM SAMPLES ==========
samples = uniform.rvs(size=1000)

# ========== PLOT ==========
x = np.linspace(-1, 11, 1000)
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(x, uniform.pdf(x), 'b-', linewidth=2)
plt.fill_between(x, uniform.pdf(x), alpha=0.3)
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Uniform PDF')

plt.subplot(1, 2, 2)
plt.hist(samples, bins=30, density=True, alpha=0.7, edgecolor='black')
plt.plot(x, uniform.pdf(x), 'r-', linewidth=2)
plt.xlabel('x')
plt.title('Samples vs PDF')

plt.tight_layout()
plt.show()
```

---

## R Implementation

```r
# ========== STATISTICS ==========
a <- 0
b <- 10

mean <- (a + b) / 2
var <- (b - a)^2 / 12

# ========== PROBABILITIES ==========
dunif(5, min = a, max = b)   # PDF at x=5
punif(3, min = a, max = b)   # P(X < 3)
qunif(0.5, min = a, max = b) # Median (50th percentile)

# ========== RANDOM SAMPLES ==========
samples <- runif(1000, min = a, max = b)
hist(samples, breaks = 30, probability = TRUE)
```

---

## Applications

| Application | Example |
|-------------|---------|
| **Random number generation** | Base for most RNG |
| **Rounding error** | Error uniformly distributed |
| **Waiting time** | Arrival within a time window |
| **Simulations** | Monte Carlo sampling |

---

## Related Concepts

- [[stats/01_Foundations/Discrete Uniform Distribution\|Discrete Uniform Distribution]] — Integer version
- [[stats/01_Foundations/Normal Distribution\|Normal Distribution]] — Bell curve (not uniform)
- [[stats/01_Foundations/Monte Carlo Simulation\|Monte Carlo Simulation]] — Uses uniform samples

---

## References

- **Book:** Ross, S. M. (2014). *Introduction to Probability and Statistics for Engineers and Scientists* (5th ed.). Academic Press.
