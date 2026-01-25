---
{"dg-publish":true,"permalink":"/stats/01-foundations/exponential-distribution/","tags":["Probability-Theory","Distributions","Continuous","Survival-Analysis"]}
---


# Exponential Distribution

## Definition

> [!abstract] Core Statement
> The **Exponential Distribution** models the ==time between events== in a Poisson process, where events occur continuously and independently at a constant average rate. It is the continuous analog of the geometric distribution.

---

## Purpose

1. Model **waiting times** (time until next event).
2. Reliability analysis: Time until failure.
3. Queueing theory: Time between arrivals.
4. Foundation for survival analysis.

---

## When to Use

> [!success] Use Exponential Distribution When...
> - Modeling **time until** an event occurs.
> - Events occur at a **constant rate** ($\lambda$).
> - Events are **memoryless** (past doesn't affect future).

---

## Theoretical Background

### Notation

$$
X \sim \text{Exponential}(\lambda)
$$

where $\lambda$ is the **rate parameter** (events per unit time).

### Probability Density Function (PDF)

$$
f(x | \lambda) = \lambda e^{-\lambda x}, \quad x \ge 0
$$

### Cumulative Distribution Function (CDF)

$$
F(x | \lambda) = 1 - e^{-\lambda x}
$$

> [!important] Memoryless Property
> $$
> P(X > s + t | X > s) = P(X > t)
> $$
> **Meaning:** If you've waited $s$ time units, the probability of waiting an additional $t$ units is the same as if you just started. The distribution "forgets" how long you've already waited.

### Properties

| Property | Formula |
|----------|---------|
| **Mean** | $\mu = \frac{1}{\lambda}$ |
| **Variance** | $\sigma^2 = \frac{1}{\lambda^2}$ |
| **Median** | $\frac{\ln(2)}{\lambda} \approx \frac{0.693}{\lambda}$ |
| **Mode** | 0 (peak at origin) |

### Relationship to Poisson

If the number of events in time $t$ follows $\text{Poisson}(\lambda t)$, then the time between events follows $\text{Exponential}(\lambda)$.

---

## Worked Example: Call Center Waiting Time

> [!example] Problem
> A help desk receives calls at an average rate of **4 calls per hour** ($\lambda = 4$).
> 
> **Questions:**
> 1. What is the probability that the next call comes in **less than 10 minutes**?
> 2. What is the probability that you wait **more than 30 minutes** for a call?

**Solution:**

First, convert units to be consistent. Let's work in **hours**.
- $\lambda = 4$ calls/hour.
- 10 minutes = $10/60 = 0.1667$ hours.
- 30 minutes = $30/60 = 0.5$ hours.

**1. Probability wait < 10 mins ($P(X \le 0.1667)$):**
$$ F(x) = 1 - e^{-\lambda x} $$
$$ P(X \le 0.1667) = 1 - e^{-4 \times 0.1667} = 1 - e^{-0.667} $$
$$ e^{-0.667} \approx 0.513 $$
$$ P(X \le 0.1667) = 1 - 0.513 = 0.487 $$
**Result:** ~48.7% chance the next call arrives within 10 minutes.

**2. Probability wait > 30 mins ($P(X > 0.5)$):**
$$ P(X > x) = e^{-\lambda x} $$
$$ P(X > 0.5) = e^{-4 \times 0.5} = e^{-2} $$
$$ e^{-2} \approx 0.135 $$
**Result:** ~13.5% chance you wait more than 30 minutes.

---

## Assumptions

- [ ] **Constant rate:** Event rate $\lambda$ does not change over time.
- [ ] **Independence:** Events occur independently.
- [ ] **Memorylessness:** Past has no influence on future waiting times.

---

## Limitations

> [!warning] Pitfalls
> 1.  **The "Real World Ages" Problem:** Exponential implies components *never wear out*. A brand new bulb has the same failure probability as one that has run for 10 years. For mechanical wear, use **Weibull**.
> 2.  **Varying Rates:** If calls peak at noon and drop at night, $\lambda$ is not constant. Use a Non-Homogeneous Poisson Process.
> 3.  **Clustering:** If events trigger other events (e.g., earthquakes, stock trades), independence fails.

---

## Python Implementation

```python
from scipy.stats import expon
import numpy as np
import matplotlib.pyplot as plt

# Exponential with λ=0.5 (mean = 2)
lambda_param = 0.5
dist = expon(scale=1/lambda_param)  # scipy uses scale = 1/λ

# Mean
print(f"Mean: {dist.mean():.2f}")

# P(X > 3)
prob_gt_3 = 1 - dist.cdf(3)
print(f"P(X > 3): {prob_gt_3:.4f}")

# Visualize PDF
x = np.linspace(0, 10, 500)
plt.plot(x, dist.pdf(x), label=f'λ={lambda_param}')
plt.xlabel('Time')
plt.ylabel('Density')
plt.title('Exponential Distribution')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Memoryless Property Verification
# P(X > 5 | X > 2) = P(X > 3)
cond_prob = (1 - dist.cdf(5)) / (1 - dist.cdf(2))
uncond_prob = 1 - dist.cdf(3)
print(f"Conditional P(X > 5 | X > 2): {cond_prob:.4f}")
print(f"Unconditional P(X > 3): {uncond_prob:.4f}")
```

---

## R Implementation

```r
# Exponential with λ=0.5
lambda_param <- 0.5

# Mean
1 / lambda_param

# P(X > 3)
pexp(3, rate = lambda_param, lower.tail = FALSE)

# Visualize
curve(dexp(x, rate = lambda_param), from = 0, to = 10,
      xlab = "Time", ylab = "Density",
      main = "Exponential Distribution", lwd = 2, col = "blue")

# Random sample
rexp(10, rate = lambda_param)
```

---

## Interpretation Guide

| Scenario | Interpretation |
|----------|----------------|
| Scenario | Interpretation |
|----------|----------------|
| **$\lambda = 0.2$** | Mean wait = $1/0.2 = 5$ minutes. |
| **High $\lambda$ (e.g., 100)** | Events happen very frequently; waiting times are tiny. |
| **Memorylessness** | "It's been quiet for an hour" does **not** mean a call is more likely now. |
| **Median < Mean** | The distribution is right-skewed; most events happen early, but some take a long time. |

---

## Related Concepts

- [[stats/01_Foundations/Poisson Distribution\|Poisson Distribution]] - Number of events in fixed time.
- [[Geometric Distribution\|Geometric Distribution]] - Discrete memoryless distribution.
- [[Weibull Distribution\|Weibull Distribution]] - Generalizes exponential; allows aging.
- [[Survival Analysis\|Survival Analysis]]
