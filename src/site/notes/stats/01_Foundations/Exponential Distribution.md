---
{"dg-publish":true,"permalink":"/stats/01-foundations/exponential-distribution/","tags":["probability","distributions","continuous","survival-analysis"]}
---

## Definition

> [!abstract] Core Statement
> The **Exponential Distribution** models the ==time between events== in a Poisson process, where events occur continuously and independently at a constant average rate. It is the continuous analog of the geometric distribution.

![Exponential Distribution showing PDF for different rate parameters|500](https://upload.wikimedia.org/wikipedia/commons/f/f5/Exponential_distribution_pdf_-_public_domain.svg)
*Figure 1: Exponential PDF for various values of λ. Higher λ means events happen more frequently (shorter waiting times).*

---

> [!tip] Intuition (ELI5): The Bus Stop
> You're waiting for a bus that arrives on average every 10 minutes. The Exponential Distribution tells you the probability of waiting any given amount of time. The key insight: if you've already waited 5 minutes, the expected additional wait is *still* 10 minutes—the bus doesn't "remember" that you've been waiting.

---

## Purpose

1. Model **waiting times** (time until next event).
2. Reliability analysis: Time until failure.
3. Queueing theory: Time between arrivals.
4. Foundation for survival analysis.

---

## When to Use

> [!success] Use Exponential Distribution When...
> - Modeling **time until** an event occurs
> - Events occur at a **constant rate** ($\lambda$)
> - Events are **memoryless** (past doesn't affect future)

---

## When NOT to Use

> [!danger] Do NOT Use Exponential Distribution When...
> - **Aging effects:** Components wear out over time. Use [[stats/01_Foundations/Weibull Distribution\|Weibull Distribution]] instead.
> - **Varying rates:** Rush hour vs night traffic. Use non-homogeneous Poisson process.
> - **Event clustering:** Events trigger other events (cascades). Consider Hawkes process.
> - **Time to nth event:** Use [[stats/01_Foundations/Gamma Distribution\|Gamma Distribution]] for waiting time until $n$ events.
> - **Discrete waiting:** Counting trials until success. Use [[stats/01_Foundations/Geometric Distribution\|Geometric Distribution]].

---

## Theoretical Background

### Notation

$$
X \sim \text{Exponential}(\lambda)
$$

where $\lambda$ is the **rate parameter** (events per unit time).

Alternative: Some texts use $\beta = 1/\lambda$ (scale parameter / mean).

### Probability Density Function (PDF)

$$
f(x | \lambda) = \lambda e^{-\lambda x}, \quad x \ge 0
$$

**Understanding the Formula:**
- $\lambda$: Rate of events (higher = shorter wait times)
- $e^{-\lambda x}$: Exponential decay—probability drops as $x$ increases
- Peak at $x=0$: Most events happen quickly

### Cumulative Distribution Function (CDF)

$$
F(x | \lambda) = P(X \le x) = 1 - e^{-\lambda x}
$$

### Survival Function

$$
S(x) = P(X > x) = e^{-\lambda x}
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
| **Standard Deviation** | $\sigma = \frac{1}{\lambda}$ |
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

**Verification with Code:**
```python
from scipy.stats import expon

lambda_param = 4  # calls per hour
dist = expon(scale=1/lambda_param)  # scipy uses scale = 1/λ

# P(X <= 10 mins) = P(X <= 1/6 hour)
print(f"P(X <= 10 min): {dist.cdf(1/6):.4f}")  # 0.4866

# P(X > 30 mins) = P(X > 0.5 hour)
print(f"P(X > 30 min): {1 - dist.cdf(0.5):.4f}")  # 0.1353
```

---

## Assumptions

- [ ] **Constant rate:** Event rate $\lambda$ does not change over time.
  - *Example:* Steady customer flow ✓ vs Rush hour patterns ✗
  
- [ ] **Independence:** Events occur independently.
  - *Example:* Random arrivals ✓ vs Coordinated group arrivals ✗
  
- [ ] **Memorylessness:** Past has no influence on future waiting times.
  - *Example:* Light bulb failures ✓ vs Mechanical wear ✗

---

## Limitations

> [!warning] Pitfalls
> 1.  **The "Real World Ages" Problem:** Exponential implies components *never wear out*. A brand new bulb has the same failure probability as one that has run for 10 years. For mechanical wear, use [[stats/01_Foundations/Weibull Distribution\|Weibull Distribution]].
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
print(f"Mean: {dist.mean():.2f}")  # 2.00

# P(X > 3)
prob_gt_3 = 1 - dist.cdf(3)
print(f"P(X > 3): {prob_gt_3:.4f}")  # 0.2231

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
# P(X > 5 | X > 2) should equal P(X > 3)
cond_prob = (1 - dist.cdf(5)) / (1 - dist.cdf(2))
uncond_prob = 1 - dist.cdf(3)
print(f"Conditional P(X > 5 | X > 2): {cond_prob:.4f}")
print(f"Unconditional P(X > 3): {uncond_prob:.4f}")
# Both should be equal!
```

**Expected Output:**
```
Mean: 2.00
P(X > 3): 0.2231
Conditional P(X > 5 | X > 2): 0.2231
Unconditional P(X > 3): 0.2231
```

*The matching conditional and unconditional probabilities demonstrate the memoryless property.*

---

## R Implementation

```r
# Exponential with λ=0.5
lambda_param <- 0.5

# Mean
1 / lambda_param  # 2

# P(X > 3)
pexp(3, rate = lambda_param, lower.tail = FALSE)  # 0.2231

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
| **$\lambda = 0.2$** | Mean wait = $1/0.2 = 5$ units. Events are infrequent. |
| **High $\lambda$ (e.g., 100)** | Events happen very frequently; waiting times are tiny. |
| **Memorylessness** | "It's been quiet for an hour" does **not** mean an event is more likely now. |
| **Median < Mean** | The distribution is right-skewed; most events happen early, but some take a long time. |

---

## Related Concepts

### Directly Related
- [[stats/01_Foundations/Poisson Distribution\|Poisson Distribution]] - Number of events in fixed time
- [[stats/01_Foundations/Geometric Distribution\|Geometric Distribution]] - Discrete memoryless distribution
- [[stats/01_Foundations/Weibull Distribution\|Weibull Distribution]] - Generalizes exponential; allows aging
- [[stats/01_Foundations/Gamma Distribution\|Gamma Distribution]] - Time until $n$th event

### Applications
- [[stats/07_Causal_Inference/Survival Analysis\|Survival Analysis]] - Time-to-event modeling
- [[stats/02_Statistical_Inference/Hazard Ratio\|Hazard Ratio]] - Constant hazard in exponential
- [[stats/02_Statistical_Inference/Kaplan-Meier Curves\|Kaplan-Meier Curves]] - Non-parametric survival

### Other Related Topics
- [[stats/01_Foundations/Beta Distribution\|Beta Distribution]]
- [[stats/01_Foundations/Chi-Square Distribution\|Chi-Square Distribution]]
- [[stats/01_Foundations/Dirichlet Distribution\|Dirichlet Distribution]]
- [[stats/01_Foundations/F-Distribution\|F-Distribution]]
- [[stats/01_Foundations/Gamma Distribution\|Gamma Distribution]]

{ .block-language-dataview}

---

## References

1. Ross, S. M. (2014). *A First Course in Probability* (9th ed.). Pearson. Chapter 5. [Available online](https://www.pearson.com/us/higher-education/program/Ross-A-First-Course-in-Probability-9th-Edition/PGM220165.html)

2. Lawless, J. F. (2003). *Statistical Models and Methods for Lifetime Data* (2nd ed.). Wiley. [Available online](https://www.wiley.com/en-us/Statistical+Models+and+Methods+for+Lifetime+Data%2C+2nd+Edition-p-9780471372158)

3. Casella, G., & Berger, R. L. (2002). *Statistical Inference* (2nd ed.). Duxbury. Chapter 3. [Available online](https://www.routledge.com/Statistical-Inference/Casella-Berger/p/book/9781032593036)

### Additional Resources
- [Exponential Distribution Interactive Visualization](https://seeing-theory.brown.edu/probability-distributions/index.html)
