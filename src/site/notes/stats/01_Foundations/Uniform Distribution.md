---
{"dg-publish":true,"permalink":"/stats/01-foundations/uniform-distribution/","tags":["probability","distributions","continuous","foundations"]}
---

## Definition

> [!abstract] Core Statement
> The **Uniform Distribution** (Continuous) assigns ==equal probability== to all values in a specified interval $[a, b]$. Every outcome in the range is equally likely. It is the "flat" distribution.

![Uniform Distribution showing constant PDF|500](https://upload.wikimedia.org/wikipedia/commons/9/96/Uniform_Distribution_PDF_SVG.svg)
*Figure 1: Uniform distribution has constant density—a perfect rectangle.*

---

> [!tip] Intuition (ELI5): The Spinning Wheel
> Imagine spinning a wheel that stops at a random position. If the wheel is perfectly balanced, every position is equally likely. That's Uniform. There's no "hot spot" or preferred region—pure randomness across the range.

---

## Purpose

1. Model scenarios where all outcomes are **equally likely**
2. Generate **random numbers** for simulations (basis of random number generators)
3. Serve as a **non-informative prior** in [[stats/01_Foundations/Bayesian Statistics\|Bayesian Statistics]]
4. Baseline for comparing other distributions

---

## When to Use

> [!success] Use Uniform Distribution When...
> - All values in a range are **equally probable**
> - No information suggests one value is more likely than another
> - Generating random samples for [[stats/01_Foundations/Monte Carlo Simulation\|Monte Carlo Simulation]]

---

## When NOT to Use

> [!danger] Do NOT Use Uniform Distribution When...
> - **Data has structure:** Real data is rarely uniform—consider Normal, Exponential, etc.
> - **Discrete outcomes:** Use [[stats/01_Foundations/Discrete Uniform Distribution\|Discrete Uniform Distribution]] (e.g., dice)
> - **Prior knowledge exists:** Using Uniform as a "lazy prior" can be inappropriate
> - **Unbounded data:** Uniform has finite bounds $[a, b]$

---

## Theoretical Background

### Notation

$$
X \sim \text{Uniform}(a, b) \quad \text{or} \quad X \sim U(a, b)
$$

where $a$ is the minimum and $b$ is the maximum.

### Probability Density Function (PDF)

$$
f(x | a, b) = 
\begin{cases}
\frac{1}{b - a} & \text{if } a \le x \le b \\
0 & \text{otherwise}
\end{cases}
$$

**Constant density** across the interval. Height = $\frac{1}{b-a}$.

### Cumulative Distribution Function (CDF)

$$
F(x | a, b) = 
\begin{cases}
0 & \text{if } x < a \\
\frac{x - a}{b - a} & \text{if } a \le x \le b \\
1 & \text{if } x > b
\end{cases}
$$

### Properties

| Property | Formula |
|----------|---------|
| **Mean** | $\mu = \frac{a + b}{2}$ (midpoint) |
| **Variance** | $\sigma^2 = \frac{(b - a)^2}{12}$ |
| **Median** | $\frac{a + b}{2}$ (same as mean) |
| **Mode** | Any value in $[a, b]$ (all equally likely) |

### Standard Uniform: $U(0, 1)$

Special case where $a=0, b=1$. Used as the foundation for generating all other random variables via **inverse transform sampling**.

---

## Worked Example: Waiting for a Train

> [!example] Problem
> A commuter train arrives every **15 minutes**. You arrive at the station at a random time, so your waiting time $X$ is uniformly distributed between **0 and 15 minutes**.
> 
> **Questions:**
> 1. What is the probability you wait **less than 5 minutes**?
> 2. What is the average (expected) waiting time?

**Solution:**

Parameters: $a = 0$, $b = 15$. Distribution $X \sim U(0, 15)$.
PDF height = $\frac{1}{15 - 0} = \frac{1}{15}$.

**1. Probability wait < 5 mins ($P(X < 5)$):**
$$ P(X < 5) = \text{Base} \times \text{Height} = (5 - 0) \times \frac{1}{15} = \frac{5}{15} = \frac{1}{3} \approx 0.333 $$
**Result:** ~33.3% chance of a short wait.

**2. Average Waiting Time ($E[X]$):**
$$ \mu = \frac{a + b}{2} = \frac{0 + 15}{2} = 7.5 \text{ minutes} $$
**Result:** On average, you will wait 7.5 minutes.

**Verification with Code:**
```python
from scipy.stats import uniform

a, b = 0, 15
dist = uniform(loc=a, scale=b-a)  # scipy uses loc=a, scale=b-a

print(f"P(X < 5): {dist.cdf(5):.4f}")  # 0.3333
print(f"Mean: {dist.mean():.2f}")      # 7.50
```

---

## Assumptions

The Uniform distribution is a **model choice**:
- [ ] You believe all outcomes in $[a, b]$ are **equally probable**.
  - *Example:* Random arrival time ✓ vs Peak hour arrivals ✗
  
- [ ] No prior knowledge favors any particular value.
  - *Example:* Spinning wheel ✓ vs Historical sales data ✗

---

## Limitations

> [!warning] Pitfalls
> 1.  **The "Lazy Prior" Fallacy:** Assuming a distribution is uniform just because you have *no data* can be dangerous (the "Principle of Indifference"). Sometimes reality is bell-shaped or power-law.
> 2.  **Pseudo-randomness:** Computer "uniform" generators are deterministic algorithms. For cryptography, you need cryptographically secure RNGs.
> 3.  **Boundary Bias:** Real-world metrics rarely have hard "walls" like $a$ and $b$ with zero probability outside.

---

## Python Implementation

```python
from scipy.stats import uniform
import numpy as np
import matplotlib.pyplot as plt

# Uniform on [2, 8]
a, b = 2, 8
dist = uniform(loc=a, scale=b-a)  # scipy uses loc=a, scale=b-a

# Mean and Variance
print(f"Mean: {dist.mean():.2f}")  # 5.00
print(f"Variance: {dist.var():.2f}")  # 3.00

# P(3 < X < 6)
prob = dist.cdf(6) - dist.cdf(3)
print(f"P(3 < X < 6): {prob:.4f}")  # 0.5000

# Visualize PDF
x = np.linspace(0, 10, 500)
plt.figure(figsize=(10, 6))
plt.plot(x, dist.pdf(x), lw=3, label=f'Uniform({a}, {b})')
plt.fill_between(x, dist.pdf(x), alpha=0.3)
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Uniform Distribution')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

**Expected Output:**
```
Mean: 5.00
Variance: 3.00
P(3 < X < 6): 0.5000
```

---

## R Implementation

```r
# Uniform on [2, 8]
a <- 2
b <- 8

# Mean
(a + b) / 2  # 5

# Variance
(b - a)^2 / 12  # 3

# P(3 < X < 6)
punif(6, min = a, max = b) - punif(3, min = a, max = b)  # 0.5

# Visualize PDF
curve(dunif(x, min = a, max = b), from = 0, to = 10, lwd = 3,
      xlab = "x", ylab = "Density",
      main = paste("Uniform(", a, ", ", b, ")", sep=""), col = "blue")

# Random Sample
runif(10, min = a, max = b)
```

---

## Interpretation Guide

| Scenario | Interpretation |
|----------|----------------|
| **$U(0, 1)$** | Standard reference. If p-values are uniform, $H_0$ is true. |
| **Mean = Median** | In Uniform, Mean = Median = Midpoint (symmetry). |
| **Variance** | Depends heavily on range width: $\sigma^2 = (b-a)^2/12$ |
| **Constant PDF** | "Flat" likelihood. Every value is equally surprising. |

---

## Related Concepts

### Directly Related
- [[stats/01_Foundations/Discrete Uniform Distribution\|Discrete Uniform Distribution]] - For discrete outcomes (e.g., dice)
- [[stats/01_Foundations/Bayesian Statistics\|Bayesian Statistics]] - Uniform used as non-informative prior
- [[stats/01_Foundations/Monte Carlo Simulation\|Monte Carlo Simulation]] - Random number generation

### Contrast With
- [[stats/01_Foundations/Normal Distribution\|Normal Distribution]] - Uniform is the opposite (flat vs bell)
- [[stats/01_Foundations/Exponential Distribution\|Exponential Distribution]] - Has decay structure

### Other Related Topics
- [[stats/01_Foundations/Beta Distribution\|Beta Distribution]]
- [[stats/01_Foundations/Chi-Square Distribution\|Chi-Square Distribution]]
- [[stats/01_Foundations/Dirichlet Distribution\|Dirichlet Distribution]]
- [[stats/01_Foundations/Exponential Distribution\|Exponential Distribution]]
- [[stats/01_Foundations/F-Distribution\|F-Distribution]]

{ .block-language-dataview}

---

## References

1. Ross, S. M. (2014). *A First Course in Probability* (9th ed.). Pearson. [Available online](https://www.pearson.com/en-us/subject-catalog/p/first-course-in-probability-a/P200000006198/)

2. DeGroot, M. H., & Schervish, M. J. (2012). *Probability and Statistics* (4th ed.). Pearson. [Available online](https://www.pearson.com/en-us/subject-catalog/p/probability-and-statistics/P200000006277/)
