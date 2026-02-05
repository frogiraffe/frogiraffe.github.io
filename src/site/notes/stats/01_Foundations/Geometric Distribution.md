---
{"dg-publish":true,"permalink":"/stats/01-foundations/geometric-distribution/","tags":["probability","distributions","discrete","foundations"]}
---

## Definition

> [!abstract] Core Statement
> The **Geometric Distribution** models the number of ==Bernoulli trials needed to get the first success==. It answers: "How many coin flips until I get heads?"

![Geometric Distribution showing PMF for different success probabilities|500](https://upload.wikimedia.org/wikipedia/commons/4/4b/Geometric_pmf.svg)
*Figure 1: Geometric PMF for different values of p. Higher p means fewer trials needed on average.*

---

> [!tip] Intuition (ELI5): The Dice Roll
> You're rolling a die waiting to get a 6. Some days you get lucky on the first roll, other days it takes 10+ rolls. The Geometric Distribution tells you the probability of needing exactly $k$ rolls before you finally succeed.

---

## Purpose

1. Model **waiting times** until first success (discrete)
2. Foundation for understanding [[stats/03_Regression_Analysis/Negative Binomial Regression\|Negative Binomial Regression]]
3. Discrete analog of [[stats/01_Foundations/Exponential Distribution\|Exponential Distribution]]

---

## When to Use

> [!success] Use Geometric Distribution When...
> - Counting **number of trials until first success**
> - Each trial is independent with same probability $p$
> - Trials are discrete (countable)

---

## When NOT to Use

> [!danger] Do NOT Use Geometric Distribution When...
> - **Fixed number of trials:** Use [[stats/01_Foundations/Binomial Distribution\|Binomial Distribution]]
> - **Continuous time:** Use [[stats/01_Foundations/Exponential Distribution\|Exponential Distribution]]
> - **Waiting for $r$ successes:** Use Negative Binomial Distribution
> - **Non-constant probability:** If $p$ changes over trials

---

## Theoretical Background

### Notation

$$
X \sim \text{Geometric}(p)
$$

where $p$ is the probability of success on each trial.

### Probability Mass Function (PMF)

$$
P(X = k) = (1-p)^{k-1} p \quad \text{for } k = 1, 2, 3, \ldots
$$

**Understanding the Formula:**
- $(1-p)^{k-1}$: Probability of failing the first $k-1$ trials
- $p$: Probability of succeeding on trial $k$

> [!note] Alternative Definition
> Some texts define $X$ as the number of *failures* before first success:
> $$P(X = k) = (1-p)^k p \quad \text{for } k = 0, 1, 2, \ldots$$

### Properties

| Property | Value |
|----------|-------|
| **Mean** | $E[X] = \frac{1}{p}$ |
| **Variance** | $\text{Var}(X) = \frac{1-p}{p^2}$ |
| **Support** | $X \in \{1, 2, 3, \ldots\}$ |
| **Mode** | 1 (always most likely to succeed on first try) |

### Memoryless Property

$$
P(X > m+n \mid X > m) = P(X > n)
$$

**Meaning:** If you've already failed $m$ times, the expected additional trials is the same as if you just started. The distribution "forgets" past failures.

---

## Worked Example: Free Throw Shooter

> [!example] Problem
> A basketball player makes 70% of free throws ($p = 0.7$).
> 
> **Questions:**
> 1. What is the probability of making the **first basket on the 3rd attempt**?
> 2. **Expected attempts** until first make?

**Solution:**

**1. P(First success on 3rd attempt):**
$$ P(X=3) = (1-0.7)^{3-1} \times 0.7 = (0.3)^2 \times 0.7 = 0.09 \times 0.7 = 0.063 $$
**Result:** ~6.3% chance

**2. Expected attempts:**
$$ E[X] = \frac{1}{p} = \frac{1}{0.7} = 1.43 $$
**Result:** On average, ~1.4 attempts until first make.

**Verification with Code:**
```python
from scipy.stats import geom

p = 0.7
dist = geom(p)

print(f"P(X=3): {dist.pmf(3):.4f}")  # 0.0630
print(f"E[X]: {dist.mean():.2f}")    # 1.43
print(f"Var(X): {dist.var():.2f}")   # 0.61
```

---

## Assumptions

- [ ] **Binary Outcome:** Each trial is success or failure.
  - *Example:* Make/miss shot ✓ vs Score 0-3 points ✗
  
- [ ] **Independence:** Trials don't affect each other.
  - *Example:* Dice rolls ✓ vs Getting tired over time ✗
  
- [ ] **Constant $p$:** Same probability each trial.
  - *Example:* Fair die ✓ vs Improving with practice ✗

---

## Limitations

> [!warning] Pitfalls
> 1. **Long tails:** Geometric has no upper bound—extreme waiting times are possible.
> 2. **Independence assumption:** In real life, fatigue or learning can change $p$.
> 3. **Definition confusion:** Always clarify if counting trials or failures.

---

## Python Implementation

```python
from scipy.stats import geom
import numpy as np
import matplotlib.pyplot as plt

p = 0.3  # Success probability
dist = geom(p)

# PMF: P(X=5) = first success on 5th trial
print(f"P(X=5): {dist.pmf(5):.4f}")

# Mean and variance
print(f"Mean: {dist.mean():.2f}, Variance: {dist.var():.2f}")

# P(X > 10) - still waiting after 10 trials
print(f"P(X > 10): {1 - dist.cdf(10):.4f}")

# Visualize
x = np.arange(1, 20)
plt.bar(x, dist.pmf(x), alpha=0.7, edgecolor='black')
plt.xlabel('Number of Trials Until First Success')
plt.ylabel('Probability')
plt.title(f'Geometric Distribution (p={p})')
plt.grid(axis='y', alpha=0.3)
plt.show()
```

**Expected Output:**
```
P(X=5): 0.0720
Mean: 3.33, Variance: 7.78
P(X > 10): 0.0282
```

---

## R Implementation

```r
p <- 0.3

# PMF: P(X=5)
# Note: R's dgeom counts failures (k-1), so use k-1
dgeom(4, prob = p)  # 0.0720

# Mean
1 / p  # 3.33

# P(X > 10)
1 - pgeom(9, prob = p)  # 0.0282

# Simulation
samples <- rgeom(10000, p) + 1  # Add 1 to convert to trials
mean(samples)  # ~3.33
```

---

## Interpretation Guide

| Scenario | Interpretation |
|----------|----------------|
| **High $p$ (e.g., 0.9)** | Success comes quickly; E[X] = 1.1 trials |
| **Low $p$ (e.g., 0.1)** | Long wait; E[X] = 10 trials, high variance |
| **Memorylessness** | "Bad luck streak" doesn't increase future success probability |
| **Var >> Mean** | When $p$ is small, outcomes are highly variable |

---

## Comparison to Related Distributions

| Distribution | Models | Key Difference |
|--------------|--------|----------------|
| [[stats/01_Foundations/Bernoulli Distribution\|Bernoulli Distribution]] | Single trial | Geometric = "repeat until success" |
| [[stats/01_Foundations/Binomial Distribution\|Binomial Distribution]] | Fixed n trials | Binomial = fixed n; Geometric = random n |
| [[stats/01_Foundations/Exponential Distribution\|Exponential Distribution]] | Continuous waiting time | Discrete vs continuous |
| Negative Binomial | Trials until r successes | Geometric = special case (r=1) |

---

## Related Concepts

### Directly Related
- [[stats/01_Foundations/Bernoulli Distribution\|Bernoulli Distribution]] - Single trial building block
- [[stats/01_Foundations/Binomial Distribution\|Binomial Distribution]] - Fixed number of trials
- [[stats/01_Foundations/Exponential Distribution\|Exponential Distribution]] - Continuous analog

### Applications
- [[stats/03_Regression_Analysis/Negative Binomial Regression\|Negative Binomial Regression]] - Generalization for count data
- [[stats/07_Causal_Inference/Survival Analysis\|Survival Analysis]] - Discrete failure times

### Other Related Topics
- [[stats/01_Foundations/Bernoulli Distribution\|Bernoulli Distribution]]
- [[stats/01_Foundations/Binomial Distribution\|Binomial Distribution]]
- [[stats/01_Foundations/Categorical Distribution\|Categorical Distribution]]
- [[stats/01_Foundations/Discrete Uniform Distribution\|Discrete Uniform Distribution]]
- [[stats/01_Foundations/Hypergeometric Distribution\|Hypergeometric Distribution]]

{ .block-language-dataview}

---

## References

1. Ross, S. M. (2014). *A First Course in Probability* (9th ed.). Pearson. Chapter 4. [Available online](https://www.pearson.com/us/higher-education/program/Ross-A-First-Course-in-Probability-9th-Edition/PGM220165.html)

2. Casella, G., & Berger, R. L. (2002). *Statistical Inference* (2nd ed.). Duxbury. [Available online](https://www.routledge.com/Statistical-Inference/Casella-Berger/p/book/9781032593036)
