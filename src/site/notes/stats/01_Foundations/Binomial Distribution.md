---
{"dg-publish":true,"permalink":"/stats/01-foundations/binomial-distribution/","tags":["probability","distributions","discrete","foundations"]}
---

## Definition

> [!abstract] Core Statement
> The **Binomial Distribution** models the number of ==successes in a fixed number of independent trials==, where each trial has the same probability of success. It answers: "If I flip a coin 10 times, what's the probability of getting exactly 6 heads?"

![Binomial Distribution showing probability mass for different values of k|500](https://upload.wikimedia.org/wikipedia/commons/7/75/Binomial_distribution_pmf.svg)
*Figure 1: Binomial probability mass function for various parameter combinations. Notice how distribution shape changes with $p$ and $n$.*

---

> [!tip] Intuition (ELI5): The "Scoreboard"
> Imagine you are playing a game where you get 10 attempts to score. Each attempt is independent (like a coin flip). The Binomial Distribution is the scoreboard that tells you the probability of finishing with exactly 0, 1, 2... or 10 points. 

---

## Purpose

1. Model **discrete outcomes** with two possibilities (success/failure).
2. Calculate probabilities for quality control, polling, and A/B testing.
3. Foundation for **proportion tests** and [[stats/02_Statistical_Inference/Confidence Intervals\|Confidence Intervals]] for proportions.

---

## When to Use

> [!success] Use Binomial Distribution When...
> - **Fixed number** of independent trials ($n$).
> - Each trial has **two outcomes** (success or failure).
> - **Constant probability** of success ($p$) across trials.
> - Trials are **independent**.

---

## When NOT to Use

> [!danger] Do NOT Use Binomial Distribution When...
> - **Trials are dependent:** E.g., sampling without replacement from a small population. Use [[stats/01_Foundations/Hypergeometric Distribution\|Hypergeometric Distribution]] instead.
> - **Probability changes:** If success probability varies between trials (e.g., a basketball player gets tired). Consider Beta-Binomial Distribution.
> - **More than two outcomes:** If trials have 3+ possible outcomes. Use [[stats/01_Foundations/Multinomial Distribution\|Multinomial Distribution]].
> - **Continuous outcomes:** If measuring time, weight, etc. Binomial is for discrete counts only.
> - **Unspecified trial count:** If asking "how many trials until first success?", use [[stats/01_Foundations/Geometric Distribution\|Geometric Distribution]].

---

## Theoretical Background

### Notation

$$
X \sim \text{Binomial}(n, p)
$$

where:
- $n$ = number of trials
- $p$ = probability of success on each trial
- $X$ = number of successes

### Probability Mass Function (PMF)

$$
P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}
$$

where $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ is the binomial coefficient.

**Understanding the Formula:**

The formula might look intimidating, but it has a beautiful logic:

$$P(X = k) = \underbrace{\binom{n}{k}}_{\text{Ways to arrange}} \times \underbrace{p^k}_{\text{Success}} \times \underbrace{(1-p)^{n-k}}_{\text{Failure}}$$

Breaking it down:
- $\binom{n}{k}$: In how many ways can we choose which $k$ trials out of $n$ will be successes? This is the binomial coefficient.
- $p^k$: What is the probability that those $k$ specific trials succeed? Since trials are independent, we multiply $p$ by itself $k$ times.
- $(1-p)^{n-k}$: What is the probability that the remaining $n-k$ trials fail? Each has probability $(1-p)$, so we multiply $(1-p)$ by itself $n-k$ times.

The formula essentially says: count all possible arrangements of $k$ successes, then multiply by the probability of any one specific arrangement.

### Properties

| Property | Formula |
|----------|---------|
| **Mean** | $\mu = np$ |
| **Variance** | $\sigma^2 = np(1-p)$ |
| **Standard Deviation** | $\sigma = \sqrt{np(1-p)}$ |
| **Skewness** | $\frac{1-2p}{\sqrt{np(1-p)}}$ |

### Approximations

> [!tip] Normal Approximation
> For large $n$ and $p$ not near 0 or 1:
> $$
> \text{Binomial}(n, p) \approx N(np, np(1-p))
> $$
> **Rule of thumb:** Valid if $np \ge 5$ and $n(1-p) \ge 5$.

> [!tip] Poisson Approximation
> For large $n$ and small $p$ (rare events):
> $$
> \text{Binomial}(n, p) \approx \text{Poisson}(\lambda = np)
> $$

---

## Worked Example: Quality Control

> [!example] Problem
> A factory produces light bulbs with a **defect rate of 2%** ($p=0.02$).
> A quality inspector randomly selects a batch of **20 bulbs** ($n=20$).
> 
> **Questions:**
> 1. What is the probability that **exactly 2** bulbs are defective?
> 2. What is the probability that **at least 1** bulb is defective?

**Solution:**

**1. Probability of exactly 2 defects ($X=2$):**
$$ P(X=2) = \binom{20}{2} (0.02)^2 (0.98)^{18} $$
$$ \binom{20}{2} = \frac{20 \times 19}{2} = 190 $$
$$ P(X=2) = 190 \times 0.0004 \times 0.695 = 0.0528 $$
**Result:** ~5.3% chance of finding exactly 2 bad bulbs.

**2. Probability of at least 1 defect ($X \ge 1$):**
It's easier to calculate $1 - P(X=0)$.
$$ P(X=0) = \binom{20}{0} (0.02)^0 (0.98)^{20} = 1 \times 1 \times 0.6676 = 0.6676 $$
$$ P(X \ge 1) = 1 - 0.6676 = 0.3324 $$
**Result:** ~33.2% chance of finding at least one bad bulb in the batch.

**Verification with Code:**
```python
from scipy.stats import binom

# Problem setup
n = 20  # batch size
p = 0.02  # defect rate
dist = binom(n, p)

# Question 1: P(X = 2)
prob_exactly_2 = dist.pmf(2)
print(f"P(X = 2): {prob_exactly_2:.4f}")  # Should match ~0.0528

# Question 2: P(X >= 1)
prob_at_least_1 = 1 - dist.pmf(0)
print(f"P(X >= 1): {prob_at_least_1:.4f}")  # Should match ~0.3324

# Bonus: Full distribution
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, n+1)
plt.bar(x, dist.pmf(x), alpha=0.7, edgecolor='black')
plt.axvline(2, color='red', linestyle='--', label='Exactly 2 defects')
plt.xlabel('Number of Defective Bulbs')
plt.ylabel('Probability')
plt.title('Binomial(n=20, p=0.02): Quality Control Distribution')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.show()
```

---

## Assumptions

- [ ] **Fixed $n$:** Number of trials is predetermined before the experiment begins.
  - *Example:* "I will flip this coin exactly 10 times" ✓ vs "I will flip until I get heads" ✗
  
- [ ] **Binary Outcomes:** Each trial results in exactly one of two outcomes (success or failure).
  - *Example:* Pass/Fail ✓ vs Letter Grades (A/B/C/D/F) ✗
  
- [ ] **Independence:** The outcome of one trial does not affect the probability of success in another trial.
  - *Example:* Coin flips ✓ vs Drawing cards without replacement ✗
  
- [ ] **Constant $p$:** Probability of success is the same for all trials.
  - *Example:* Fair coin (always 0.5) ✓ vs Tired basketball player (decreasing accuracy) ✗

---

## Limitations

> [!warning] Pitfalls
> 1.  **Independence Violation (Clustering):** If defects happen in clusters (e.g., a machine breaks down and produces 10 bad bulbs in a row), independence is violated. The Binomial model will underestimate the probability of extreme outcomes.
> 2.  **Overdispersion:** If the observed variance is significantly larger than $np(1-p)$, the data is overdispersed. Use the **Beta-Binomial Distribution** instead.
> 3.  **Variable $p$:** If the defect rate changes during the day, a simple Binomial model is invalid.


---

## Python Implementation

```python
from scipy.stats import binom
import numpy as np
import matplotlib.pyplot as plt

# Binomial(n=10, p=0.5)
n, p = 10, 0.5
dist = binom(n, p)

# P(X = 6)
prob_6 = dist.pmf(6)
print(f"P(X = 6 | n={n}, p={p}): {prob_6:.4f}")

# P(X <= 7)
prob_le_7 = dist.cdf(7)
print(f"P(X <= 7): {prob_le_7:.4f}")

# Visualize PMF
x = np.arange(0, n+1)
plt.bar(x, dist.pmf(x), alpha=0.7, edgecolor='black')
plt.xlabel('Number of Successes (k)')
plt.ylabel('P(X = k)')
plt.title(f'Binomial Distribution (n={n}, p={p})')
plt.xticks(x)
plt.grid(axis='y', alpha=0.3)
plt.show()
```

**Expected Output:**
```
P(X = 6 | n=10, p=0.5): 0.2051
P(X <= 7): 0.9453
```

*The resulting plot will show a symmetric bell-shaped distribution centered at k=5, which is the expected value (np = 10 × 0.5).*

---

## R Implementation

```r
# Binomial(n=10, p=0.5)
n <- 10
p <- 0.5

# P(X = 6)
dbinom(6, size = n, prob = p)

# P(X <= 7)
pbinom(7, size = n, prob = p)

# Random sample
rbinom(20, size = n, prob = p)

# Visualize
x <- 0:n
plot(x, dbinom(x, size = n, prob = p), type = "h", lwd = 3,
     xlab = "Number of Successes", ylab = "Probability",
     main = paste("Binomial(n=", n, ", p=", p, ")", sep=""))
points(x, dbinom(x, size = n, prob = p), pch = 16, col = "blue")
```

---

## Interpretation Guide

| Scenario | Interpretation |
|----------|----------------|
| **$n=100$, $p=0.5$** | **Mean = 50.** Typical coin flip scenario. Symmetric distribution. |
| **$n=1000$, $p=0.001$** | **Rare Events.** Distribution is highly right-skewed. Better modeled by Poisson. |
| **P(X $\ge$ 1)** | **"At Least One" Risk.** Even with low $p$, large $n$ makes failure likely. |
| **Spread vs Mean** | Standard Deviation $\sigma \propto \sqrt{n}$. Percentage error decreases as $n$ grows. |

---

## Related Concepts

### Directly Related
- [[stats/01_Foundations/Normal Distribution\|Normal Distribution]] - Approximation for large $n$
- [[stats/01_Foundations/Poisson Distribution\|Poisson Distribution]] - Approximation for rare events ($p$ small, $n$ large)
- [[stats/01_Foundations/Hypergeometric Distribution\|Hypergeometric Distribution]] - For sampling without replacement
- [[stats/01_Foundations/Bernoulli Distribution\|Bernoulli Distribution]] - Special case with $n=1$
- [[stats/01_Foundations/Central Limit Theorem (CLT)\|Central Limit Theorem (CLT)]] - Why Binomial approximates Normal for large $n$

### Applications
- [[stats/02_Statistical_Inference/Hypothesis Testing (P-Value & CI)\|Hypothesis Testing (P-Value & CI)]] - Proportion tests rely on Binomial distribution
- [[stats/02_Statistical_Inference/Confidence Intervals\|Confidence Intervals]] - For population proportions

### Other Related Topics
- [[stats/04_Supervised_Learning/Activation Functions\|Activation Functions]]
- [[stats/04_Supervised_Learning/AdaBoost\|AdaBoost]]
- [[stats/03_Regression_Analysis/Adaptive Lasso\|Adaptive Lasso]]
- [[stats/03_Regression_Analysis/AIC (Akaike Information Criterion)\|AIC (Akaike Information Criterion)]]
- [[stats/04_Supervised_Learning/Anomaly Detection\|Anomaly Detection]]

{ .block-language-dataview}

---

## References

1. Ross, S. M. (2014). *A First Course in Probability* (9th ed.). Pearson. Chapter 4: The Binomial Distribution. [Available online](https://www.pearson.com/us/higher-education/program/Ross-A-First-Course-in-Probability-9th-Edition/PGM220165.html)

2. DeGroot, M. H., & Schervish, M. J. (2012). *Probability and Statistics* (4th ed.). Pearson. Chapter 5: Special Distributions. [Available online](https://www.pearson.com/en-us/subject-catalog/p/probability-and-statistics/P200000003541/9780137981694)

3. Casella, G., & Berger, R. L. (2002). *Statistical Inference* (2nd ed.). Duxbury. Chapter 3: Common Families of Distributions. [Available online](https://www.routledge.com/Statistical-Inference/Casella-Berger/p/book/9781032593036)

### Additional Resources
- [Binomial Distribution Interactive Visualization](https://seeing-theory.brown.edu/probability-distributions/index.html#section2)
- [Khan Academy: Binomial Distribution](https://www.khanacademy.org/math/statistics-probability/random-variables-stats-library/binomial-random-variables)
