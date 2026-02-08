---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/multinomial-distribution/","tags":["probability","foundations"]}
---

## Definition

> [!abstract] Core Statement
> The **Multinomial Distribution** generalizes the binomial to ==k > 2 categories==. It models the counts of outcomes when sampling n times from k categories with fixed probabilities.

![Multinomial: Generalizes binomial to multiple categories|500](https://upload.wikimedia.org/wikipedia/commons/8/84/Multinomial_distribution_pmf.png)
*Figure 1: Multinomial models counts across K categories in n trials.*

$$
P(X_1=x_1, \ldots, X_k=x_k) = \frac{n!}{x_1! \cdots x_k!} \prod_{i=1}^k p_i^{x_i}
$$

Where $\sum x_i = n$ and $\sum p_i = 1$.

---

> [!tip] Intuition (ELI5): The Bag of Colored Marbles
> You have a bag with red, blue, and green marbles in some proportions. You draw 100 marbles (putting each back). The Multinomial tells you the probability of getting exactly 30 red, 45 blue, and 25 green.

---

## Purpose

1. **Count data:** Occurrences across multiple categories
2. **Topic modeling:** Word counts in documents (LDA)
3. **Genetics:** Genotype frequencies
4. **Survey analysis:** Response patterns

---

## When to Use

> [!success] Use Multinomial Distribution When...
> - Counting outcomes across **K categories** in **n trials**
> - Trials are **independent** with **constant probabilities**
> - Need to model **joint counts** (not just one category)

---

## When NOT to Use

> [!danger] Do NOT Use Multinomial Distribution When...
> - **Only 2 categories:** Use [[30_Knowledge/Stats/01_Foundations/Binomial Distribution\|Binomial Distribution]]
> - **Single trial:** Use [[30_Knowledge/Stats/01_Foundations/Categorical Distribution\|Categorical Distribution]]
> - **Without replacement:** Use Multivariate Hypergeometric
> - **Continuous outcomes:** Multinomial is for discrete counts

---

## Theoretical Background

### Notation

$$
\mathbf{X} \sim \text{Multinomial}(n, \mathbf{p})
$$

where:
- $n$ = number of trials
- $\mathbf{p} = (p_1, \ldots, p_k)$ = probability vector
- $\mathbf{X} = (X_1, \ldots, X_k)$ = count vector

### Properties

| Property | Formula |
|----------|---------|
| **E[$X_i$]** | $np_i$ |
| **Var[$X_i$]** | $np_i(1-p_i)$ |
| **Cov[$X_i$, $X_j$]** | $-np_ip_j$ (for $i \neq j$) |
| **Constraint** | $\sum_i X_i = n$ |

### Marginal Distributions

Each $X_i$ marginally follows a [[30_Knowledge/Stats/01_Foundations/Binomial Distribution\|Binomial Distribution]]:
$$X_i \sim \text{Binomial}(n, p_i)$$

### Negative Covariance

The counts are **negatively correlated**: if one category has more observations, others must have fewer (since they sum to $n$).

---

## Worked Example: Rolling Dice 60 Times

> [!example] Problem
> Roll a fair die 60 times.
> 
> **Questions:**
> 1. What is the expected count for each face?
> 2. What is $P(X_1 = 10, X_2 = 10, X_3 = 10, X_4 = 10, X_5 = 10, X_6 = 10)$?

**Solution:**

Parameters: $n = 60$, $p_i = 1/6$ for all $i$.

**1. Expected counts:**
$$E[X_i] = np_i = 60 \times \frac{1}{6} = 10$$

**2. Probability of exactly 10 of each:**
```python
from scipy.stats import multinomial
import numpy as np

n = 60
probs = [1/6] * 6
dist = multinomial(n, probs)

# P(exactly 10 of each)
counts = [10] * 6
prob = dist.pmf(counts)
print(f"P(10 of each): {prob:.6f}")  # ~0.0010
```

---

## Assumptions

- [ ] **Fixed n:** Total number of trials is known.
  - *Example:* 60 dice rolls ✓ vs Rolling until bored ✗
  
- [ ] **Independence:** Trials don't affect each other.
  - *Example:* Dice rolls ✓ vs Contagious outcomes ✗
  
- [ ] **Constant probabilities:** $\mathbf{p}$ is fixed across trials.
  - *Example:* Same die ✓ vs Changing dice ✗

---

## Python Implementation

```python
import numpy as np
from scipy.stats import multinomial

# ========== SAMPLING ==========
n = 100
probs = [0.2, 0.3, 0.35, 0.15]

# Multiple samples
samples = np.random.multinomial(n, probs, size=1000)
print(f"Sample means: {samples.mean(axis=0)}")
print(f"Expected: {np.array(probs) * n}")

# ========== PMF ==========
dist = multinomial(n, probs)
x = np.array([20, 30, 35, 15])
prob = dist.pmf(x)
print(f"P(X = {x}) = {prob:.6e}")

# ========== COVARIANCE ==========
print(f"Var[X_1]: {n * probs[0] * (1 - probs[0]):.2f}")
print(f"Cov[X_1, X_2]: {-n * probs[0] * probs[1]:.2f}")
```

**Expected Output:**
```
Sample means: [20.05, 29.93, 35.02, 15.00]  # Approximately
Expected: [20. 30. 35. 15.]
P(X = [20 30 35 15]) = 4.38e-03
Var[X_1]: 16.00
Cov[X_1, X_2]: -6.00
```

---

## R Implementation

```r
# Sampling
samples <- rmultinom(1, size = 100, prob = c(0.2, 0.3, 0.35, 0.15))
print(samples)

# PMF
prob <- dmultinom(c(20, 30, 35, 15), prob = c(0.2, 0.3, 0.35, 0.15))
print(paste("P(X = [20,30,35,15]):", format(prob, scientific = TRUE)))
```

---

## Interpretation Guide

| Scenario | Interpretation |
|----------|----------------|
| **Large n** | Counts approach expected values $np_i$ |
| **Negative covariance** | High count in one category → lower in others |
| **Marginal binomial** | Each $X_i$ individually is Binomial |
| **Chi-square test** | Use to test if observed ~ expected |

---

## Applications

| Application | Categories |
|-------------|------------|
| **Dice rolls** | 6 outcomes |
| **Survey responses** | Multiple choice options |
| **Topic modeling** | Document-word counts |
| **Genetics** | Genotype frequencies |
| **A/B/C testing** | User actions across variants |

---

## Related Concepts

### Directly Related
- [[30_Knowledge/Stats/01_Foundations/Binomial Distribution\|Binomial Distribution]] - Special case (k=2)
- [[30_Knowledge/Stats/01_Foundations/Categorical Distribution\|Categorical Distribution]] - Single trial (n=1)
- [[30_Knowledge/Stats/01_Foundations/Dirichlet Distribution\|Dirichlet Distribution]] - Conjugate prior for $\mathbf{p}$

### Tests
- [[30_Knowledge/Stats/02_Statistical_Inference/Chi-Square Test of Independence\|Chi-Square Test of Independence]] - Goodness-of-fit for multinomial
- [[30_Knowledge/Stats/02_Statistical_Inference/G-Test\|G-Test]] - Log-likelihood alternative

### Other Related Topics

{ .block-language-dataview}

---

## References

1. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. Section 2.2. [Available online](https://www.springer.com/gp/book/9780387310732)

2. Agresti, A. (2013). *Categorical Data Analysis* (3rd ed.). Wiley. [Available online](https://www.wiley.com/en-us/Categorical+Data+Analysis%2C+3rd+Edition-p-9780470463635)
