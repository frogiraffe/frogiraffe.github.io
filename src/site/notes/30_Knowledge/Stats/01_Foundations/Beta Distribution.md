---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/beta-distribution/","tags":["probability","foundations"]}
---

## Definition

> [!abstract] Core Statement
> The **Beta Distribution** is a continuous distribution defined on the interval **[0, 1]**. It is parametrized by two shape parameters, $\alpha$ (Alpha) and $\beta$ (Beta). It is widely used to model **probabilities** or proportions, and serves as the **conjugate prior** for the Bernoulli, Binomial, and Geometric distributions.

![Beta Distribution showing PDF for different parameter combinations|500](https://upload.wikimedia.org/wikipedia/commons/f/f3/Beta_distribution_pdf.svg)
*Figure 1: Beta distribution PDF for various (α, β) combinations. Notice the flexibility in shape.*

---

> [!tip] Intuition (ELI5): Prior Beliefs as Coin Flips
> Imagine you've flipped a coin some number of times before. $\alpha$ is like your "prior heads" count plus 1, and $\beta$ is your "prior tails" count plus 1. Beta(1,1) means "I've never seen this coin"—anything is possible. Beta(100, 100) means "I've seen 99 heads and 99 tails"—I'm confident it's fair.

---

## Purpose

1.  **Bayesian Inference:** Representing belief about a probability (e.g., "I think the conversion rate is between 2% and 5%")
2.  **Modeling Rates:** Click-through rates, batting averages, defect rates
3.  **Project Management:** PERT distribution (Optimistic vs Pessimistic estimates)

---

## When to Use

> [!success] Use Beta Distribution When...
> - Modeling **probabilities** or proportions (values in [0, 1])
> - Using **Bayesian inference** with binomial/Bernoulli data
> - Need a flexible prior for unknown probability parameters

---

## When NOT to Use

> [!danger] Do NOT Use Beta Distribution When...
> - **Data outside [0, 1]:** Beta is strictly bounded. Use appropriate transformation.
> - **Integer counts:** Use [[30_Knowledge/Stats/01_Foundations/Binomial Distribution\|Binomial Distribution]] for counts, not rates.
> - **More than 2 outcomes:** Use [[30_Knowledge/Stats/01_Foundations/Dirichlet Distribution\|Dirichlet Distribution]] for multi-category probabilities.
> - **Heavy tails needed:** Beta has bounded support—can't model extreme values.

---

## Theoretical Background

### Notation

$$
X \sim \text{Beta}(\alpha, \beta)
$$

where $\alpha > 0$ and $\beta > 0$ are shape parameters.

### Probability Density Function (PDF)

$$
f(x | \alpha, \beta) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}
$$

where $B(\alpha, \beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}$ is the Beta function.

### Understanding Parameters as Pseudo-Counts

You can think of $\alpha$ and $\beta$ as **pseudo-counts** of prior history:
- $\alpha - 1$: Number of prior successes
- $\beta - 1$: Number of prior failures
- $\alpha + \beta - 2$: Effective sample size of prior

### Properties

| Property | Formula |
|----------|---------|
| **Mean** | $\mu = \frac{\alpha}{\alpha + \beta}$ |
| **Mode** | $\frac{\alpha - 1}{\alpha + \beta - 2}$ (for $\alpha, \beta > 1$) |
| **Variance** | $\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$ |
| **Support** | $x \in [0, 1]$ |

### Common Shapes

| Parameters | Shape | Interpretation |
|------------|-------|----------------|
| **Beta(1, 1)** | Flat (Uniform) | "I have no idea." (All probs equally likely) |
| **Beta(2, 2)** | Mound at 0.5 | "Weakly believe it's fair." |
| **Beta(100, 100)** | Sharp Spike at 0.5 | "Strongly believe it's fair." |
| **Beta(10, 1)** | Skewed Right (near 1) | "Almost certain to succeed." |
| **Beta(1, 10)** | Skewed Left (near 0) | "Almost certain to fail." |
| **Beta(0.5, 0.5)** | U-Shape (bathtub) | "Either 0 or 1, but not middle." |

---

## Worked Example: Batting Average

> [!example] Problem
> A new baseball player appears. **Estimate his batting average.**
> 
> 1. **Prior:** League average is 0.260. We use a prior of **Beta(81, 219)** (Mean = $81/300 = 0.27$, effectively 300 "prior at-bats").
> 2. **Data:** In his first game, he hits **1 out of 1**. (100% average!).

**Naive vs Bayesian:**
- **Naive Mean:** $1/1 = 1.000$ (Way too high)

**Bayesian Update:**
- New $\alpha = 81 + 1 = 82$ (prior + hits)
- New $\beta = 219 + 0 = 219$ (prior + misses)
- **Posterior Mean:** $82 / (82+219) = 82/301 \approx 0.272$

**Conclusion:** The massive weight of the prior ("He's a rookie") keeps the estimate grounded. One hit doesn't make him a god. This is **regularization**.

**Verification with Code:**
```python
from scipy.stats import beta

# Prior
alpha_prior, beta_prior = 81, 219

# Data: 1 hit, 0 misses
hits, misses = 1, 0

# Posterior
alpha_post = alpha_prior + hits
beta_post = beta_prior + misses

print(f"Prior mean: {alpha_prior/(alpha_prior+beta_prior):.3f}")
print(f"Posterior mean: {alpha_post/(alpha_post+beta_post):.3f}")
print(f"Naive estimate: {hits/(hits+misses):.3f}")
```

---

## Assumptions

- [ ] **Bounded values:** Data must be in [0, 1] range.
  - *Example:* Proportions ✓ vs Dollar amounts ✗
  
- [ ] **Continuous:** Beta is for continuous probabilities.
  - *Example:* Conversion rate ✓ vs Count of conversions ✗

---

## Limitations

> [!warning] Pitfalls
> 1. **Bounded support:** Cannot model values outside [0, 1].
> 2. **Sensitivity to small α, β:** Values < 1 create U-shaped or J-shaped distributions that may not match reality.
> 3. **Not for counts:** Use Binomial/Poisson for count data.

---

## Python Implementation

```python
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 100)

# Different shape combinations
shapes = [(1, 1), (2, 2), (5, 1), (1, 5), (2, 5), (0.5, 0.5)]

plt.figure(figsize=(10, 6))
for (a, b) in shapes:
    plt.plot(x, beta.pdf(x, a, b), label=f'Beta({a},{b})')

plt.xlabel('x')
plt.ylabel('Density')
plt.title('Beta Distribution Shapes')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Bayesian update example
a_prior, b_prior = 10, 10  # Prior
successes, failures = 7, 3  # Data

a_post = a_prior + successes
b_post = b_prior + failures
print(f"Prior mean: {a_prior/(a_prior+b_prior):.3f}")
print(f"Posterior mean: {a_post/(a_post+b_post):.3f}")
```

**Expected Output:**
```
Prior mean: 0.500
Posterior mean: 0.567
```

---

## R Implementation

```r
# Beta Distribution
alpha <- 2
beta_param <- 5

# Generate random samples
samples <- rbeta(1000, alpha, beta_param)

# Density at x=0.5
dens <- dbeta(0.5, alpha, beta_param)
print(paste("Density at 0.5:", round(dens, 4)))

# Mean
print(paste("Mean:", alpha / (alpha + beta_param)))

# Plot
hist(samples, freq=FALSE, main="Beta(2,5)", col="lightblue")
curve(dbeta(x, alpha, beta_param), add=TRUE, col="red", lwd=2)
```

---

## Interpretation Guide

| Scenario | Interpretation |
|----------|----------------|
| **Large α + β** | Strong prior/posterior; narrow distribution |
| **Small α + β** | Weak prior; wide, uncertain distribution |
| **α = β** | Symmetric around 0.5 |
| **α > β** | Skewed toward 1 (higher probability) |
| **α < β** | Skewed toward 0 (lower probability) |

---

## Related Concepts

### Directly Related
- [[30_Knowledge/Stats/01_Foundations/Bernoulli Distribution\|Bernoulli Distribution]] - Beta is its conjugate prior
- [[30_Knowledge/Stats/01_Foundations/Binomial Distribution\|Binomial Distribution]] - Beta + Binomial → Beta posterior
- [[30_Knowledge/Stats/01_Foundations/Bayesian Statistics\|Bayesian Statistics]] - Primary use case
- [[30_Knowledge/Stats/01_Foundations/Conjugate Prior\|Conjugate Prior]] - Mathematical property making Beta useful

### Generalizations
- [[30_Knowledge/Stats/01_Foundations/Dirichlet Distribution\|Dirichlet Distribution]] - Multivariate generalization (>2 categories)

### Other Related Topics
- [[30_Knowledge/Stats/03_Regression_Analysis/BIC (Bayesian Information Criterion)\|BIC (Bayesian Information Criterion)]]

{ .block-language-dataview}

---

## References

1. Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press. [Available online](https://www.routledge.com/Bayesian-Data-Analysis/Gelman-Carlin-Stern-Dunson-Vehtari-Rubin/p/book/9781439840955)

2. Johnson, N. L., Kotz, S., & Balakrishnan, N. (1995). *Continuous Univariate Distributions, Vol. 2* (2nd ed.). Wiley. Chapter 25. [Available online](https://www.wiley.com/en-us/Continuous+Univariate+Distributions%2C+Volume+2%2C+2nd+Edition-p-9780471584940)

3. Gupta, A. K., & Nadarajah, S. (2004). *Handbook of Beta Distribution and Its Applications*. CRC Press. [Available online](https://www.routledge.com/Handbook-of-Beta-Distribution-and-Its-Applications/Gupta-Nadarajah/p/book/9781138473218)
