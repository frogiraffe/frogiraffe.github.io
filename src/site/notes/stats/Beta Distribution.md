---
{"dg-publish":true,"permalink":"/stats/beta-distribution/","tags":["Statistics","Probability-Theory","Distributions","Continuous","Bayesian"]}
---


# Beta Distribution

## Definition

> [!abstract] Core Statement
> The **Beta Distribution** is a continuous distribution defined on the interval **[0, 1]**. It is parametrized by two shape parameters, $\alpha$ (Alpha) and $\beta$ (Beta). It is widely used to model **probabilities** or proportions, and serves as the **conjugate prior** for the Bernoulli, Binomial, and Geometric distributions.

---

## Purpose

1.  **Bayesian Inference:** Representing belief about a probability (e.g., "I think the conversion rate is between 2% and 5%").
2.  **Modeling Rates:** Click-through rates, batting averages, defect rates.
3.  **Project Management:** PERT distribution (Optimistic vs Pessimistic estimates).

---

## Intuition: "Successes and Failures"

You can think of $\alpha$ and $\beta$ as **pseudo-counts** of prior history:
-   $\alpha - 1$: Number of Successes.
-   $\beta - 1$: Number of Failures.

| Parameters | Shape | Interpretation |
|------------|-------|----------------|
| **Beta(1, 1)** | Flat (Uniform) | "I have no idea." (All probs equally likely). |
| **Beta(2, 2)** | Mound at 0.5 | "Weakly believe it's fair." |
| **Beta(100, 100)** | Sharp Spike at 0.5 | "Strongly believe it's fair." |
| **Beta(10, 1)** | Skewed Right (near 1) | "Almost certain to succeed." |
| **Beta(0.5, 0.5)** | U-Shape (bathtub) | "Either 0 or 1, but not middle." |

---

## Worked Example: Batting Average

> [!example] Problem
> A new baseball player appears.
> **Estimate his batting average.**
> 
> 1.  **Prior:** League average is 0.260. We use a prior of **Beta(81, 219)** (Mean = $81/300 = 0.27$, effectively 300 "prior at-bats").
> 2.  **Data:** In his first game, he hits **1 out of 1**. (100% average!).
> 3.  **Naive Mean:** $1.000$ (Way too high).
> 
> **Bayesian Update:**
> -   New $\alpha = 81 + 1 = 82$.
> -   New $\beta = 219 + 0 = 219$.
> -   **Posterior Mean:** $82 / (82+219) = 82/301 \approx 0.272$.
> 
> **Conclusion:** The massive weight of the prior ("He's a rookie") keeps the estimate grounded. One hit doesn't make him a god. This is **regularization**.

---

## Key Properties

-   **Domain:** $x \in [0, 1]$.
-   **Mean:** $\mu = \frac{\alpha}{\alpha + \beta}$.
-   **Mode:** $\frac{\alpha - 1}{\alpha + \beta - 2}$ (for $\alpha, \beta > 1$).

---

## Python Implementation

```python
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 100)

# 1. Uninformed Prior
y1 = beta.pdf(x, 1, 1)

# 2. Strong Belief in Fairness
y2 = beta.pdf(x, 50, 50)

# 3. Skewed Belief (Low prob)
y3 = beta.pdf(x, 2, 8)

plt.plot(x, y1, label='Beta(1,1) [Uniform]')
plt.plot(x, y2, label='Beta(50,50) [Peaked]')
plt.plot(x, y3, label='Beta(2,8) [Low Rate]')
plt.legend()
plt.title("Beta Distribution Shapes")
plt.show()
```

---

## Related Concepts

- [[stats/Bernoulli Distribution\|Bernoulli Distribution]] - Beta is derived from it.
- [[stats/Bayesian Statistics\|Bayesian Statistics]] - Heavy user of Beta.
- [[Conjugate Prior\|Conjugate Prior]] - Mathematical property making Beta useful.
- [[Dirichlet Distribution\|Dirichlet Distribution]] - Multivariate generalization (Beta for >2 categories).
