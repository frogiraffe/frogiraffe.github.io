---
{"dg-publish":true,"permalink":"/stats/01-foundations/geometric-distribution/","tags":["Distributions","Probability","Discrete"]}
---


## Definition

> [!abstract] Core Statement
> The **Geometric Distribution** models the number of ==Bernoulli trials needed to get the first success==. If p is the success probability:
> $$P(X = k) = (1-p)^{k-1} p$$

![Geometric Distribution PMF for different success probabilities](https://commons.wikimedia.org/wiki/Special:FilePath/Geometric_pmf.svg)

**Alternative:** Some texts define X as number of *failures* before first success.

---

## Properties

| Property | Value |
|----------|-------|
| **Mean** | $E[X] = \frac{1}{p}$ |
| **Variance** | $Var(X) = \frac{1-p}{p^2}$ |
| **Support** | $X \in \{1, 2, 3, \dots\}$ |
| **Memoryless** | $P(X > m+n \| X > m) = P(X > n)$ |

---

## Python Implementation

```python
from scipy import stats
import numpy as np

p = 0.3  # Success probability
geom = stats.geom(p)

# PMF: P(X=5) = number of trials until first success is 5
print(f"P(X=5): {geom.pmf(5):.4f}")

# Mean and variance
print(f"Mean: {geom.mean():.2f}, Variance: {geom.var():.2f}")

# Simulation
samples = geom.rvs(size=10000)
print(f"Simulated mean: {samples.mean():.2f}")
```

---

## R Implementation

```r
p <- 0.3

# PMF
dgeom(4, prob = p)  # Note: R counts failures (k-1)

# Mean
1 / p

# Simulation
samples <- rgeom(10000, p) + 1  # Add 1 to convert to trials
mean(samples)
```

---

## Worked Example

> [!example] Free Throw Shooter
> P(make) = 0.7. Expected attempts until first miss?
> 
> X ~ Geometric(p = 0.3 for miss)
> E[X] = 1/0.3 = 3.33 attempts

---

## Related Concepts

- [[stats/01_Foundations/Bernoulli Distribution\|Bernoulli Distribution]] - Single trial
- [[stats/01_Foundations/Binomial Distribution\|Binomial Distribution]] - Fixed n trials
- [[stats/03_Regression_Analysis/Negative Binomial Regression\|Negative Binomial Regression]] - Count regression

---

## References

- **Book:** Ross, S. M. (2014). *A First Course in Probability*. Pearson. [Pearson Link](https://www.pearson.com/us/higher-education/program/Ross-A-First-Course-in-Probability-9th-Edition/PGM220165.html)
