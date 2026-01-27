---
{"dg-publish":true,"permalink":"/stats/01-foundations/gamma-distribution/","tags":["Probability-Theory","Distributions","Continuous","Reliability"]}
---

## Definition

> [!abstract] Core Statement
> The **Gamma Distribution** models the time until **$k$ events** occur in a Poisson process. It is a generalization of the [[stats/01_Foundations/Exponential Distribution\|Exponential Distribution]] (which models time until *1* event). It is defined by shape parameter $k$ (or $\alpha$) and scale parameter $\theta$ (or rate $\beta$).

![Gamma Distribution PDF](https://upload.wikimedia.org/wikipedia/commons/e/e6/Gamma_distribution_pdf.svg)

---

## Purpose

1.  **Waiting Times:** How long until 5 customers arrive? How long until the 3rd component fails?
2.  **Reliability Engineering:** Modeling fatigue life where damage accumulates.
3.  **Bayesian Statistics:** Conjugate prior for the rate parameter of a Poisson or Exponential likelihood.
4.  **Financial Modeling:** Asset sizes or insurance claims (skewed, positive data).

---

## Relations (The Family Tree)

-   **Exponential:** Gamma with shape $k=1$. (Wait for 1st event).
-   **Chi-Square:** Gamma with specific parameters ($\alpha = \nu/2, \beta=1/2$).
-   **Erlang:** Gamma where $k$ is an integer.

---

## Worked Example: Multiple Failures

> [!example] Problem
> A server crashes on average once every 2 days ($\lambda = 0.5$ per day).
> **Question:** What is the probability that the **3rd crash** happens within 10 days?
> 
> **Setup:**
> -   Events to wait for ($k$ or $\alpha$): 3.
> -   Rate ($\lambda$ or $\beta$): 0.5.
> -   This is **Gamma(3, 0.5)**.
> 
> **Calculation:**
> Using Python or Tables ($P(X \le 10)$):
> $$ \text{Result} \approx 0.875 $$
> 
> *Intuition:* Since average time per crash is 2 days, 3 crashes take ~6 days. So 10 days is plenty of time. 87.5% chance.

---

## Parameters

Warning: Different fields use different parametrizations!
-   **Shape-Scale ($k, \theta$):** Mean = $k\theta$. (Engineering).
-   **Shape-Rate ($\alpha, \beta$):** Mean = $\alpha / \beta$. (Bayesian/Math).
    -   $\beta = 1/\theta$.

---

## Python Implementation

```python
from scipy.stats import gamma
import matplotlib.pyplot as plt

# Parameters (using 'a' as shape alpha)
alpha = 3  # Wait for 3 events
loc = 0
scale = 2  # Mean time per event (theta = 1/beta) (If rate=0.5, scale=2)

rv = gamma(alpha, loc=loc, scale=scale)

# Probability wait < 10
prob = rv.cdf(10)
print(f"Prob < 10: {prob:.4f}")

# Plot
import numpy as np
x = np.linspace(0, 20, 100)
plt.plot(x, rv.pdf(x))
plt.title("Gamma Distribution (k=3, theta=2)")
plt.show()
```

---

## R Implementation

```r
# Gamma Distribution
k <- 2   # shape
theta <- 2 # scale (rate = 1/theta)

# Sample
samples <- rgamma(1000, shape=k, scale=theta)

# Plot Density
hist(samples, freq=FALSE, main="Gamma(2,2)", col="lightgreen")
curve(dgamma(x, shape=k, scale=theta), add=TRUE, col="darkgreen", lwd=2)

# CDF
print(paste("Prob < 5:", pgamma(5, shape=k, scale=theta)))
```

---

## Related Concepts

- [[stats/01_Foundations/Exponential Distribution\|Exponential Distribution]] - Special case ($k=1$).
- [[stats/01_Foundations/Poisson Distribution\|Poisson Distribution]] - Counts events in fixed time (Dual concept).
- [[stats/01_Foundations/Chi-Square Distribution\|Chi-Square Distribution]] - Another special case.
- [[stats/01_Foundations/Beta Distribution\|Beta Distribution]] - Associated conjugate prior.

---

## References

- **Book:** Hogg, R. V., & Tanis, E. A. (2010). *Probability and Statistical Inference* (8th ed.). Pearson. [Pearson Link](https://www.pearson.com/en-us/subject-catalog/p/probability-and-statistical-inference/P200000003540/9780137981502)
- **Book:** Casella, G., & Berger, R. L. (2002). *Statistical Inference* (2nd ed.). Duxbury. [Publisher Link](https://www.routledge.com/Statistical-Inference/Casella-Berger/p/book/9781032593036)
- **Article:** Huschke, R. E. (1959). *Glossary of Meteorology*. American Meteorological Society. [Archive.org](https://archive.org/details/glossaryofmeteoro00husc)
