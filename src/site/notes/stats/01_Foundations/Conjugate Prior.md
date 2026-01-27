---
{"dg-publish":true,"permalink":"/stats/01-foundations/conjugate-prior/","tags":["Bayesian","Probability","Prior"]}
---


## Definition

> [!abstract] Core Statement
> A **Conjugate Prior** is a prior distribution that, when combined with a particular likelihood, produces a ==posterior in the same distributional family==. This enables closed-form Bayesian updating.

---

## Common Conjugate Pairs

| Likelihood | Conjugate Prior | Posterior |
|------------|-----------------|-----------|
| Binomial(n, θ) | Beta(α, β) | Beta(α+x, β+n-x) |
| Poisson(λ) | Gamma(α, β) | Gamma(α+Σx, β+n) |
| Normal(μ, σ²) | Normal(μ₀, τ²) | Normal (weighted mean) |
| Exponential(λ) | Gamma(α, β) | Gamma(α+n, β+Σx) |

---

## Example: Beta-Binomial

**Prior:** $\theta \sim \text{Beta}(2, 2)$

**Data:** 7 successes in 10 trials

**Posterior:**
$$\theta | x \sim \text{Beta}(2+7, 2+3) = \text{Beta}(9, 5)$$

**Posterior Mean:** $\frac{9}{9+5} = 0.64$

---

## Python Implementation

```python
from scipy import stats
import numpy as np

# Prior: Beta(2, 2)
alpha_prior, beta_prior = 2, 2

# Data: 7 successes, 3 failures
successes, failures = 7, 3

# Posterior: Beta(alpha + x, beta + n - x)
alpha_post = alpha_prior + successes
beta_post = beta_prior + failures

posterior = stats.beta(alpha_post, beta_post)
print(f"Posterior mean: {posterior.mean():.3f}")
print(f"95% CI: [{posterior.ppf(0.025):.3f}, {posterior.ppf(0.975):.3f}]")
```

---

## R Implementation

```r
# Prior: Beta(2, 2)
alpha_prior <- 2; beta_prior <- 2
# Data
successes <- 7; failures <- 3

# Posterior
alpha_post <- alpha_prior + successes
beta_post <- beta_prior + failures

# Posterior summary
qbeta(c(0.025, 0.975), alpha_post, beta_post)
```

---

## Why Use Conjugate Priors?

1. **Analytical solutions:** No MCMC needed
2. **Interpretable:** Prior as "pseudo-data"
3. **Computational efficiency**

---

## Related Concepts

- [[stats/01_Foundations/Bayes' Theorem\|Bayes' Theorem]] - Update mechanism
- [[stats/01_Foundations/Bayesian Statistics\|Bayesian Statistics]] - Framework
- [[stats/01_Foundations/Beta Distribution\|Beta Distribution]] - Common conjugate prior

---

## References

- **Historical:** Raiffa, H., & Schlaifer, R. (1961). *Applied Statistical Decision Theory*. Harvard University. [Google Books](https://books.google.com/books/about/Applied_Statistical_Decision_Theory.html?id=oB5mJ0uD6kQC)
- **Book:** Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press. [Publisher Link](https://www.routledge.com/Bayesian-Data-Analysis/Gelman-Carlin-Stern-Dunson-Vehtari-Rubin/p/book/9781439840955) (Ch. 2)
