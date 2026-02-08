---
{"dg-publish":true,"permalink":"/30-knowledge/stats/04-supervised-learning/bayesian-statistics-via-probabilistic-programming/","tags":["machine-learning","supervised"]}
---

## Overview

> [!abstract] Definition
> This note illustrates the practical application of [[30_Knowledge/Stats/01_Foundations/Bayesian Statistics\|Bayesian Statistics]] using modern **Probabilistic Programming** frameworks. It bridges abstract Bayesian theory with computational implementation for real-world inference.

---

## 1. Bayesian A/B Testing

### The Problem
Traditional frequentist approaches rely on p-values and binary decision rules. The Bayesian approach directly estimates the probability that one variant is superior to another ($P(B > A)$) and quantifies the expected uplift.

### Implementation Example

```python
import pymc as pm
import numpy as np

# Sample Data
visitors_A, conversions_A = 1000, 50
visitors_B, conversions_B = 1000, 65

with pm.Model() as ab_test:
    # Priors: Uniformative Beta priors (equivalent to Uniform[0,1])
    p_A = pm.Beta('p_A', alpha=1, beta=1)
    p_B = pm.Beta('p_B', alpha=1, beta=1)
    
    # Likelihoods: Binomial distribution
    obs_A = pm.Binomial('obs_A', n=visitors_A, p=p_A, observed=conversions_A)
    obs_B = pm.Binomial('obs_B', n=visitors_B, p=p_B, observed=conversions_B)
    
    # Deterministic: Difference
    delta = pm.Deterministic('delta', p_B - p_A)
    
    # Inference: MCMC Sampling
    trace = pm.sample(draws=2000, tune=1000)

# Analysis
prob_b_better = (trace.posterior['delta'] > 0).mean()
print(f"Probability B is better than A: {prob_b_better:.1%}")
```

---

## 2. Bayesian Linear Regression

Unlike OLS, which provides point estimates, Bayesian regression yields full posterior distributions for coefficients, allowing for robust uncertainty quantification.

```python
with pm.Model() as linear_model:
    # Priors for unknown model parameters
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=1)
    
    # Expected value of outcome
    mu = alpha + beta * X
    
    # Likelihood (Sampling distribution) of observations
    y_obs = pm.Normal('y', mu=mu, sigma=sigma, observed=y)
    
    # Inference
    trace = pm.sample()
```

---

## 3. Hierarchical Examples

**Hierarchical (Multilevel) Models** are a hallmark of Bayesian analysis, allowing information sharing (partial pooling) across groups.

```python
with pm.Model() as hierarchical_model:
    # Hyperpriors (Population level)
    mu_global = pm.Normal('mu_global', mu=0, sigma=10)
    sigma_global = pm.HalfNormal('sigma_global', sigma=5)
    
    # Group-level parameters (Subject to population prior)
    mu_group = pm.Normal('mu_group', mu=mu_global, sigma=sigma_global, shape=n_groups)
    
    # Likelihood using group-specific parameters
    sigma = pm.HalfNormal('sigma', sigma=1)
    pm.Normal('y', mu=mu_group[group_idx], sigma=sigma, observed=y)
    
    trace = pm.sample()
```

---

## 4. Model Workflow and Checking

1.  **Prior Predictive Checks:** Simulate data from priors before observing actual data to ensure assumptions are reasonable.
    ```python
    with model:
        idata_prior = pm.sample_prior_predictive()
    ```
    
2.  **Convergence Diagnostics:** Check $\hat{R}$ (R-hat) and Effective Sample Size (ESS).
    
3.  **Posterior Predictive Checks:** Compare simulated data from the fitted model against observed data to assess fit.
    ```python
    with model:
        pm.sample_posterior_predictive(trace, extend_inferencedata=True)
    ```

---

## 5. Related Concepts

- [[30_Knowledge/Stats/01_Foundations/Bayesian Statistics\|Bayesian Statistics]] - Theoretical foundations.
- [[30_Knowledge/Stats/04_Supervised_Learning/Probabilistic Programming\|Probabilistic Programming]] - Framework details.
- [[30_Knowledge/Stats/01_Foundations/Likelihood Function\|Likelihood Function]] - Core component.

---

## Definition

> [!abstract] Core Statement
> **Bayesian Statistics via Probabilistic Programming** ... Refer to standard documentation

---

> [!tip] Intuition (ELI5)
> Refer to standard documentation

---

## When to Use

> [!success] Use Bayesian Statistics via Probabilistic Programming When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Dataset is too small for training
> - Interpretability is more important than accuracy

---

## Python Implementation

```python
import numpy as np
import pandas as pd

# Example implementation of Bayesian Statistics via Probabilistic Programming
# See documentation for details

data = np.random.randn(100)
print(f"Mean: {np.mean(data):.3f}")
print(f"Std: {np.std(data):.3f}")
```

---

## R Implementation

```r
# Bayesian Statistics via Probabilistic Programming in R
set.seed(42)

# Example implementation
data <- rnorm(100)
summary(data)
```

---

## Related Concepts

- [[30_Knowledge/Stats/04_Supervised_Learning/Random Forest\|Random Forest]]
- [[30_Knowledge/Stats/04_Supervised_Learning/Gradient Boosting\|Gradient Boosting]]
- [[30_Knowledge/Stats/04_Supervised_Learning/Cross-Validation\|Cross-Validation]]

---

## References

- **Book:** Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press. [Official Site](http://www.stat.columbia.edu/~gelman/book/)
- **Book:** Davidson-Pilon, C. (2015). *Bayesian Methods for Hackers*. Addison-Wesley. [GitHub / Online Edition](https://camdavidsonpilon.github.io/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/)
- **Book:** McElreath, R. (2020). *Statistical Rethinking* (2nd ed.). CRC Press. [Book Website](https://xcelab.net/rm/statistical-rethinking/)