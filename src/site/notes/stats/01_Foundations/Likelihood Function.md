---
{"dg-publish":true,"permalink":"/stats/01-foundations/likelihood-function/","tags":["probability","inference","estimation","mle","bayesian"]}
---

## Definition

> [!abstract] Core Statement
> The **Likelihood Function** measures how ==probable the observed data is== for different parameter values. Given data $x$ and parameter $\theta$:
> $$L(\theta | x) = P(x | \theta)$$
> It answers: "How likely would we see this data if θ were true?"

![Likelihood Function showing peak at MLE|500](https://upload.wikimedia.org/wikipedia/commons/f/ff/Probability_density_function_of_a_normal_distribution.svg)
*Figure 1: Likelihood function peaks at the most likely parameter value (MLE).*

---

> [!tip] Intuition (ELI5): The Crime Scene Detective
> You found a footprint at the crime scene (data). The likelihood function asks: "If the suspect has size 10 shoes, how likely is this footprint? Size 11? Size 9?" You're not changing the footprint—you're asking which shoe size makes the footprint most believable.

---

## Purpose

1. **[[stats/01_Foundations/Maximum Likelihood Estimation (MLE)\|Maximum Likelihood Estimation (MLE)]]:** Find θ that maximizes $L(\theta|x)$
2. **Model comparison:** Compare how well different models explain data
3. **[[stats/01_Foundations/Bayesian Statistics\|Bayesian inference]]:** Posterior ∝ Likelihood × Prior

---

## Key Distinction

| Concept | Fixes | Asks About |
|---------|-------|------------|
| **Probability** | Parameters θ | Data X: $P(X|\theta)$ |
| **Likelihood** | Data X | Parameters θ: $L(\theta|X)$ |

> [!important] Critical Insight
> Probability and likelihood use the same formula but with different interpretations. In probability, θ is known and you ask about X. In likelihood, X is known and you ask about θ.

---

## When to Use

> [!success] Use Likelihood When...
> - Estimating parameters from observed data
> - Comparing models (likelihood ratio tests)
> - Constructing Bayesian posteriors

---

## When NOT to Use

> [!danger] Do NOT Interpret Likelihood as Probability!
> - $L(\theta|x)$ is **NOT** $P(\theta|x)$
> - Likelihoods don't integrate to 1 across θ
> - To get $P(\theta|x)$, you need [[stats/01_Foundations/Bayes' Theorem\|Bayes' Theorem]]

---

## Theoretical Background

### For IID Observations

$$L(\theta | x_1, \ldots, x_n) = \prod_{i=1}^{n} f(x_i | \theta)$$

### Log-Likelihood (More Practical)

$$\ell(\theta) = \log L(\theta) = \sum_{i=1}^{n} \log f(x_i | \theta)$$

**Why log?**
- Turns products into sums (numerically stable)
- Same maximum location (log is monotonic)
- Easier derivatives

### MLE

$$\hat{\theta}_{MLE} = \arg\max_\theta \ell(\theta)$$

Solve: $\frac{\partial \ell}{\partial \theta} = 0$

---

## Worked Example: Binomial MLE

> [!example] Problem
> **Data:** 7 successes in 10 trials. Find MLE for p.

**Solution:**

**1. Likelihood:**
$$L(p) = \binom{10}{7} p^7 (1-p)^3$$

**2. Log-likelihood:**
$$\ell(p) = 7\log(p) + 3\log(1-p) + C$$

**3. Differentiate:**
$$\frac{d\ell}{dp} = \frac{7}{p} - \frac{3}{1-p}$$

**4. Set to zero:**
$$\frac{7}{p} = \frac{3}{1-p}$$
$$7(1-p) = 3p$$
$$7 = 10p$$

**5. Solve:**
$$\hat{p}_{MLE} = \frac{7}{10} = 0.70$$

**Verification:**
```python
import numpy as np
from scipy.optimize import minimize_scalar

# Log-likelihood for binomial
def log_lik(p, successes=7, n=10):
    if p <= 0 or p >= 1:
        return -np.inf
    return successes * np.log(p) + (n - successes) * np.log(1 - p)

# Find MLE
result = minimize_scalar(lambda p: -log_lik(p), bounds=(0.01, 0.99), method='bounded')
print(f"MLE: {result.x:.4f}")  # 0.7000
```

---

## Python Implementation

```python
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar

# Data: Heights (assume normal distribution)
data = np.array([170, 165, 180, 175, 168, 172, 177, 169, 174, 171])

# Log-likelihood for Normal with known σ=5
def log_likelihood(mu, data, sigma=5):
    return np.sum(stats.norm.logpdf(data, loc=mu, scale=sigma))

# Find MLE for mu
result = minimize_scalar(lambda mu: -log_likelihood(mu, data), 
                         bounds=(150, 200), method='bounded')
mle_mu = result.x
print(f"MLE for μ: {mle_mu:.2f}")
print(f"Sample mean: {data.mean():.2f}")  # Should match!
```

**Expected Output:**
```
MLE for μ: 172.10
Sample mean: 172.10
```

---

## R Implementation

```r
data <- c(170, 165, 180, 175, 168, 172, 177, 169, 174, 171)

# Log-likelihood for Normal
log_likelihood <- function(mu, data, sigma = 5) {
  sum(dnorm(data, mean = mu, sd = sigma, log = TRUE))
}

# Find MLE
result <- optimize(function(mu) -log_likelihood(mu, data), 
                   interval = c(150, 200))
cat("MLE for μ:", round(result$minimum, 2), "\n")
cat("Sample mean:", round(mean(data), 2), "\n")
```

---

## Likelihood vs Posterior

| Aspect | Likelihood | Posterior |
|--------|------------|-----------|
| **Formula** | $L(\theta|x) = P(x|\theta)$ | $P(\theta|x) \propto L(\theta|x) \cdot P(\theta)$ |
| **Prior** | None | Required |
| **Interpretation** | Data compatibility | Updated belief |
| **Integrates to 1** | No | Yes |

---

## Related Concepts

### Directly Related
- [[stats/01_Foundations/Maximum Likelihood Estimation (MLE)\|Maximum Likelihood Estimation (MLE)]] - Maximizes likelihood
- [[stats/01_Foundations/Bayes' Theorem\|Bayes' Theorem]] - Posterior ∝ Likelihood × Prior
- [[stats/01_Foundations/Log Transformation\|Log Transformation]] - Why we use log-likelihood

### Applications
- [[stats/03_Regression_Analysis/Binary Logistic Regression\|Binary Logistic Regression]] - Uses likelihood for estimation
- [[Likelihood Ratio Test\|Likelihood Ratio Test]] - Compares nested models

### Other Related Topics
- [[stats/01_Foundations/Bayes' Theorem\|Bayes' Theorem]]
- [[stats/01_Foundations/Bayesian Statistics\|Bayesian Statistics]]
- [[stats/04_Supervised_Learning/Bootstrap Methods\|Bootstrap Methods]]
- [[stats/01_Foundations/Central Limit Theorem (CLT)\|Central Limit Theorem (CLT)]]
- [[stats/02_Statistical_Inference/Confidence Intervals\|Confidence Intervals]]

{ .block-language-dataview}

---

## References

1. Casella, G., & Berger, R. L. (2002). *Statistical Inference* (2nd ed.). Duxbury. Chapter 7. [Available online](https://www.routledge.com/Statistical-Inference/Casella-Berger/p/book/9781032593036)

2. Pawitan, Y. (2001). *In All Likelihood: Statistical Modelling and Inference Using Likelihood*. Oxford University Press. [Available online](https://global.oup.com/academic/product/in-all-likelihood-9780199671229)
