---
{"dg-publish":true,"permalink":"/stats/01-foundations/likelihood-function/","tags":["Probability-Theory","Inference","MLE","Bayesian"]}
---


## Definition

> [!abstract] Core Statement
> The **Likelihood Function** measures how ==probable the observed data is== for different parameter values. Given data $x$ and parameter $\theta$:
> $$L(\theta | x) = P(x | \theta)$$
> It answers: "How likely would we see this data if θ were true?"

![Likelihood Function for a Binomial parameter p](https://commons.wikimedia.org/wiki/Special:FilePath/MLfunction.svg)

**Key Distinction:** Probability fixes θ and asks about data. Likelihood fixes data and asks about θ.

---

## Purpose

1.  **Maximum Likelihood Estimation (MLE):** Find θ that maximizes $L(\theta|x)$.
2.  **Model Comparison:** Compare how well different models explain data.
3.  **Bayesian Inference:** Posterior ∝ Likelihood × Prior.

---

## Theoretical Background

### For IID Observations
$$L(\theta | x_1, \dots, x_n) = \prod_{i=1}^{n} f(x_i | \theta)$$

### Log-Likelihood (More Practical)
$$\ell(\theta) = \log L(\theta) = \sum_{i=1}^{n} \log f(x_i | \theta)$$

### MLE
$$\hat{\theta}_{MLE} = \arg\max_\theta \ell(\theta)$$

Solve: $\frac{\partial \ell}{\partial \theta} = 0$

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
cat("MLE for μ:", round(result$minimum, 2))
```

---

## Worked Example

> [!example] Binomial MLE
> **Data:** 7 successes in 10 trials. Find MLE for p.
> 
> $$L(p) = \binom{10}{7} p^7 (1-p)^3$$
> $$\ell(p) = 7\log(p) + 3\log(1-p) + C$$
> $$\frac{d\ell}{dp} = \frac{7}{p} - \frac{3}{1-p} = 0$$
> $$\hat{p} = \frac{7}{10} = 0.70$$

---

## Related Concepts

- [[stats/01_Foundations/Maximum Likelihood Estimation (MLE)\|Maximum Likelihood Estimation (MLE)]] - Optimization method
- [[stats/01_Foundations/Bayes' Theorem\|Bayes' Theorem]] - Posterior ∝ Likelihood × Prior
- [[stats/01_Foundations/Log Transformation\|Log Transformation]] - Why we use log-likelihood

---

## References

- **Book:** Casella, G., & Berger, R. L. (2002). *Statistical Inference*. Duxbury. (Ch. 7) [Publisher Link](https://www.routledge.com/Statistical-Inference/Casella-Berger/p/book/9781032593036)
- **Book:** Pawitan, Y. (2001). *In All Likelihood*. Oxford University Press. [Oxford Link](https://global.oup.com/academic/product/in-all-likelihood-9780199671229)
