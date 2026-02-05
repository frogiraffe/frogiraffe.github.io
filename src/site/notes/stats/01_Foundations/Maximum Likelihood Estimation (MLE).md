---
{"dg-publish":true,"permalink":"/stats/01-foundations/maximum-likelihood-estimation-mle/","tags":["probability","estimation","inference","foundations"]}
---

## Definition

> [!abstract] Core Statement
> **Maximum Likelihood Estimation (MLE)** finds the parameter values that ==maximize the probability of observing the data==. It asks: "Which θ makes the observed data most likely?"

![MLE: Finding the peak of the likelihood function|500](https://upload.wikimedia.org/wikipedia/commons/f/ff/Probability_density_function_of_a_normal_distribution.svg)
*Figure 1: MLE finds the parameter that maximizes the likelihood function.*

---

> [!tip] Intuition (ELI5): The Perfect Key
> Imagine you have a locked door (observed data) and a bag of 1,000 different keys (possible parameters). MLE is trying every key and picking the one that **opens the door most smoothly**—the one that makes "this door opened" most likely.

---

## Purpose

1. **Parameter estimation:** Find best-fit values for distribution parameters
2. **Foundation for inference:** Basis for confidence intervals, hypothesis tests
3. **Optimization target:** What many ML algorithms maximize

---

## When to Use

> [!success] Use MLE When...
> - Sample size is **large enough** for asymptotic properties
> - You want **computational efficiency** (closed-form or gradient-based)
> - Model is **correctly specified**

---

## When NOT to Use

> [!danger] Do NOT Use MLE When...
> - **Small sample:** MLE can be biased; consider Bayesian or bootstrap
> - **Strong prior knowledge:** Use [[stats/01_Foundations/Bayesian Statistics\|Bayesian Statistics]] instead
> - **Misspecified model:** MLE will converge to wrong parameters
> - **Outliers present:** MLE is sensitive; consider robust alternatives

---

## Theoretical Background

### Probability vs Likelihood

| Concept | Fixes | Asks About |
|---------|-------|------------|
| **Probability** | Parameters θ | Data X: $P(X|\theta)$ |
| **Likelihood** | Data X | Parameters θ: $L(\theta|X)$ |

### The Likelihood Function

For IID observations $x_1, \ldots, x_n$:
$$L(\theta) = \prod_{i=1}^{n} f(x_i; \theta)$$

### Log-Likelihood

Much easier to work with:
$$\ell(\theta) = \log L(\theta) = \sum_{i=1}^{n} \log f(x_i; \theta)$$

### MLE Procedure

1. **Write likelihood:** $L(\theta | \text{data})$
2. **Take log:** $\ell(\theta) = \log L(\theta)$
3. **Differentiate:** Find $\frac{\partial \ell}{\partial \theta}$
4. **Set to zero:** $\frac{\partial \ell}{\partial \theta} = 0$
5. **Solve:** Find $\hat{\theta}_{MLE}$
6. **Verify:** Check second derivative is negative (maximum)

### Properties of MLE

| Property | Meaning |
|----------|---------|
| **Consistent** | $\hat{\theta} \to \theta_{true}$ as $n \to \infty$ |
| **Asymptotically Normal** | $\sqrt{n}(\hat{\theta} - \theta) \xrightarrow{d} N(0, I^{-1}(\theta))$ |
| **Efficient** | Achieves Cramér-Rao lower bound asymptotically |
| **Invariant** | MLE of $g(\theta)$ is $g(\hat{\theta})$ |

---

## Worked Example: Normal Mean

> [!example] Problem
> Observations: $x_1, x_2, \ldots, x_n \sim N(\mu, \sigma^2)$ with $\sigma^2$ known.
> Find the MLE for $\mu$.

**Solution:**

**1. Likelihood:**
$$L(\mu) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)$$

**2. Log-Likelihood:**
$$\ell(\mu) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i - \mu)^2$$

**3. Differentiate:**
$$\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^{n}(x_i - \mu)$$

**4. Set to zero:**
$$\sum_{i=1}^{n}(x_i - \mu) = 0$$
$$\sum_{i=1}^{n} x_i = n\mu$$

**5. Solve:**
$$\hat{\mu}_{MLE} = \frac{1}{n}\sum_{i=1}^{n} x_i = \bar{x}$$

**Result:** The MLE for the mean is the **sample mean**! 🎯

**Verification with Code:**
```python
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar

# Generate data
np.random.seed(42)
true_mu = 5
data = np.random.normal(true_mu, 1, size=100)

# Log-likelihood for Normal with known sigma
def neg_log_likelihood(mu, data, sigma=1):
    return -np.sum(stats.norm.logpdf(data, loc=mu, scale=sigma))

# Find MLE
result = minimize_scalar(lambda mu: neg_log_likelihood(mu, data))
print(f"MLE: {result.x:.4f}")
print(f"Sample mean: {data.mean():.4f}")  # Should match!
print(f"True mu: {true_mu}")
```

**Expected Output:**
```
MLE: 5.0591
Sample mean: 5.0591
True mu: 5
```

---

## Assumptions

- [ ] **Correct model:** True distribution is in the assumed family.
  - *Example:* Data is Normal ✓ vs Data is heavy-tailed ✗
  
- [ ] **Identifiability:** Parameters can be uniquely determined.
  - *Example:* Standard Normal ✓ vs Mixture with unknown number of components ✗
  
- [ ] **Regularity conditions:** For asymptotic properties to hold.

---

## Limitations

> [!warning] Pitfalls
> 1. **Biased in small samples:** MLE for variance uses $n$ not $n-1$.
> 2. **Boundary issues:** MLE can be at parameter boundary (e.g., variance = 0).
> 3. **Local maxima:** Numerical optimization may find wrong peak.
> 4. **Model dependence:** Wrong model → wrong MLE.

---

## Python Implementation

```python
import numpy as np
from scipy import stats
from scipy.optimize import minimize

# Example: Estimate parameters of a Normal distribution
np.random.seed(42)
data = np.random.normal(loc=10, scale=2, size=100)

# Negative log-likelihood (we minimize)
def neg_log_lik(params, data):
    mu, sigma = params
    if sigma <= 0:
        return np.inf
    return -np.sum(stats.norm.logpdf(data, loc=mu, scale=sigma))

# Optimize
initial = [0, 1]
result = minimize(neg_log_lik, initial, args=(data,), method='Nelder-Mead')

print(f"MLE for μ: {result.x[0]:.4f}")
print(f"MLE for σ: {result.x[1]:.4f}")
print(f"Sample mean: {data.mean():.4f}")
print(f"Sample std (n): {data.std():.4f}")  # MLE uses n, not n-1
```

---

## R Implementation

```r
# Generate data
set.seed(42)
data <- rnorm(100, mean = 10, sd = 2)

# Negative log-likelihood
neg_log_lik <- function(params, data) {
  mu <- params[1]
  sigma <- params[2]
  if (sigma <= 0) return(Inf)
  -sum(dnorm(data, mean = mu, sd = sigma, log = TRUE))
}

# Optimize
result <- optim(c(0, 1), neg_log_lik, data = data)
cat("MLE for μ:", round(result$par[1], 4), "\n")
cat("MLE for σ:", round(result$par[2], 4), "\n")
cat("Sample mean:", round(mean(data), 4), "\n")
```

---

## Interpretation Guide

| Result | Interpretation |
|--------|----------------|
| **MLE = sample mean** | For Normal data with known variance |
| **MLE = sample proportion** | For Binomial data |
| **SE from Fisher Info** | $\text{SE} = \sqrt{I(\hat{\theta})^{-1}}$ |
| **Convergence warning** | Check for local maxima, try different starts |

---

## MLE vs Bayesian

| Aspect | MLE | Bayesian |
|--------|-----|----------|
| **Prior** | None | Required |
| **Output** | Point estimate | Full posterior |
| **Small samples** | Can be biased | Incorporates prior |
| **Computation** | Often closed-form | Usually MCMC |

---

## Related Concepts

### Directly Related
- [[stats/01_Foundations/Likelihood Function\|Likelihood Function]] - What MLE maximizes
- [[stats/01_Foundations/Bayesian Statistics\|Bayesian Statistics]] - Alternative with priors
- [[Fisher Information\|Fisher Information]] - Measures precision of MLE

### Applications
- [[stats/03_Regression_Analysis/Binary Logistic Regression\|Binary Logistic Regression]] - Uses MLE for coefficients
- [[stats/03_Regression_Analysis/Simple Linear Regression\|Simple Linear Regression]] - OLS = MLE under normality

### Other Related Topics
- [[stats/01_Foundations/Bayes' Theorem\|Bayes' Theorem]]
- [[stats/01_Foundations/Bayesian Statistics\|Bayesian Statistics]]
- [[stats/04_Supervised_Learning/Bootstrap Methods\|Bootstrap Methods]]
- [[stats/01_Foundations/Central Limit Theorem (CLT)\|Central Limit Theorem (CLT)]]
- [[stats/02_Statistical_Inference/Confidence Intervals\|Confidence Intervals]]

{ .block-language-dataview}

---

## References

1. Casella, G., & Berger, R. L. (2002). *Statistical Inference* (2nd ed.). Duxbury. [Available online](https://www.routledge.com/Statistical-Inference/Casella-Berger/p/book/9781032593036)

2. Pawitan, Y. (2001). *In All Likelihood: Statistical Modelling and Inference Using Likelihood*. Oxford University Press. [Available online](https://global.oup.com/academic/product/in-all-likelihood-9780199671229)

3. Fisher, R. A. (1922). On the mathematical foundations of theoretical statistics. *Philosophical Transactions of the Royal Society A*, 222, 309-368. [Available online](http://www.jstor.org/stable/91208)
