---
{"dg-publish":true,"permalink":"/stats/01-foundations/monte-carlo-simulation/","tags":["Simulation","Computational-Statistics","Numerical-Methods"]}
---


## Definition

> [!abstract] Core Statement
> **Monte Carlo Simulation** uses ==repeated random sampling== to estimate numerical quantities that are difficult to compute analytically. By simulating many random scenarios, we approximate probabilities, expectations, and distributions.

![Monte Carlo Method for Estimating Pi](https://upload.wikimedia.org/wikipedia/commons/e/ea/Monte-Carlo_method_pi.svg)

**Intuition:** Can't solve the math? Simulate it 10,000 times and count.

---

## Purpose

1.  **Estimate Probabilities:** P(complex event) via simulation.
2.  **Approximate Integrals:** Especially high-dimensional.
3.  **Uncertainty Quantification:** Propagate errors through complex models.
4.  **Bayesian Inference:** MCMC for posterior distributions.

---

## Theoretical Background

### Law of Large Numbers
$$\frac{1}{n}\sum_{i=1}^{n} g(X_i) \xrightarrow{n \to \infty} E[g(X)]$$

### Estimating Expectations
$$E[g(X)] \approx \frac{1}{n}\sum_{i=1}^{n} g(X_i)$$

### Standard Error
$$SE = \frac{\sigma}{\sqrt{n}}$$

More samples → smaller error.

---

## Python Implementation

```python
import numpy as np

# Example 1: Estimate π using random points
np.random.seed(42)
n = 100000

x = np.random.uniform(-1, 1, n)
y = np.random.uniform(-1, 1, n)
inside_circle = (x**2 + y**2) <= 1
pi_estimate = 4 * inside_circle.mean()
print(f"π estimate: {pi_estimate:.4f} (actual: {np.pi:.4f})")

# Example 2: Option pricing (simplified Black-Scholes)
S0 = 100  # Initial stock price
K = 105   # Strike price
r = 0.05  # Risk-free rate
T = 1     # Time to expiry
sigma = 0.2  # Volatility

n_simulations = 100000
Z = np.random.standard_normal(n_simulations)
ST = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
payoffs = np.maximum(ST - K, 0)
option_price = np.exp(-r*T) * payoffs.mean()
print(f"Call option price: ${option_price:.2f}")
```

---

## R Implementation

```r
set.seed(42)

# Estimate π
n <- 100000
x <- runif(n, -1, 1)
y <- runif(n, -1, 1)
inside <- (x^2 + y^2) <= 1
pi_estimate <- 4 * mean(inside)
cat("π estimate:", round(pi_estimate, 4))

# Bootstrap confidence interval
data <- c(23, 25, 28, 24, 26, 27, 22, 29)
bootstrap_means <- replicate(10000, mean(sample(data, replace = TRUE)))
cat("\n95% CI:", round(quantile(bootstrap_means, c(0.025, 0.975)), 2))
```

---

## Worked Example

> [!example] Birthday Problem
> P(at least 2 people share birthday in group of 23)?
> 
> **Simulation:**
> ```python
> n = 100000
> matches = 0
> for _ in range(n):
>     birthdays = np.random.randint(1, 366, 23)
>     if len(birthdays) != len(set(birthdays)):
>         matches += 1
> print(f"P(match) ≈ {matches/n:.4f}")  # ≈ 0.507
> ```

---

## Related Concepts

- [[stats/04_Supervised_Learning/Bootstrap Methods\|Bootstrap Methods]] - Resampling for inference
- [[stats/01_Foundations/Bayesian Statistics\|Bayesian Statistics]] - MCMC sampling
- [[stats/01_Foundations/Law of Large Numbers\|Law of Large Numbers]] - Theoretical foundation

---

## References

- **Book:** Kroese, D. P., et al. (2011). *Handbook of Monte Carlo Methods*. Wiley. [Wiley Link](https://www.wiley.com/en-us/Handbook+of+Monte+Carlo+Methods-p-9780470177938)
- **Book:** Robert, C. P., & Casella, G. (2004). *Monte Carlo Statistical Methods*. Springer. [Springer Link](https://link.springer.com/book/10.1007/978-1-4757-4145-2)
