---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/convergence/","tags":["probability","foundations"]}
---

## Definition

> [!abstract] Core Statement
> **Convergence** describes how a sequence of random variables or iterative estimates ==approaches a limiting value==. In statistics, it underpins asymptotic theory; in optimization, it indicates algorithm termination.

---

> [!tip] Intuition (ELI5): The Targeting Dart
> Imagine throwing darts at a bullseye. At first they're scattered, but as you practice, they cluster closer and closer to the center. Convergence is the idea that with enough tries, you'll reliably hit near the target.

---

## Purpose

1. **Asymptotic theory:** Justify using Normal approximations (CLT, LLN)
2. **Algorithm termination:** Know when to stop optimization
3. **Consistency proofs:** Show estimators approach true values

---

## Types of Convergence in Probability

| Type | Notation | Meaning |
|------|----------|---------|
| **Almost Sure (a.s.)** | $X_n \xrightarrow{a.s.} X$ | $P(\lim_{n\to\infty} X_n = X) = 1$ |
| **In Probability** | $X_n \xrightarrow{p} X$ | $P(\|X_n - X\| > \epsilon) \to 0$ |
| **In Distribution** | $X_n \xrightarrow{d} X$ | CDFs converge: $F_n(x) \to F(x)$ |
| **In Mean Square** | $X_n \xrightarrow{L^2} X$ | $E[(X_n - X)^2] \to 0$ |

### Hierarchy (Strongest to Weakest)

```
Almost Sure → In Probability → In Distribution
     ↓              ↓
Mean Square ────────┘
```

---

## When to Use

> [!success] Use Convergence Analysis When...
> - Proving **consistency** of estimators
> - Justifying **asymptotic normality** (needed for CLT)
> - Setting **stopping criteria** for iterative algorithms

---

## When NOT to Use

> [!danger] Pitfalls
> - **Finite samples:** Convergence is asymptotic; $n=50$ may not be "large enough"
> - **Slow convergence:** Heavy-tailed distributions may need millions of samples
> - **Non-IID data:** Standard convergence results may not apply

---

## Convergence in Optimization

### Stopping Criteria

| Criterion | Condition | Use Case |
|-----------|-----------|----------|
| **Gradient norm** | $\|\nabla L\| < \epsilon$ | Smooth objectives |
| **Parameter change** | $\|\theta_{t+1} - \theta_t\| < \epsilon$ | General |
| **Loss change** | $\|L_{t+1} - L_t\| < \epsilon$ | General |
| **Max iterations** | $t > t_{max}$ | Safeguard |

### Convergence Rates

| Rate | Definition | Example |
|------|------------|---------|
| **Linear (O(cⁿ))** | $\|e_{t+1}\| \leq c \|e_t\|$, $c < 1$ | Gradient descent |
| **Superlinear** | $\frac{\|e_{t+1}\|}{\|e_t\|} \to 0$ | BFGS, L-BFGS |
| **Quadratic** | $\|e_{t+1}\| \leq c \|e_t\|^2$ | Newton's method |

---

## Worked Example: LLN Convergence

> [!example] Problem
> Show that sample mean converges in probability to population mean.

**Solution:**

By Chebyshev's inequality:
$$P(|\bar{X}_n - \mu| \geq \epsilon) \leq \frac{\text{Var}(\bar{X}_n)}{\epsilon^2} = \frac{\sigma^2}{n\epsilon^2}$$

As $n \to \infty$, the right side $\to 0$.

Therefore: $\bar{X}_n \xrightarrow{p} \mu$ ✓

---

## Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

# Demonstration: Sample mean converging to population mean
np.random.seed(42)
true_mean = 5
population = np.random.exponential(true_mean, 100000)

# Track running mean
n_samples = np.arange(1, 10001)
running_means = np.cumsum(np.random.choice(population, 10000)) / n_samples

# Plot convergence
plt.figure(figsize=(10, 5))
plt.plot(n_samples, running_means, alpha=0.7)
plt.axhline(true_mean, color='red', linestyle='--', label=f'True Mean = {true_mean}')
plt.xlabel('Sample Size (n)')
plt.ylabel('Sample Mean')
plt.title('Convergence in Probability (LLN)')
plt.legend()
plt.xscale('log')
plt.grid(alpha=0.3)
plt.show()

# Gradient descent convergence check
def gradient_descent(f, grad, x0, lr=0.01, tol=1e-6, max_iter=1000):
    x = x0
    history = [x]
    for i in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            print(f"Converged in {i} iterations")
            break
        x = x - lr * g
        history.append(x)
    return x, history

# Example: minimize (x-3)^2
f = lambda x: (x - 3)**2
grad = lambda x: 2 * (x - 3)
x_opt, hist = gradient_descent(f, grad, x0=0.0)
print(f"Optimal x: {x_opt:.6f}")
```

---

## R Implementation

```r
set.seed(42)

# LLN demonstration
true_mean <- 5
samples <- rexp(10000, rate = 1/true_mean)
running_mean <- cumsum(samples) / seq_along(samples)

plot(running_mean, type = "l", 
     xlab = "n", ylab = "Sample Mean",
     main = "Convergence to True Mean")
abline(h = true_mean, col = "red", lty = 2)
```

---

## Signs of Non-Convergence

| Symptom | Possible Cause | Fix |
|---------|----------------|-----|
| **Loss oscillating** | Learning rate too high | Reduce learning rate |
| **Gradients exploding** | Numerical instability | Gradient clipping, smaller lr |
| **Stuck at boundary** | Constrained optimization | Check constraints |
| **Slow progress** | Poor conditioning | Use adaptive methods (Adam) |

---

## Related Concepts

### Statistical Convergence
- [[30_Knowledge/Stats/01_Foundations/Law of Large Numbers\|Law of Large Numbers]] - $\bar{X}_n \xrightarrow{p} \mu$
- [[30_Knowledge/Stats/01_Foundations/Central Limit Theorem (CLT)\|Central Limit Theorem (CLT)]] - $\sqrt{n}(\bar{X}_n - \mu) \xrightarrow{d} N(0,\sigma^2)$
- [[30_Knowledge/Stats/01_Foundations/Chebyshev's Inequality\|Chebyshev's Inequality]] - Used in convergence proofs

### Optimization
- [[30_Knowledge/Stats/04_Supervised_Learning/Gradient Descent\|Gradient Descent]] - Common algorithm needing convergence check
- [[30_Knowledge/Stats/01_Foundations/Loss Function\|Loss Function]] - What optimization minimizes

### Other Related Topics
- [[30_Knowledge/Stats/04_Supervised_Learning/Overfitting\|Overfitting]]

{ .block-language-dataview}

---

## References

1. Nocedal, J., & Wright, S. (2006). *Numerical Optimization* (2nd ed.). Springer. [Available online](https://link.springer.com/book/10.1007/978-0-387-40065-5)

2. Casella, G., & Berger, R. L. (2002). *Statistical Inference* (2nd ed.). Duxbury. Chapter 5.

3. Billingsley, P. (1995). *Probability and Measure* (3rd ed.). Wiley.
