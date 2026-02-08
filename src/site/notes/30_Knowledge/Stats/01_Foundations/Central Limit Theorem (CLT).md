---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/central-limit-theorem-clt/","tags":["probability","foundations"]}
---

## Definition

> [!abstract] Core Statement
> The **Central Limit Theorem (CLT)** states that the sampling distribution of the sample mean will approximate a **Normal Distribution** as the sample size ($n$) becomes sufficiently large, ==regardless of the shape of the population distribution==.

![Central Limit Theorem Visualization](https://upload.wikimedia.org/wikipedia/commons/7/7b/IllustrationCentralTheorem.png)

---

> [!tip] Intuition (ELI5): The Soup Taster
> Imagine you made a giant pot of soup. Some parts might be saltier than others (messy distribution). If you take a tiny sip ($n=1$), it might be too salty or too bland. But if you take a large bowl (big $n$), and you do this 100 times, the average saltiness of those bowls will always form a perfect bell curve, centered around the true saltiness of the whole pot.

---

## Purpose

The CLT is the **foundational pillar** of inferential statistics. It justifies:
1.  Using Z-scores and T-scores to calculate [[30_Knowledge/Stats/02_Statistical_Inference/Confidence Intervals\|Confidence Intervals]].
2.  Using parametric tests (T-tests, ANOVA) on non-normal data when $n$ is large.
3.  The assumption of normality for sample means in hypothesis testing.

---

## When to Use

> [!success] Rely on CLT When...
> - You are working with **sample means** (not raw individual data points).
> - Your sample size is **$n \ge 30$** (classic rule of thumb).
> - You need to make inferences about a population mean from a sample.

> [!failure] CLT Does NOT Apply When...
> - Analyzing individual observations (not means).
> - Sample size is very small ($n < 15$) *and* the population is heavily skewed.
> - Data has extreme outliers that distort the mean.

---

## Theoretical Background

### The Formal Statement

Let $X_1, X_2, \dots, X_n$ be a random sample of size $n$ from a population with mean $\mu$ and finite variance $\sigma^2$. Define the sample mean:
$$ \bar{X}_n = \frac{1}{n} \sum_{i=1}^{n} X_i $$

As $n \to \infty$, the standardized sample mean converges in distribution to a Standard Normal:
$$ Z_n = \frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} N(0, 1) $$

### Key Implications

| Concept | Formula | Meaning |
|---------|---------|---------|
| **Expected Value** | $E[\bar{X}] = \mu$ | The average of sample averages equals the population mean. |
| **Standard Error (SE)** | $SE = \frac{\sigma}{\sqrt{n}}$ | As $n$ increases, the spread of the sampling distribution **shrinks**. |
| **Precision** | $\propto \sqrt{n}$ | To double precision, you need $4 \times$ the sample size. |

---

## Assumptions

- [ ] **Random Sampling:** Each observation must be drawn randomly from the population.
- [ ] **Independence:** Observations must be independent of each other. (If sampling without replacement, sample should be $< 10\%$ of population).
- [ ] **Finite Variance:** The population must have a finite standard deviation ($\sigma < \infty$). (Violated by Cauchy distribution).

---


## Worked Example: Delivery Truck Safety

> [!example] Problem
> A delivery truck can carry a maximum load of **2100 kg**. It is loaded with **40 boxes**.
> - The weight of any individual box is a random variable with **Mean ($\mu$) = 50 kg** and **Standard Deviation ($\sigma$) = 10 kg**.
> - The distribution of box weights is **non-normal** (some heavy outliers, some light).
> 
> **Question:** What is the probability that the total weight exceeds the 2100 kg limit?

**Solution:**

1.  **Identify Parameters:**
    -   $n = 40$ (Sample size > 30, so CLT applies).
    -   $\mu_{sum} = n \times \mu = 40 \times 50 = 2000 \text{ kg}$.
    -   $\sigma_{sum} = \sqrt{n} \times \sigma = \sqrt{40} \times 10 \approx 6.325 \times 10 = 63.25 \text{ kg}$.
    -   Target Value ($X$) = 2100 kg.

2.  **Calculate Z-Score:**
    -   Since $n$ is large, we approximate the distribution of the *sum* as Normal.
    $$ Z = \frac{X - \mu_{sum}}{\sigma_{sum}} = \frac{2100 - 2000}{63.25} = \frac{100}{63.25} \approx 1.58 $$

3.  **Find Probability:**
    -   Looking up $Z=1.58$ in a Z-table gives an area of $0.9429$ to the left.
    -   $P(Weight > 2100) = 1 - 0.9429 = 0.0571$.

**Interpretation:**
There is approximately a **5.7%** chance the truck will be overloaded, even though the expected weight (2000 kg) is well below the limit. This calculation relies on the CLT because the individual box weights are not normally distributed.

---

## Limitations

> [!warning] Pitfalls
> 1.  **The "Population Becomes Normal" Fallacy:** A common misconception is that if $n$ is large, the *original data* becomes normal. **False.** The population distribution remains exactly the same; only the *distribution of means* looks normal.
> 2.  **"Large enough" is relative:** For symmetric distributions, $n \ge 15$ may suffice. For highly skewed distributions (e.g., Exponential), $n \ge 50$ or more may be needed.
> 3.  **Outliers:** Extreme values can distort the mean, requiring even larger $n$ for normality.
> 4.  **Does not apply to medians or variances:** CLT is specifically about sample *means*. The sampling distribution of the median or variance has different properties.

---

## Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# --- Simulation ---
np.random.seed(42)

# 1. Create a Non-Normal Population (Exponential)
population = np.random.exponential(scale=2, size=100000)

# 2. Draw Many Samples and Calculate Means
sample_sizes = [5, 30, 100]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, n in enumerate(sample_sizes):
    sample_means = [np.mean(np.random.choice(population, n)) for _ in range(1000)]
    
    # Shapiro-Wilk Test for Normality
    _, p_val = stats.shapiro(sample_means)
    
    sns.histplot(sample_means, kde=True, ax=axes[i], stat='density')
    axes[i].set_title(f'n = {n} (Shapiro p = {p_val:.3f})')

plt.suptitle('CLT in Action: Exponential Pop -> Normal Sampling Distribution')
plt.tight_layout()
plt.show()
```

---

## R Implementation

```r
library(ggplot2)

set.seed(42)

# 1. Population (Exponential - Right Skewed)
pop <- rexp(100000, rate = 0.5)

# 2. Draw Samples
simulate_means <- function(n, reps = 1000) {
  replicate(reps, mean(sample(pop, n)))
}

means_5 <- simulate_means(5)
means_30 <- simulate_means(30)
means_100 <- simulate_means(100)

# 3. Combine and Plot
df <- data.frame(
  mean = c(means_5, means_30, means_100),
  n = factor(rep(c("n=5", "n=30", "n=100"), each=1000), 
             levels=c("n=5", "n=30", "n=100"))
)

ggplot(df, aes(x=mean, fill=n)) +
  geom_histogram(aes(y=..density..), bins=30, alpha=0.7) +
  stat_function(fun = dnorm, 
                args = list(mean = mean(pop), sd = sd(pop)/sqrt(30)),
                color = "red", size=1, linetype="dashed") +
  facet_wrap(~n) +
  labs(title = "CLT Simulation: Exponential -> Normal") +
  theme_minimal()
```

---

## Interpretation Guide

| Observation | Meaning |
|-------------|---------|
| **Sampling distributon is Bell-Shaped** | CLT is working; parametric inference (T-tests, Z-tests) is valid. |
| **$n < 30$ and Data Skewed** | CLT **cannot** be relied upon. Use non-parametric tests (e.g., Wilcoxon). |
| **Standard Error decreases** | As $n \uparrow$, the mean becomes more precise ($SE = \sigma/\sqrt{n}$). |
| **Sum vs Mean** | CLT applies to **Sums** as well as Means (Sum $\sim N(n\mu, n\sigma^2)$). |

---

## Related Concepts

- [[30_Knowledge/Stats/01_Foundations/Normal Distribution\|Normal Distribution]] - The target distribution.
- [[30_Knowledge/Stats/01_Foundations/Law of Large Numbers\|Law of Large Numbers]] - Related but distinct (LLN is about accuracy, CLT is about distribution shape).
- [[30_Knowledge/Stats/01_Foundations/Standard Error\|Standard Error]] - Derived directly from CLT.
- [[30_Knowledge/Stats/01_Foundations/T-Distribution\|T-Distribution]] - Used when $\sigma$ is unknown.
- [[30_Knowledge/Stats/02_Statistical_Inference/Confidence Intervals\|Confidence Intervals]] - Built on CLT logic.

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## References

- **Book:** Rice, J. A. (2007). *Mathematical Statistics and Data Analysis* (3rd ed.). Duxbury. (Chapter 6) [Publisher Link](https://www.cengage.com/c/mathematical-statistics-and-data-analysis-3e-rice/9780534399429/)
- **Book:** Casella, G., & Berger, R. L. (2002). *Statistical Inference* (2nd ed.). Duxbury. (Chapter 5) [Publisher Link](https://www.routledge.com/Statistical-Inference/Casella-Berger/p/book/9781032593036)
- **Book:** Fischer, H. (2011). *A History of the Central Limit Theorem*. Springer. [Springer Link](https://link.springer.com/book/10.1007/978-0-387-87857-7)
- **Historical:** Laplace, P. S. (1812). *Théorie Analytique des Probabilités*. [Archive.org](https://archive.org/details/thorieanalytiqu01laplgoog)
