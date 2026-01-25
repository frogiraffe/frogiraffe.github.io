---
{"dg-publish":true,"permalink":"/stats/bootstrap-methods/","tags":["Statistics","Resampling","Machine-Learning","Inference","Uncertainty-Quantification"]}
---


# Bootstrap Methods

## Definition

> [!abstract] Core Statement
> **Bootstrap** is a ==resampling method== that estimates the sampling distribution of a statistic by repeatedly resampling (with replacement) from the observed data. It allows estimation of [[stats/Confidence Intervals\|Confidence Intervals]], [[stats/Standard Error\|Standard Error]], and hypothesis testing **without parametric assumptions**.

---

## Purpose

1. Estimate **confidence intervals** for statistics with unknown distributions.
2. Calculate **standard errors** without formulas.
3. Test hypotheses when assumptions (e.g., normality) are violated.
4. Assess **model stability** and **uncertainty**.

---

## When to Use

> [!success] Use Bootstrap When...
> - You need confidence intervals for **complex statistics** (median, correlation, custom metrics).
> - **Distributional assumptions** are questionable.
> - **Small sample size** makes asymptotics unreliable.
> - Analytical formulas for SE are unavailable or intractable.

> [!failure] Limitations
> - Requires **representative sample** (garbage in, garbage out).
> - Computationally intensive (requires many resamples, typically 1000-10000).

---

## Theoretical Background

### The Algorithm

1. **Original Sample:** You have data $X = \{x_1, x_2, \dots, x_n\}$.
2. **Resample:** Draw $n$ observations **with replacement** from $X$ to create bootstrap sample $X^*$.
3. **Calculate Statistic:** Compute statistic of interest (e.g., mean, median) on $X^*$.
4. **Repeat:** Repeat steps 2-3 many times (e.g., $B = 10{,}000$).
5. **Construct Distribution:** The collection of bootstrap statistics approximates the **sampling distribution**.

## Worked Example: Estimating Median House Price

> [!example] Problem
> You have a small sample of 5 house prices (in \$k): $[100, 100, 100, 400, 800]$.
> **Mean:** \$300k. (Skewed by the 800 outlier).
> **Median:** \$100k.
> 
> **Question:** How confident are we in this \$100k median? The formula for "Standard Error of the Median" is complex/unknown for this specific distribution.
> 
> **Bootstrap Solution:**
> 1.  **Resample 1:** $[100, 100, 400, 400, 800] \to \text{Median} = 400$.
> 2.  **Resample 2:** $[100, 100, 100, 100, 800] \to \text{Median} = 100$.
> 3.  **Resample 3:** $[100, 100, 100, 400, 800] \to \text{Median} = 100$.
> ... (Repeat 10,000 times).
> 
> **Resulting Distribution:**
> -   Most medians are \$100k.
> -   Some are \$400k (if we pick the large values often).
> -   **95% CI:** Maybe $[100, 400]$.
> 
> **Conclusion:** The median is likely \$100k, but there is significant risk it could be effectively \$400k if the population is heavy-tailed. The bootstrap reveals this instability.

---

## Assumptions

- [ ] **Original sample is representative:** If your original sample missed the "real" outliers, bootstrap can't create them. It assumes the sample *is* the population.
- [ ] **Method Choice:** Percentile method is simple, but "BCa" (Bias-Corrected) is better for skewed data.

---

## Limitations

> [!warning] Pitfalls
> 1.  **The "Clone Army" Problem:** If you have 3 distinct data points duplicated 100 times, bootstrap thinks you have $n=300$. It will underestimate variance. **Resample from unique valid observations.**
> 2.  **Not for Time Series:** Standard shuffling destroys trends. Use **Block Bootstrap** (resampling chunks of time).
> 3.  **Extreme Tails:** Bootstrap is bad at estimating Max/Min (the limits of the distribution) because it can never generate a value larger than the sample max.

---

## Python Implementation

```python
import numpy as np
from scipy import stats

# Original Sample
data = np.array([23, 25, 27, 22, 24, 26, 28, 21, 29, 30])
n = len(data)

# Bootstrap
B = 10000  # Number of bootstrap samples
bootstrap_means = []

for _ in range(B):
    resample = np.random.choice(data, size=n, replace=True)
    bootstrap_means.append(np.mean(resample))

bootstrap_means = np.array(bootstrap_means)

# 95% Confidence Interval (Percentile Method)
ci_lower = np.percentile(bootstrap_means, 2.5)
ci_upper = np.percentile(bootstrap_means, 97.5)

print(f"Bootstrap Mean: {np.mean(bootstrap_means):.2f}")
print(f"Bootstrap SE: {np.std(bootstrap_means):.2f}")
print(f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")

# Compare to parametric CI
from scipy.stats import t
param_ci = t.interval(0.95, df=n-1, loc=np.mean(data), scale=stats.sem(data))
print(f"Parametric 95% CI: [{param_ci[0]:.2f}, {param_ci[1]:.2f}]")
```

---

## R Implementation

```r
library(boot)

# Original Sample
data <- c(23, 25, 27, 22, 24, 26, 28, 21, 29, 30)

# Define Statistic Function (mean)
boot_mean <- function(data, indices) {
  return(mean(data[indices]))
}

# Bootstrap
set.seed(42)
boot_result <- boot(data, boot_mean, R = 10000)

# Results
print(boot_result)

# 95% Confidence Interval (Percentile and BCa)
boot.ci(boot_result, type = c("perc", "bca"))

# Compare to Parametric CI
t.test(data)$conf.int
```

---

## Interpretation Guide

| Output | Interpretation |
|--------|----------------|
| Bootstrap SE = 0.8 | Estimated standard error of the statistic. |
| 95% CI = [23.2, 26.8] | We are 95% confident the true parameter lies in this range. |
| Bootstrap CI wider than parametric | Data may be non-normal; bootstrap adjusts for this. |

---

## Related Concepts

- [[stats/Confidence Intervals\|Confidence Intervals]]
- [[stats/Standard Error\|Standard Error]]
- [[Permutation Tests\|Permutation Tests]] - Resampling for hypothesis testing.
- [[stats/Cross-Validation\|Cross-Validation]] - Resampling for model validation.
- [[Jackknife\|Jackknife]] - Alternative resampling method (leave-one-out).
