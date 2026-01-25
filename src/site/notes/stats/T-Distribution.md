---
{"dg-publish":true,"permalink":"/stats/t-distribution/","tags":["Statistics","Probability-Theory","Distributions","Foundations"]}
---


# T-Distribution

## Definition

> [!abstract] Core Statement
> The **Student's t-distribution** is a probability distribution used when estimating the mean of a normally distributed population in situations where the sample size is small ($n < 30$) and the population standard deviation ($\sigma$) is ==unknown==. It has **heavier tails** than the Normal Distribution, reflecting the increased uncertainty.

---

## Purpose

1.  **Constructing [[stats/Confidence Intervals\|Confidence Intervals]]** for the mean when $\sigma$ is unknown.
2.  **Hypothesis Testing** ([[stats/Student's T-Test\|Student's T-Test]], [[stats/Welch's T-Test\|Welch's T-Test]]) comparing means.
3.  **Regression Inference** - Coefficients in [[stats/Simple Linear Regression\|Simple Linear Regression]] are t-distributed under $H_0$.

---

## When to Use

> [!success] Use T-Distribution When...
> - Sample size is small ($n < 30$).
> - Population standard deviation ($\sigma$) is **unknown** and estimated by sample SD ($s$).
> - Data is approximately normal (or $n$ is large enough for CLT to apply).

> [!tip] Modern Practice
> Modern software uses the t-distribution by default for mean comparisons, even for large $n$. This is because as $n \to \infty$, the t-distribution converges to the Normal distribution anyway, so it is always safe to use.

---

## Theoretical Background

### Degrees of Freedom ($df$)

The shape of the t-distribution is controlled by the **degrees of freedom**, typically $df = n - 1$ for a single sample mean.

| $df$ | Behavior |
|------|----------|
| Small (e.g., 3) | Very heavy tails; accounts for high uncertainty. |
| Medium (e.g., 15) | Moderately heavy tails. |
| Large (e.g., $\ge 30$) | Almost indistinguishable from Normal. |

### Mathematical Definition

If $Z \sim N(0,1)$ and $V \sim \chi^2(k)$ are independent, then:
$$
T = \frac{Z}{\sqrt{V/k}} \sim t(k)
$$

### Comparison: T vs Normal

| Feature | Normal Distribution ($Z$) | T-Distribution ($t$) |
|---------|---------------------------|----------------------|
| **When $\sigma$ is** | Known | Unknown (Estimated by $s$) |
| **Parameter** | None (Standard) | Degrees of Freedom ($df$) |
| **Tail Thickness** | Thin | Heavy (More extreme outcomes) |
| **Critical Value (95%, Two-Tailed)** | 1.96 (Always) | Varies: 2.57 ($df=5$), 2.09 ($df=20$), 1.98 ($df=100$) |

---

## Worked Example: Confidence Interval

> [!example] Quality Control
> A factory produces screws with a target length of **50 mm**. A quality engineer takes a random sample of **$n=10$** screws.
> - **Sample Mean ($\bar{x}$):** 50.02 mm
> - **Sample Standard Deviation ($s$):** 0.05 mm
> 
> **Task:** Construct a **95% Confidence Interval** for the true mean length.

**Solution:**

1.  **Identify Parameters:**
    -   $n = 10$, so $df = n - 1 = 9$.
    -   $\alpha = 0.05$ (95% confidence).
    -   Since $n < 30$ and $\sigma$ is unknown, we use the **t-distribution**.

2.  **Find Critical Value ($t^*$):**
    -   From t-table for $df=9$ and two-tailed $\alpha=0.05$: $t^* \approx 2.262$.

3.  **Calculate Margin of Error (ME):**
    $$ ME = t^* \times \frac{s}{\sqrt{n}} = 2.262 \times \frac{0.05}{\sqrt{10}} $$
    $$ ME = 2.262 \times \frac{0.05}{3.162} = 2.262 \times 0.0158 \approx 0.0357 $$

4.  **Construct Interval:**
    $$ CI = [\bar{x} - ME, \bar{x} + ME] $$
    $$ CI = [50.02 - 0.036, 50.02 + 0.036] = [49.984, 50.056] $$

**Interpretation:**
We are 95% confident that the true mean length of the screws is between **49.98 mm** and **50.06 mm**. Since 50 mm is inside this interval, the process is likely on target.

---

## Assumptions

- [ ] **Random Sampling:** Data is a random sample from the population.
- [ ] **Independence:** Observations are independent.
- [ ] **Normality:** The population from which the sample is drawn is approximately normal. (Robust if $n$ is large due to CLT).

---

## Limitations

> [!warning] Pitfalls
> 1.  **Skewed Data in Small Samples:** If $n=10$ and the data is highly skewed (e.g., reaction times), the t-distribution is **not valid** and CIs will be incorrect. Use bootstrapping or Wilcoxon tests.
> 2.  **The "Z instead of T" Error:** A common mistake is using $Z$-score (1.96) for small samples. This produces an interval that is *too narrow*, leading to false confidence (underestimating uncertainty).
> 3.  **Outliers:** Since standard deviation $s$ is not robust to outliers, the t-statistic can be severely distorted by a single extreme value.

---

## Python Implementation

```python
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# --- Critical Values ---
# 95% Confidence, Two-Tailed
df_values = [5, 15, 30, 100]
for df in df_values:
    t_crit = stats.t.ppf(0.975, df=df)
    print(f"df = {df:3}: t_critical = {t_crit:.3f}")

# Compare: Normal
print(f"Z_critical (Normal): {stats.norm.ppf(0.975):.3f}")

# --- Visualization ---
x = np.linspace(-4, 4, 500)
plt.plot(x, stats.norm.pdf(x), 'k-', lw=2, label='Normal (Z)')
for df in [2, 5, 30]:
    plt.plot(x, stats.t.pdf(x, df=df), '--', label=f't (df={df})')
plt.legend()
plt.title("T-Distribution vs Normal: Tail Comparison")
plt.show()
```

---

## R Implementation

```r
# --- Critical Values ---
# 95% Confidence, Two-Tailed
df_values <- c(5, 15, 30, 100)
for (df in df_values) {
  t_crit <- qt(0.975, df = df)
  cat(sprintf("df = %3d: t_critical = %.3f\n", df, t_crit))
}

# Compare: Normal
cat(sprintf("Z_critical (Normal): %.3f\n", qnorm(0.975)))

# --- Visualization ---
curve(dnorm(x), from = -4, to = 4, lwd = 2, col = "black",
      ylab = "Density", main = "T vs Normal")
curve(dt(x, df = 2), add = TRUE, col = "red", lwd = 2, lty = 2)
curve(dt(x, df = 5), add = TRUE, col = "blue", lwd = 2, lty = 2)
curve(dt(x, df = 30), add = TRUE, col = "green", lwd = 2, lty = 2)
legend("topright", 
       legend = c("Normal", "t(df=2)", "t(df=5)", "t(df=30)"),
       col = c("black", "red", "blue", "green"), lwd = 2, lty = c(1,2,2,2))
```

---

## Interpretation Guide

| Value | Interpretation |
|-------|----------------|
| Value | Interpretation |
|-------|----------------|
| **$|t| > t_{critical}$** | Reject Null Hypothesis ($p < \alpha$). Result is significant. |
| **Small $df$ (e.g., 5)** | **Fat tails.** Much higher chance of extreme values than Normal. Critical values are large (e.g., > 2.5). |
| **Large $df$ (e.g., >30)** | **T $\approx$ Z.** The distribution is effectively Normal. |
| **Confidence Interval** | Wider than Z-interval reflecting uncertainty about $\sigma$. |

---

## Related Concepts

- [[stats/Normal Distribution\|Normal Distribution]] - The ideal scenario (known $\sigma$).
- [[stats/Student's T-Test\|Student's T-Test]] - Primary application.
- [[stats/Confidence Intervals\|Confidence Intervals]] - Calculated using t-scores.
- [[stats/Welch's T-Test\|Welch's T-Test]] - Robust version when variances differ.
- [[stats/Degrees of Freedom\|Degrees of Freedom]] - The shape parameter.
