---
{"dg-publish":true,"permalink":"/stats/02-hypothesis-testing/one-sample-t-test/","tags":["Hypothesis-Testing","Parametric-Tests","T-Test"]}
---

## Definition

> [!abstract] Core Statement
> The **One-Sample t-test** determines whether the sample mean is statistically different from a known or hypothesized **population mean** ($\mu_0$). It is used when the population standard deviation is **unknown**.

$$ t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}} $$

---

## Purpose

1.  **Quality Control:** Does the average weight of bags produced equal the target (e.g., 500g)?
2.  **Benchmarking:** Does our school's average score differ from the national average?
3.  **Difference Scores:** Testing if the average difference (Pre - Post) is non-zero (Paired t-test is technically a one-sample test on differences).

---

## When to Use

> [!success] Use When...
> - You have **one group** of continuous data.
> - You want to compare the mean to a **specific value** ($\mu_0$).
> - Population Deviation ($\sigma$) is **Unknown** (use $s$ from sample).
> - $n < 30$ (if $n > 30$, t converges to z, but t-test is still safe).

---

## Assumptions

- [ ] **Independence:** Observations are independent.
- [ ] **Normality:** Sample data should be approximately normally distributed (or $n$ large enough for [[stats/01_Foundations/Central Limit Theorem (CLT)\|Central Limit Theorem (CLT)]]).

---

## Worked Example: Potato Chips

> [!example] Problem
> Bags of chips claim to hold **200g** ($\mu_0 = 200$).
> You sample **10 bags**: Mean = 195g, StDev = 5g.
> 
> 1.  **Calculate SE:** $s / \sqrt{n} = 5 / \sqrt{10} \approx 1.58$.
> 2.  **Calculate t:**
>     $$ t = \frac{195 - 200}{1.58} = \frac{-5}{1.58} \approx -3.16 $$
> 3.  **Decision:**
>     -   Degrees of freedom = $10 - 1 = 9$.
>     -   Critical t ($\alpha=0.05$, two-tailed) $\approx \pm 2.26$.
>     -   $|-3.16| > 2.26$. **Reject Null.**
> 
> **Conclusion:** The bags are significantly underfilled.

---

## Python Implementation

```python
import scipy.stats as stats

data = [198, 195, 202, 190, 194, 199, 195, 192, 196, 189]
mu_0 = 200

t_stat, p_val = stats.ttest_1samp(data, popmean=mu_0)

print(f"T-statistic: {t_stat:.3f}")
print(f"P-value: {p_val:.4f}")

if p_val < 0.05:
    print("Reject Null: Mean is different from 200.")
```

---

## Related Concepts

- [[stats/02_Hypothesis_Testing/Student's T-Test\|Student's T-Test]] - The two-sample version.
- [[stats/02_Hypothesis_Testing/Z-Test\|Z-Test]] - Used if $\sigma$ is known.
- [[stats/02_Hypothesis_Testing/Confidence Intervals\|Confidence Intervals]] - The interval version of this test.
