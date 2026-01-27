---
{"dg-publish":true,"permalink":"/stats/02-hypothesis-testing/hypothesis-testing-p-value-and-ci/","tags":["Foundations","Hypothesis-Testing","Inference"]}
---

## Definition

> [!abstract] Core Statement
> **Hypothesis Testing** is a statistical method used to make inferences about a population based on sample data. It involves formulating a ==Null Hypothesis ($H_0$)==, collecting data, computing a test statistic, and deciding whether the evidence is strong enough to ==reject $H_0$== in favor of an Alternative Hypothesis ($H_1$).

---

> [!tip] Intuition (ELI5): The Superpower Test
> A friend claims they can **guess jelly bean flavors** just by touching them.
> 1. **$H_0$:** They are just guessing (Luck).
> 2. **$H_1$:** They have a superpower.
> They get 10/10 right. The **P-Value** is the probability a normal person would get 10 right by luck (e.g., 0.001). Since 0.001 is smaller than your threshold (0.05), you say "I don't think it's luck anymore!" and reject the Null.

> [!example] Real-Life Example: New Feature Launch
> A tech company splits users: half get an old algorithm, half get the new one. They test if the **click-rate** increases. If they see a 2% lift with a p-value of 0.001, they "reject the null" (that the algorithms are the same) and launch the new version.

---

## Purpose

1.  To determine whether observed data is consistent with a specific claim (e.g., "The drug has no effect").
2.  To provide a framework for making decisions under uncertainty.
3.  To quantify the strength of evidence against $H_0$ via the **p-value**.

---

## When to Use

> [!success] Use Hypothesis Testing When...
> - You have a specific claim to test (e.g., "The mean is 50").
> - You want to decide between two competing hypotheses.
> - You need a standardized framework for scientific inquiry.

> [!failure] Limitations of NHST
> - It does **not** tell you the probability that $H_0$ is true.
> - A significant p-value does not imply a *large* or *important* effect.
> - Over-reliance can lead to "p-hacking" and reproducibility issues.

---

## Theoretical Background

### The Hypotheses

| Hypothesis | Symbol | Description |
|------------|--------|-------------|
| **Null Hypothesis** | $H_0$ | The default assumption; typically "no effect" or "no difference." |
| **Alternative Hypothesis** | $H_1$ or $H_a$ | The claim we are testing for; "there is an effect." |

### The P-Value

> [!important] Critical Definition
> The **p-value** is the probability of obtaining a test statistic ==at least as extreme== as the one observed, **assuming $H_0$ is true**.
> $$ P\text{-value} = P(\text{Observed Data or More Extreme} | H_0 \text{ is True}) $$

![P-value Visualization](https://upload.wikimedia.org/wikipedia/commons/0/07/P-value_in_statistical_significance_testing.svg)

**Interpretation:**
| P-Value | Evidence | Decision |
|---------|----------|----------|
| **Small ($< \alpha$)** | Unlikely under $H_0$. Surprising. | **Reject $H_0$.** |
| **Large ($\ge \alpha$)** | Consistent with $H_0$. Expected. | **Fail to Reject $H_0$.** |

### Significance Level ($\alpha$)

The pre-defined threshold for rejecting $H_0$. Commonly $\alpha = 0.05$ (5%).

### Confidence Intervals (CI)

A **Confidence Interval** is a range of plausible values for the population parameter.
- A 95% CI means: "If we repeated this experiment many times, 95% of the calculated intervals would contain the true parameter."

> [!tip] CI and P-Value Relationship
> - If the 95% CI for a **difference** contains 0, the result is **not significant** at $\alpha = 0.05$.
> - If the 95% CI for an **Odds Ratio** contains 1, the result is **not significant**.

---

## Decision Errors

| . | **$H_0$ is True** | **$H_0$ is False** |
|---|-------------------|-------------------|
| **Reject $H_0$** | **Type I Error ($\alpha$)** False Positive | **Correct Decision** True Positive (Power = $1 - \beta$) |
| **Fail to Reject $H_0$** | **Correct Decision** True Negative | **Type II Error ($\beta$)** False Negative |

- **Type I Error ($\alpha$):** Finding an effect that doesn't exist.
- **Type II Error ($\beta$):** Missing an effect that does exist.
- **Power ($1 - \beta$):** The probability of correctly detecting a true effect. Aim for Power $\ge 0.80$.

---

## Assumptions

- [ ] **Random Sampling:** The sample must be representative of the population.
- [ ] **Independence:** Observations are independent of each other.
- [ ] **Test-Specific Assumptions:** Each test has its own (e.g., normality for t-tests).

---

## Limitations

> [!warning] Common Pitfalls
> 1.  **P-value is NOT $P(H_0 | \text{Data})$.** It is $P(\text{Data} | H_0)$. These are not the same (Prosecutor's Fallacy).
> 2.  **Statistical Significance $\neq$ Practical Significance.** A p-value of 0.001 for a tiny effect (e.g., 0.001 kg weight loss) is meaningless. Always report [[stats/02_Hypothesis_Testing/Effect Size Measures\|Effect Size Measures]].
> 3.  **Multiple Comparisons Problem:** Running 20 tests at $\alpha = 0.05$ guarantees roughly 1 false positive by chance. Use [[stats/02_Hypothesis_Testing/Bonferroni Correction\|Bonferroni Correction]] or FDR.
> 4.  **Dichotomization:** Treating $p = 0.049$ as "significant" and $p = 0.051$ as "not significant" ignores the inherent uncertainty.

---

## Python Implementation

```python
from scipy import stats
import numpy as np

# Scenario: We test if a coin is biased (H0: p = 0.5)
# Observed: 60 heads in 100 tosses.

# Binomial Test
result = stats.binomtest(k=60, n=100, p=0.5, alternative='two-sided')

print(f"P-value: {result.pvalue:.4f}")
print(f"95% CI for p: {result.proportion_ci(confidence_level=0.95)}")

if result.pvalue < 0.05:
    print("Reject H0: The coin is likely biased.")
else:
    print("Fail to Reject H0: Could be fair.")
```

---

## R Implementation

```r
# Scenario: 60 Heads, 100 Tosses, H0: p = 0.5
test_res <- binom.test(x = 60, n = 100, p = 0.5)

print(test_res)

# Output:
# - p-value
# - 95% Confidence Interval
# - Sample estimate of p

# Interpretation
if(test_res$p.value < 0.05) {
  cat("Reject H0: Coin is biased.\n")
} else {
  cat("Fail to Reject H0: Coin may be fair.\n")
}
```

---

## Interpretation Guide

| Scenario | Interpretation |
|----------|----------------|
| p = 0.03 | Evidence against $H_0$ at the 5% level. Reject $H_0$. |
| p = 0.15 | Not enough evidence against $H_0$. Fail to reject. |
| 95% CI = [1.2, 3.5] for OR | The effect is significant (doesn't contain 1) and the OR is between 1.2 and 3.5. |
| 95% CI = [-0.5, 0.8] for Diff | The effect is NOT significant (contains 0). |

---

## Related Concepts

- [[stats/02_Hypothesis_Testing/Type I & Type II Errors\|Type I & Type II Errors]]
- [[stats/02_Hypothesis_Testing/Power Analysis\|Power Analysis]]
- [[stats/02_Hypothesis_Testing/Confidence Intervals\|Confidence Intervals]]
- [[stats/02_Hypothesis_Testing/Effect Size Measures\|Effect Size Measures]]
- [[stats/02_Hypothesis_Testing/Bonferroni Correction\|Bonferroni Correction]]
- [[stats/01_Foundations/Bayesian Statistics\|Bayesian Statistics]] - An alternative framework.

---

## References

- **Book:** Wasserman, L. (2004). *All of Statistics*. Springer. [Springer Link](https://link.springer.com/book/10.1007/978-0-387-21736-9)
- **Book:** Casella, G., & Berger, R. L. (2002). *Statistical Inference* (2nd ed.). Duxbury. [Cengage Link](https://www.cengage.com/c/statistical-inference-2e-casella/9780534243128/)
- **Article:** Wasserstein, R. L., & Lazar, N. A. (2016). The ASA statement on p-values: Context, process, and purpose. *The American Statistician*, 70(2), 129-133. [Taylor & Francis Link](https://www.tandfonline.com/doi/full/10.1080/00031305.2016.1154108)
