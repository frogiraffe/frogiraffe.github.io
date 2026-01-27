---
{"dg-publish":true,"permalink":"/stats/02-statistical-inference/tukey-s-hsd/","tags":["Hypothesis-Testing","Post-Hoc","ANOVA","Multiple-Comparisons"]}
---

## Definition

> [!abstract] Core Statement
> **Tukey's HSD (Honestly Significant Difference)** is a **post-hoc test** used after a significant [[stats/02_Statistical_Inference/One-Way ANOVA\|One-Way ANOVA]] to determine ==which specific group pairs are significantly different==. It controls the **Family-Wise Error Rate (FWER)**, preventing inflation of Type I error from multiple comparisons.

---

## Purpose

1.  ANOVA tells you *that* at least one group is different; Tukey tells you *which* groups differ.
2.  Make all pairwise comparisons while maintaining an overall alpha level (e.g., 0.05).

---

## When to Use

> [!success] Use Tukey's HSD When...
> - ANOVA result is **significant** ($p < 0.05$).
> - You want to compare **all pairs of groups**.
> - Groups have **equal sample sizes** (or use Tukey-Kramer for unequal).

> [!failure] Alternatives
> - If you have **specific hypotheses** (not all pairs): Use planned contrasts or [[stats/02_Statistical_Inference/Bonferroni Correction\|Bonferroni Correction]].
> - If ANOVA assumptions are violated: Use [[stats/02_Statistical_Inference/Kruskal-Wallis Test\|Kruskal-Wallis Test]] followed by Dunn's test.

---

## Theoretical Background

### How It Works

Tukey's HSD calculates a **critical difference** based on the **Studentized Range Distribution**:
$$
HSD = q \cdot \sqrt{\frac{MS_{within}}{n}}
$$
where $q$ is the critical value from the Studentized Range table.

**Decision Rule:** Two groups are significantly different if $|\bar{X}_i - \bar{X}_j| > HSD$.

### Family-Wise Error Rate (FWER)

Without adjustment, running $k$ comparisons inflates Type I error:
$$
P(\text{At least 1 false positive}) = 1 - (1 - \alpha)^k
$$
Tukey adjusts the critical value so the overall error rate stays at $\alpha$.

---

## Assumptions

Inherits assumptions from [[stats/02_Statistical_Inference/One-Way ANOVA\|One-Way ANOVA]]:
- [ ] Independence.
- [ ] Normality.
- [ ] Homogeneity of variance.
- [ ] Equal (or approximately equal) sample sizes per group.

---

## Limitations

> [!warning] Pitfalls
> 1.  **Conservative with Unequal n:** For very unequal sample sizes, Tukey-Kramer is used but may be conservative.
> 2.  **Only for Pairwise Comparisons:** Does not handle complex contrasts (e.g., Group A+B vs Group C).
> 3.  **Requires Significant ANOVA:** Running Tukey without a significant F-test is discouraged (fishing for differences).

---

## Python Implementation

```python
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd

# Assume df has columns 'value' and 'group'
tukey = pairwise_tukeyhsd(endog=df['value'], groups=df['group'], alpha=0.05)

print(tukey)

# Visualization
tukey.plot_simultaneous()
```

---

## R Implementation

```r
# 1. Fit ANOVA
model <- aov(Value ~ Group, data = df)

# 2. Tukey HSD
tukey_result <- TukeyHSD(model)
print(tukey_result)

# 3. Plot Confidence Intervals
plot(tukey_result)
# If the interval crosses 0, the difference is NOT significant.
```

---

## Worked Numerical Example

> [!example] Crop Yields: Fertilizer A vs B vs C
> **ANOVA:** Significant ($p=0.01$).
> **Means:** A=50, B=55, C=60.
> **HSD Critical Diff:** 4.5.
> 
> **Comparisons:**
> - **|A - B| = 5:** $5 > 4.5$ $\to$ **Significant**.
> - **|B - C| = 5:** $5 > 4.5$ $\to$ **Significant**.
> - **|A - C| = 10:** $10 > 4.5$ $\to$ **Significant**.
> 
> **Result:** All groups are distinct. A < B < C.
> 
> *(If Critical Diff were 6.0, then A vs B would NOT be significant).*

---

## Interpretation Guide

| Comparison | Diff | p-adj | Interpretation | Edge Case Notes |
|------------|------|-------|----------------|-----------------|
| G2 - G1 | 5.2 | 0.002 | Significant difference ($p < 0.05$). | CI excludes 0. |
| G3 - G1 | 1.5 | 0.45 | NOT significant. | CI includes 0. |
| 95% CI | [-1, 4] | - | Range includes 0. No evidence of diff. | |
| p-adj = 0.051 | - | 0.051 | **Not Significant**. | Strict thresholding applies. |

**Plot Interpretation:**
- If the horizontal confidence interval for a pair **does not cross 0**, the difference is significant.

---

## Common Pitfall Example

> [!warning] Fishing for Significance
> **Scenario:** ANOVA p-value = 0.08 (Not significant).
> **Analyst:** "Let me run Tukey anyway, maybe Group 1 vs Group 5 is different."
> 
> **Result:** Tukey shows G1-G5 difference is significant.
> 
> **Problem:**
> - You violated the logic of the test ("Protected Least Significant Difference").
> - By ignoring the non-significant ANOVA, you are capitalizing on chance.
> - This is p-hacking.
> 
> **Rule:** If ANOVA F-test > 0.05, **STOP**. Do not run post-hoc tests.

---

## Related Concepts

- [[stats/02_Statistical_Inference/One-Way ANOVA\|One-Way ANOVA]]
- [[stats/02_Statistical_Inference/Bonferroni Correction\|Bonferroni Correction]] - More conservative alternative.
- [[stats/02_Statistical_Inference/Kruskal-Wallis Test\|Kruskal-Wallis Test]] - Non-parametric ANOVA.
- [[stats/01_Foundations/Multiple Comparisons Problem\|Multiple Comparisons Problem]]

---

## References

- **Historical:** Tukey, J. W. (1949). Comparing individual means in the analysis of variance. *Biometrics*. [JSTOR](https://www.jstor.org/stable/3001913)
- **Book:** Montgomery, D. C. (2017). *Design and Analysis of Experiments*. Wiley. [Wiley Link](https://www.wiley.com/en-us/Design+and+Analysis+of+Experiments%2C+9th+Edition-p-9781119113478)
- **Article:** Abdi, H., & Williams, L. J. (2010). Tukey’s honestly significant difference (HSD) test. [Link](https://personal.utdallas.edu/~herve/abdi-TukeyHSD2010-pretty.pdf)