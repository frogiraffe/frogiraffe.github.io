---
{"dg-publish":true,"permalink":"/stats/02-hypothesis-testing/welch-s-anova/","tags":["Hypothesis-Testing","ANOVA"]}
---


# Welch's ANOVA

## Overview

> [!abstract] Definition
> **Welch's ANOVA** is a robust alternative to standard [[stats/02_Hypothesis_Testing/One-Way ANOVA\|One-Way ANOVA]] used when the assumption of **homogeneity of variances** is violated. It adjusts the F-statistic and degrees of freedom to account for unequal variances across groups, reducing the risk of Type I errors.

---

## 1. Indications for Use

Welch's ANOVA is required when:
1. **Unequal Variances:** Levene's test is significant ($p < 0.05$).
2. **Unequal Sample Sizes:** The combination of unequal variances and unequal groups sizes dramatically affects standard ANOVA.

> [!note] Decision Rule
> If variances are unequal, standard ANOVA is invalid. If variances are equal, Welch's ANOVA performs nearly as well as standard ANOVA. Therefore, some statisticians recommend using Welch's ANOVA as the default procedure.

---

## 2. Comparison: Standard vs. Welch's

| Feature | Standard ANOVA | Welch's ANOVA |
|---------|----------------|---------------|
| **Assumption** | Equal Variances | Unequal Variances Allowed |
| **F-Statistic** | Based on pooled Mean Square Error | Based on weighted group variances |
| **Degrees of Freedom** | Integer | Non-integer (adjusted) |
| **Post-Hoc Test** | Tukey's HSD | Games-Howell |

---

## 3. The Problem with Standard ANOVA

When variances are unequal (heteroscedasticity), the pooled variance estimate in standard ANOVA becomes biased toward the variance of the larger groups.

- **Large n + Large Variance:** The standard ANOVA becomes conservative (Power loss).
- **Large n + Small Variance:** The standard ANOVA becomes liberal (Inflated Type I error rate).

Welch's ANOVA corrects for this by weighting the contribution of each group based on its own sample size and variance.

---

## 4. Post-Hoc Analysis: Games-Howell

If Welch's ANOVA indicates a significant difference, **Tukey's HSD** is inappropriate because it assumes equal variances. Instead, use the **Games-Howell** test.

- **Games-Howell:**
  - Does not assume equal variances.
  - Does not assume equal sample sizes.
  - Controls the Family-Wise Error Rate (FWER).

---

## 5. Python Implementation Example

```python
import pandas as pd
import pingouin as pg

# Welch's ANOVA
welch_res = pg.welch_anova(data=df, dv='score', between='group')
print("--- Welch's ANOVA Results ---")
print(welch_res)

# Effect Size (Partial Eta-Squared or Omega-Squared)
# Pingouin outputs np2 (partial eta-squared) by default
print(f"Effect Size (np2): {welch_res['np2'][0]:.3f}")

# Post-Hoc: Games-Howell
print("\n--- Games-Howell Post-Hoc ---")
gh_res = pg.pairwise_gameshowell(data=df, dv='score', between='group')
print(gh_res)
```

---

## 6. Related Concepts

- [[stats/02_Hypothesis_Testing/One-Way ANOVA\|One-Way ANOVA]] - The standard technique assuming equal variances.
- [[stats/02_Hypothesis_Testing/Levene's Test\|Levene's Test]] - Diagnostic for variance homogeneity.
- [[stats/02_Hypothesis_Testing/Welch's T-Test\|Welch's T-Test]] - The two-group equivalent of Welch's ANOVA.
- [[stats/02_Hypothesis_Testing/Kruskal-Wallis Test\|Kruskal-Wallis Test]] - Non-parametric alternative when normality is also violated.
