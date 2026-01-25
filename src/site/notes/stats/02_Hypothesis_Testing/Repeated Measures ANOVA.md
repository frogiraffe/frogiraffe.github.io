---
{"dg-publish":true,"permalink":"/stats/02-hypothesis-testing/repeated-measures-anova/","tags":["Hypothesis-Testing","ANOVA","Within-Subjects","Parametric-Tests"]}
---

## Definition

> [!abstract] Core Statement
> **Repeated Measures ANOVA** is used to compare means when the ==same subjects are measured multiple times== (e.g., pre-test, post-test, follow-up). It accounts for the **correlation between measurements** from the same individual, increasing statistical power.

---

## Purpose

1. Test if means differ across **repeated measurements** on the same subjects.
2. **Account for within-subject correlation** (violates independence assumption of standard ANOVA).
3. More **powerful** than between-subjects designs (subjects serve as their own controls).

---

## When to Use

> [!success] Use Repeated Measures ANOVA When...
> - **Same subjects** measured at multiple time points or conditions.
> - You have a **continuous dependent variable**.
> - **Sphericity assumption** is met (or corrected).

> [!failure] Alternatives
> - **Two time points only:** Use paired [[stats/02_Hypothesis_Testing/Student's T-Test\|Student's T-Test]].
> - **Sphericity violated:** Use **Greenhouse-Geisser correction** or **MANOVA**.
> - **Non-normal data:** Use [[stats/02_Hypothesis_Testing/Friedman Test\|Friedman Test]].

---

## Theoretical Background

### The Model

Similar to One-Way ANOVA, but includes a **subject effect**:
$$
Y_{ij} = \mu + \alpha_i + \pi_j + \varepsilon_{ij}
$$

| Term | Meaning |
|------|---------|
| $\mu$ | Grand mean |
| $\alpha_i$ | Effect of time/condition $i$ |
| $\pi_j$ | **Subject effect** (individual differences) |
| $\varepsilon_{ij}$ | Random error |

### Sphericity Assumption

> [!important] Critical Assumption
> **Sphericity** means the **variances of differences** between all pairs of repeated measures are equal.
> 
> **Test:** [[Mauchly's Test of Sphericity\|Mauchly's Test of Sphericity]].
> - If violated ($p < 0.05$), use **Greenhouse-Geisser** or **Huynh-Feldt** correction.

### Why Not Standard ANOVA?

Standard ANOVA assumes **independence**. Repeated measures from the same person are **correlated**, violating this. RM-ANOVA accounts for this correlation.

---

## Assumptions

- [ ] **Continuous dependent variable.**
- [ ] **Normality** of residuals (or differences).
- [ ] ==**Sphericity:**== Variances of all pairwise differences are equal. (Test with Mauchly's Test).
- [ ] **No missing data** (or handle appropriately with mixed models).

---

## Limitations

> [!warning] Pitfalls
> 1. **Sphericity violations are common.** Always check and apply corrections if needed.
> 2. **Missing data is problematic.** RM-ANOVA requires complete data for all time points. Use [[stats/03_Regression_Analysis/Linear Mixed Models (LMM)\|Linear Mixed Models (LMM)]] for flexibility.
> 3. **Carryover effects:** If conditions are sequential, earlier conditions may influence later ones.

---

## Python Implementation

```python
import pandas as pd
from statsmodels.stats.anova import AnovaRM

# Example: Pain Scores at 3 Time Points
data = pd.DataFrame({
    'Subject': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
    'Time': ['Pre', 'Mid', 'Post'] * 4,
    'Pain': [8, 6, 4, 7, 5, 3, 9, 7, 5, 6, 4, 2]
})

# Repeated Measures ANOVA
rm_anova = AnovaRM(data, depvar='Pain', subject='Subject', within=['Time'])
result = rm_anova.fit()
print(result.summary())
```

---

## R Implementation

```r
# Example Data (Wide Format)
df <- data.frame(
  Subject = 1:4,
  Pre = c(8, 7, 9, 6),
  Mid = c(6, 5, 7, 4),
  Post = c(4, 3, 5, 2)
)

# Convert to Long Format
library(tidyr)
df_long <- pivot_longer(df, cols = c(Pre, Mid, Post), 
                        names_to = "Time", values_to = "Pain")

# Repeated Measures ANOVA
library(rstatix)
res.aov <- anova_test(data = df_long, dv = Pain, wid = Subject, within = Time)
get_anova_table(res.aov)

# Check Sphericity (Mauchly's Test)
# If p < 0.05, apply Greenhouse-Geisser correction

# Alternative: ezANOVA
library(ez)
ezANOVA(data = df_long, dv = Pain, wid = Subject, within = Time, detailed = TRUE)
```

---

## Interpretation Guide

| Output | Interpretation |
|--------|----------------|
| F = 12.5, p = 0.002 | Time has a significant effect on Pain. |
| Mauchly's p < 0.05 | Sphericity violated. Use corrected results (Greenhouse-Geisser). |
| Greenhouse-Geisser ε = 0.75 | Moderate violation; correction applied. |

---

## Related Concepts

- [[stats/02_Hypothesis_Testing/One-Way ANOVA\|One-Way ANOVA]] - Between-subjects version.
- [[stats/02_Hypothesis_Testing/Mixed ANOVA (Between-Within)\|Mixed ANOVA (Between-Within)]] - Combines RM and between factors.
- [[stats/02_Hypothesis_Testing/Friedman Test\|Friedman Test]] - Non-parametric alternative.
- [[stats/03_Regression_Analysis/Linear Mixed Models (LMM)\|Linear Mixed Models (LMM)]] - More flexible; handles missing data.
- [[Mauchly's Test of Sphericity\|Mauchly's Test of Sphericity]]
