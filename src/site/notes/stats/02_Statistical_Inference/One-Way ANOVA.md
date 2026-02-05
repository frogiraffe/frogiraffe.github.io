---
{"dg-publish":true,"permalink":"/stats/02-statistical-inference/one-way-anova/","tags":["probability","hypothesis-testing","parametric-tests","anova"]}
---

## Definition

> [!abstract] Core Statement
> **One-Way ANOVA (Analysis of Variance)** tests whether the means of ==three or more independent groups== are equal. It is an extension of the t-test for multiple groups, using the F-statistic to compare between-group variance to within-group variance.

![ANOVA F-Test Rejection Region](https://upload.wikimedia.org/wikipedia/commons/7/7b/F-test_plot.svg)

---

> [!tip] Intuition (ELI5): The Teacher Test
> You want to know if **three different teachers** (A, B, and C) grade differently. ANOVA looks at:
> 1. How much grades vary **within** each teacher's class.
> 2. How much the class averages vary **between** the teachers.
> If the averages are very different compared to the internal "noise," ANOVA says: "At least one teacher is grading differently!"

> [!example] Real-Life Example: Diet Plans
> A site compares **three diets**: Keto, Low-Fat, and Mediterranean. After 3 months, they measure weight loss. One-Way ANOVA determines if the diet type truly matters, or if the differences are just random variation among individuals.

---

## Purpose

1.  Test for an overall difference among group means without inflating Type I error from multiple t-tests.
2.  Partition total variance into explained (between-group) and unexplained (within-group) components.

---

## When to Use

> [!success] Use One-Way ANOVA When...
> - Outcome is **continuous**.
> - There are **three or more independent groups**.
> - Data is approximately **normally distributed** within groups.
> - **Variances are approximately equal** across groups ([[stats/02_Statistical_Inference/Levene's Test\|Levene's Test]]).

> [!failure] Alternatives
> - **Unequal variances:** [[stats/02_Statistical_Inference/Welch's ANOVA\|Welch's ANOVA]].
> - **Non-normal data:** [[stats/02_Statistical_Inference/Kruskal-Wallis Test\|Kruskal-Wallis Test]].
> - **Two groups:** [[stats/02_Statistical_Inference/Student's T-Test\|Student's T-Test]] or [[stats/02_Statistical_Inference/Welch's T-Test\|Welch's T-Test]].

---

## Theoretical Background

### Hypotheses

- **$H_0$:** $\mu_1 = \mu_2 = \dots = \mu_k$ (All group means are equal).
- **$H_1$:** At least one $\mu_j$ is different.

> [!important] ANOVA Tells You *That* Not *Where*
> A significant ANOVA result only indicates that ==at least one group differs==. To find *which* groups differ, you must run **Post-Hoc Tests** (e.g., [[stats/02_Statistical_Inference/Tukey's HSD\|Tukey's HSD]]).

### The F-Ratio

$$
F = \frac{MS_{between}}{MS_{within}} = \frac{\text{Variance Explained by Groups}}{\text{Variance Within Groups}}
$$

| Term | Formula | Meaning |
|------|---------|---------|
| $SS_{between}$ | $\sum n_j (\bar{X}_j - \bar{X}_{grand})^2$ | Variability due to group differences. |
| $SS_{within}$ | $\sum \sum (X_{ij} - \bar{X}_j)^2$ | Variability within groups (noise). |
| $MS$ | $SS / df$ | Mean Square. |

**Large F** $\to$ Group means are distinct $\to$ Reject $H_0$.

---

## Assumptions

- [ ] **Independence:** Observations are independent.
- [ ] **Normality:** Data within each group is normally distributed. (Check per-group: [[stats/02_Statistical_Inference/Shapiro-Wilk Test\|Shapiro-Wilk Test]]).
- [ ] ==**Homogeneity of Variance:**== Variances are equal across groups. (Check: [[stats/02_Statistical_Inference/Levene's Test\|Levene's Test]]). If violated, use [[stats/02_Statistical_Inference/Welch's ANOVA\|Welch's ANOVA]].

---

## Limitations

> [!warning] Pitfalls
> 1.  **Sensitive to Variance Inequality:** ANOVA assumes equal variances. If violated, Type I error increases.
> 2.  **Does Not Identify Which Groups Differ:** Significant F-test requires post-hoc analysis.
> 3.  **Assumes Normality:** For small samples, non-normality inflates error.

---

## Python Implementation

```python
import scipy.stats as stats
import statsmodels.stats.multicomp as mc

# Data: Three groups
group1 = [10, 12, 14, 16, 18]
group2 = [20, 22, 24, 26, 28]
group3 = [15, 17, 19, 21, 23]

# 1. One-Way ANOVA
f_stat, p_val = stats.f_oneway(group1, group2, group3)
print(f"F-statistic: {f_stat:.3f}, p-value: {p_val:.4f}")

# 2. Post-Hoc (Tukey HSD) if p < 0.05
if p_val < 0.05:
    data = group1 + group2 + group3
    labels = ['G1']*5 + ['G2']*5 + ['G3']*5
    tukey = mc.MultiComparison(data, labels).tukeyhsd()
    print(tukey)
```

---

## R Implementation

```r
# 1. Fit ANOVA Model
model <- aov(Score ~ Group, data = df)

# 2. Summary (F-test)
summary(model)

# 3. Check Assumptions
plot(model, 1)  # Residuals vs Fitted (Homoscedasticity)
plot(model, 2)  # Q-Q Plot (Normality)

# 4. Post-Hoc Test (Tukey HSD)
TukeyHSD(model)

# 5. Visualize Tukey Results
plot(TukeyHSD(model))
```

---

## Worked Numerical Example

> [!example] Diet Plan Comparison
> **Scenario:** 3 Groups of 10 people each. Weight loss (lbs) after 1 month.
> - **Means:** Diet A = 5 lbs, Diet B = 7 lbs, Diet C = 12 lbs.
> - **Grand Mean:** 8 lbs.
> 
> **ANOVA Table:**
> - **Between-Group Variance (MS_B):** Large (driven by Diet C's 12 lbs).
> - **Within-Group Variance (MS_W):** 4.0 (Variation within diets).
> - **F-Statistic:** $MS_B / MS_W = 15.3$.
> - **Critical F:** ~3.35 (for $\alpha=0.05$).
> 
> **Result:** $15.3 > 3.35$ ($p < 0.001$). Reject $H_0$.
> **Conclusion:** *At least one* diet is effectively different. (Likely C vs A and C vs B).

---

## Interpretation Guide

| Output | Interpretation | Edge Case Notes |
|--------|----------------|-----------------|
| F = 15.3, p < 0.001 | At least one group mean is significantly different. | Does **not** say which one (A≠B? B≠C?). Run Tukey. |
| F < 1 | Within-group variability > Between-group. | Means are very similar, or noise is huge. $H_0$ holds. |
| Tukey: G1-G2 p < 0.05 | G1 and G2 are significantly different. | Confidence interval for Diff excludes 0. |
| Tukey: G1-G3 p = 0.45 | G1 and G3 are statistically indistinguishable. | CI includes 0. |
| F significant, but no Tukey significant? | **Rare Paradox**. Variance is partitioned broadly. | The overall pattern is non-random, but no *single pair* beats the strict pairwise threshold. |

---

## Common Pitfall Example

> [!warning] The Multiple T-Test Trap
> **Scenario:** You have 4 groups (A, B, C, D).
> **Bad Approach:** Run 6 separate t-tests (A-B, A-C, A-D, B-C, ...).
> 
> **The Problem (Alpha Inflation):**
> - Each t-test has 5% error chance.
> - With 6 tests, probability of *at least one* false positive is $\approx 1 - (0.95)^6 \approx 26\%$.
> - You are very likely to find a "significant" difference that doesn't exist.
> 
> **Correct Approach:**
> 1. Run One-Way ANOVA first.
> 2. **Only** if F is significant, run Post-Hoc tests (Tukey) which control error rate mathematically.

---

## Related Concepts

- [[stats/02_Statistical_Inference/Welch's ANOVA\|Welch's ANOVA]] - Robust to unequal variances.
- [[stats/02_Statistical_Inference/Tukey's HSD\|Tukey's HSD]] - Post-hoc pairwise comparisons.
- [[stats/02_Statistical_Inference/Bonferroni Correction\|Bonferroni Correction]] - Alternative multiple comparison adjustment.
- [[stats/02_Statistical_Inference/Kruskal-Wallis Test\|Kruskal-Wallis Test]] - Non-parametric alternative.
- [[stats/02_Statistical_Inference/MANOVA\|MANOVA]] - Multiple dependent variables.

---

## References

- **Book:** Montgomery, D. C. (2017). *Design and Analysis of Experiments* (9th ed.). Wiley. (Chapter 3) [Wiley Link](https://www.wiley.com/en-us/Design+and+Analysis+of+Experiments,+9th+Edition-p-9781119113478)
- **Book:** Field, A. (2018). *Discovering Statistics Using IBM SPSS Statistics* (5th ed.). Sage. (Chapter 12) [Publisher Link](https://us.sagepub.com/en-us/nam/discovering-statistics-using-ibm-spss-statistics/book257672)
- **Historical:** Fisher, R. A. (1925). *Statistical Methods for Research Workers*. Oliver & Boyd. (Introduced ANOVA) [Archive.org](https://archive.org/details/statisticalmetho00fish)