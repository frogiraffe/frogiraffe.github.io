---
{"dg-publish":true,"permalink":"/stats/02-statistical-inference/paired-t-test/","tags":["Hypothesis-Testing","Parametric","T-Test","Paired-Samples"]}
---


## Definition

> [!abstract] Core Statement
> The **Paired T-Test** compares the means of two **related measurements** from the same subjects to determine if there is a statistically significant difference. It tests whether the population mean difference ($\mu_d$) equals zero.

**Intuition (ELI5):** Imagine weighing yourself before and after a diet. You don't care about comparing yourself to other people; you only care: "Did *I* lose weight?" The paired t-test answers exactly this question by looking at the *change within each person*, not comparing different groups.

**Key Difference from Independent T-Test:**
- **Independent T-Test:** Compares two *different* groups (Treatment vs Control).
- **Paired T-Test:** Compares *same subjects* at two time points or under two conditions.

---

## When to Use

> [!success] Use Paired T-Test When...
> - You have **two measurements from the same subjects** (before/after, left/right hand, twin studies).
> - The outcome variable is **continuous** (weight, blood pressure, test scores).
> - The **differences** between paired observations are approximately **normally distributed**.
> - You want to control for **individual variability** (each person is their own control).

> [!failure] Do NOT Use Paired T-Test When...
> - Samples are **independent** (use [[stats/02_Statistical_Inference/Student's T-Test\|Student's T-Test]] instead).
> - You have **more than 2 related groups** (use [[stats/02_Statistical_Inference/Repeated Measures ANOVA\|Repeated Measures ANOVA]]).
> - Differences are **severely non-normal** with small sample size (use [[stats/02_Statistical_Inference/Wilcoxon Signed-Rank Test\|Wilcoxon Signed-Rank Test]]).
> - Outcome is **categorical** (use [[stats/02_Statistical_Inference/McNemar's Test\|McNemar's Test]]).

---

## Theoretical Background

### The Logic

Instead of comparing raw scores, we compute the **difference** for each pair:
$$
d_i = X_{1i} - X_{2i}
$$

Then we test whether the mean of differences ($\bar{d}$) is significantly different from zero.

### Hypotheses

$$
\begin{aligned}
H_0 &: \mu_d = 0 \quad \text{(No change / No difference)} \\
H_1 &: \mu_d \neq 0 \quad \text{(Two-tailed)} \\
&\text{or } \mu_d > 0 \text{ / } \mu_d < 0 \quad \text{(One-tailed)}
\end{aligned}
$$

### Test Statistic

$$
t = \frac{\bar{d} - 0}{s_d / \sqrt{n}} = \frac{\bar{d}}{SE_d}
$$

Where:
- $\bar{d}$ = Mean of the differences
- $s_d$ = Standard deviation of the differences
- $n$ = Number of pairs
- $df = n - 1$ (degrees of freedom)

### Effect Size: Cohen's d (Paired)

$$
d = \frac{\bar{d}}{s_d}
$$

| Cohen's d | Interpretation |
|-----------|----------------|
| 0.2 | Small effect |
| 0.5 | Medium effect |
| 0.8 | Large effect |

---

## Assumptions & Diagnostics

- [ ] **Paired/Dependent Samples:** Each observation in group 1 is uniquely matched to one in group 2.
- [ ] **Continuous Outcome:** The dependent variable is interval or ratio scale.
- [ ] **Normality of Differences:** The *differences* ($d_i$) should be approximately normally distributed.
    - Check: Shapiro-Wilk test on differences, Q-Q plot.
    - Robust to violations if $n \geq 30$ (CLT applies).
- [ ] **No Extreme Outliers:** Paired t-test is sensitive to outliers in the differences.
    - Check: Boxplot of differences.

**Visual Diagnostics:**
- **Histogram/Q-Q Plot of Differences:** Should look approximately bell-shaped. Severe skewness or heavy tails suggest using Wilcoxon test.
- **Boxplot of Differences:** No extreme outliers beyond 1.5×IQR.

---

## Implementation

### Python

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Data: Blood pressure before and after medication (same 10 patients)
before = np.array([120, 122, 143, 140, 135, 150, 128, 132, 145, 138])
after  = np.array([118, 120, 138, 135, 130, 145, 125, 130, 140, 135])

# ========== STEP 1: COMPUTE DIFFERENCES ==========
differences = before - after
print(f"Mean difference: {np.mean(differences):.2f}")
print(f"SD of differences: {np.std(differences, ddof=1):.2f}")

# ========== STEP 2: CHECK ASSUMPTIONS ==========
# Normality of differences (Shapiro-Wilk)
stat, p_norm = stats.shapiro(differences)
print(f"Shapiro-Wilk p-value: {p_norm:.4f}")
if p_norm < 0.05:
    print("⚠️ Differences may not be normal. Consider Wilcoxon test.")

# Visual check
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(differences, bins=5, edgecolor='black')
axes[0].set_title('Histogram of Differences')
stats.probplot(differences, dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot')
plt.tight_layout()
plt.show()

# ========== STEP 3: PERFORM PAIRED T-TEST ==========
t_stat, p_value = stats.ttest_rel(before, after)
print(f"\nt-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

# ========== STEP 4: EFFECT SIZE ==========
cohens_d = np.mean(differences) / np.std(differences, ddof=1)
print(f"Cohen's d: {cohens_d:.2f}")

# ========== STEP 5: DECISION ==========
alpha = 0.05
if p_value < alpha:
    print(f"\n✓ Reject H₀: Significant difference (p = {p_value:.4f})")
else:
    print(f"\n✗ Fail to reject H₀: No significant difference (p = {p_value:.4f})")
```

### R

```r
# Data: Blood pressure before and after medication
before <- c(120, 122, 143, 140, 135, 150, 128, 132, 145, 138)
after  <- c(118, 120, 138, 135, 130, 145, 125, 130, 140, 135)

# ========== STEP 1: COMPUTE DIFFERENCES ==========
differences <- before - after
cat("Mean difference:", mean(differences), "\n")
cat("SD of differences:", sd(differences), "\n")

# ========== STEP 2: CHECK ASSUMPTIONS ==========
# Normality test
shapiro.test(differences)
# If p > 0.05: Normality assumption met

# Visual check
par(mfrow = c(1, 2))
hist(differences, main = "Histogram of Differences", xlab = "Difference")
qqnorm(differences); qqline(differences)

# ========== STEP 3: PERFORM PAIRED T-TEST ==========
result <- t.test(before, after, paired = TRUE)
print(result)

# ========== STEP 4: EFFECT SIZE ==========
# Cohen's d for paired samples
cohens_d <- mean(differences) / sd(differences)
cat("\nCohen's d:", round(cohens_d, 2), "\n")

# ========== STEP 5: CONFIDENCE INTERVAL ==========
# Already included in t.test output
# 95% CI for the mean difference

# ========== ALTERNATIVE: One-tailed test ==========
# If hypothesis is: medication REDUCES blood pressure
t.test(before, after, paired = TRUE, alternative = "greater")
```

---

## Interpretation Guide

| Output | Example Value | Interpretation | Edge Case/Warning |
|--------|---------------|----------------|-------------------|
| $\bar{d}$ (Mean Difference) | 4.3 mmHg | On average, blood pressure dropped by 4.3 mmHg after treatment. | If negative: after > before (worsened). |
| $t$-statistic | 3.45 | Observed difference is 3.45 standard errors away from zero. | Very large t (>10) may indicate outliers inflating the effect. |
| $p$-value | 0.007 | 0.7% probability of seeing this difference if $H_0$ is true. Reject $H_0$ at α=0.05. | p < 0.001 doesn't mean "huge effect" — check Cohen's d. |
| $p$-value | 0.12 | 12% probability. Fail to reject $H_0$. No significant change detected. | May be Type II error (underpowered). Check sample size. |
| 95% CI for $\mu_d$ | [1.2, 7.4] | True mean difference likely between 1.2 and 7.4. Does not include 0 → significant. | Wide CI suggests high variability or small n. |
| 95% CI for $\mu_d$ | [-2.1, 5.8] | CI includes 0 → Result is not significant at α=0.05. | Inconclusive. Need larger sample. |
| Cohen's d | 0.72 | Medium-to-large effect size. Clinically meaningful change. | Even non-significant results can have meaningful effect sizes (underpowered study). |
| Cohen's d | 0.15 | Small effect. Even if p < 0.05, practical significance is questionable. | Large n can make trivial effects statistically significant. |

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Using Paired Test on Independent Samples**
> - *Problem:* Researcher compares Treatment group (n=30) vs Control group (n=30) using paired t-test.
> - *Reality:* These are different people! No pairing exists.
> - *Consequence:* Degrees of freedom are wrong, p-values are invalid.
> - *Solution:* Verify that each observation in group 1 has a unique match in group 2.
>
> **2. Ignoring Carryover Effects**
> - *Problem:* In "before medication vs after medication," the effect of the first measurement persists.
> - *Example:* Measuring learning. Students improve simply by taking the test twice (practice effect).
> - *Solution:* Use counterbalanced designs (half start with condition A, half with B).
>
> **3. Outliers in Differences**
> - *Problem:* One subject has a difference of +50 when others are around +5.
> - *Consequence:* Mean difference is pulled toward the outlier; t-test is skewed.
> - *Solution:* Identify outliers (boxplot), consider Winsorizing or using Wilcoxon test.
>
> **4. Confusing Statistical vs Practical Significance**
> - *Problem:* p = 0.001 but mean difference = 0.5 mmHg blood pressure reduction.
> - *Reality:* 0.5 mmHg is clinically meaningless, even if statistically significant.
> - *Solution:* Always report effect size (Cohen's d) alongside p-value.

---

## Worked Numerical Example

> [!example] Weight Loss Program Effectiveness
> **Research Question:** Does a 4-week diet program significantly reduce weight?
>
> **Data:** 8 participants weighed before and after the program.
>
> | Participant | Before (kg) | After (kg) | Difference ($d_i$) |
> |-------------|-------------|------------|-------------------|
> | 1 | 82 | 79 | 3 |
> | 2 | 76 | 75 | 1 |
> | 3 | 91 | 86 | 5 |
> | 4 | 68 | 68 | 0 |
> | 5 | 85 | 81 | 4 |
> | 6 | 79 | 77 | 2 |
> | 7 | 94 | 89 | 5 |
> | 8 | 73 | 71 | 2 |
>
> **Step 1: Calculate Mean and SD of Differences**
> $$\bar{d} = \frac{3+1+5+0+4+2+5+2}{8} = \frac{22}{8} = 2.75 \text{ kg}$$
>
> $$s_d = \sqrt{\frac{\sum(d_i - \bar{d})^2}{n-1}} = \sqrt{\frac{(3-2.75)^2 + (1-2.75)^2 + \dots + (2-2.75)^2}{7}} = 1.75 \text{ kg}$$
>
> **Step 2: Calculate t-statistic**
> $$t = \frac{\bar{d}}{s_d / \sqrt{n}} = \frac{2.75}{1.75 / \sqrt{8}} = \frac{2.75}{0.619} = 4.44$$
>
> **Step 3: Find Critical Value**
> - $df = n - 1 = 7$
> - $\alpha = 0.05$ (two-tailed)
> - $t_{crit} = 2.365$ (from t-table)
>
> **Step 4: Decision**
> - $|t_{obs}| = 4.44 > t_{crit} = 2.365$
> - **Reject $H_0$**: The diet program significantly reduced weight.
>
> **Step 5: Effect Size**
> $$d = \frac{2.75}{1.75} = 1.57 \quad \text{(Large effect!)}$$
>
> **Conclusion:** Participants lost an average of 2.75 kg (SD = 1.75). This reduction was statistically significant, $t(7) = 4.44$, $p < 0.01$, with a large effect size ($d = 1.57$).

---

## Related Concepts

**Prerequisites:**
- [[stats/02_Statistical_Inference/Student's T-Test\|Student's T-Test]] — Independent samples version
- [[stats/01_Foundations/T-Distribution\|T-Distribution]]
- [[stats/01_Foundations/Standard Error\|Standard Error]]

**Alternatives:**
- [[stats/02_Statistical_Inference/Wilcoxon Signed-Rank Test\|Wilcoxon Signed-Rank Test]] — Non-parametric alternative when normality fails

**Extensions:**
- [[stats/02_Statistical_Inference/Repeated Measures ANOVA\|Repeated Measures ANOVA]] — More than 2 time points
- [[stats/03_Regression_Analysis/Linear Mixed Models (LMM)\|Mixed Effects Models]] — Complex repeated measures with covariates

---

## References

- **Historical:** Student (Gosset, W. S.). (1908). The probable error of a mean. *Biometrika*, 6(1), 1-25. [DOI: 10.1093/biomet/6.1.1](https://doi.org/10.1093/biomet/6.1.1)
- **Book:** Field, A. (2018). *Discovering Statistics Using IBM SPSS Statistics* (5th ed.). Sage. (Chapter 10) [Publisher Link](https://us.sagepub.com/en-us/nam/discovering-statistics-using-ibm-spss-statistics/book257672)
- **Book:** Howell, D. C. (2013). *Statistical Methods for Psychology* (8th ed.). Cengage Learning. [Publisher Link](https://www.cengage.com/c/statistical-methods-for-psychology-8e-howell/9781111835484)
