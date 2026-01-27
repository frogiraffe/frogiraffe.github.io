---
{"dg-publish":true,"permalink":"/stats/02-hypothesis-testing/chi-square-test/","tags":["Hypothesis-Testing","Non-Parametric","Categorical-Data"]}
---


## Definition

> [!abstract] Core Statement
> The **Chi-Square (χ²) Test** is a non-parametric test used to analyze **categorical data**. It compares observed frequencies to expected frequencies to determine if there is a statistically significant relationship or deviation from expectations.

**Two Main Types:**
1. **Goodness of Fit:** Does one categorical variable follow an expected distribution?
2. **Test of Independence:** Are two categorical variables related?

**Intuition (ELI5):** Imagine rolling a die 60 times. If fair, each side should appear ~10 times. If you get [5, 8, 12, 15, 10, 10], chi-square asks: "Is this weird enough to suspect the die is loaded, or just normal random variation?"

---

## When to Use

> [!success] Use Chi-Square When...
> - Data is **categorical** (counts/frequencies, not means).
> - You want to test if observed frequencies **differ from expected**.
> - You're checking if **two categorical variables are independent**.
> - Expected frequencies are **≥ 5** in each cell.

> [!failure] Do NOT Use Chi-Square When...
> - Expected cell counts are **< 5** — use [[stats/02_Hypothesis_Testing/Fisher's Exact Test\|Fisher's Exact Test]].
> - Data is **continuous** — use t-test, ANOVA, etc.
> - You want to compare **means**, not frequencies.
> - Observations are **not independent** (e.g., repeated measures).

---

## Theoretical Background

### The Chi-Square Statistic

$$
\chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i}
$$

Where:
- $O_i$ = Observed frequency in category $i$
- $E_i$ = Expected frequency in category $i$
- $k$ = Number of categories

**Interpretation:**
- Large $\chi^2$ → Observed deviates significantly from expected
- Small $\chi^2$ → Observed matches expected (no significant difference)

### Degrees of Freedom

| Test Type | Formula |
|-----------|---------|
| **Goodness of Fit** | $df = k - 1$ (k = number of categories) |
| **Independence** | $df = (r-1)(c-1)$ (r = rows, c = columns) |

### Expected Frequencies (Independence Test)

For cell at row $i$, column $j$:
$$
E_{ij} = \frac{(\text{Row}_i \text{ Total}) \times (\text{Column}_j \text{ Total})}{\text{Grand Total}}
$$

---

## Assumptions & Diagnostics

- [ ] **Categorical Data:** Must be counts, not percentages or means.
- [ ] **Independence:** Each observation is independent.
- [ ] **Expected Frequencies ≥ 5:** In each cell. If not, use Fisher's Exact Test.
- [ ] **Mutually Exclusive:** Each observation belongs to only one category.

### Effect Size: Cramér's V

$$
V = \sqrt{\frac{\chi^2}{n \cdot (k-1)}}
$$

Where $k = \min(\text{rows}, \text{cols})$

| Cramér's V | Interpretation |
|------------|----------------|
| 0.1 | Small effect |
| 0.3 | Medium effect |
| 0.5 | Large effect |


---

## References

- **Historical:** Pearson, K. (1900). On the criterion that a given system of deviations from the probable in the case of a correlated system of variables is such that it can be reasonably supposed to have arisen from random sampling. *Philosophical Magazine Series 5*, 50(302), 157-175. [Taylor & Francis Link](https://www.tandfonline.com/doi/abs/10.1080/14786440009463897)
- **Book:** Agresti, A. (2013). *Categorical Data Analysis* (3rd ed.). Wiley. [Wiley Link](https://www.wiley.com/en-us/Categorical+Data+Analysis,+3rd+Edition-p-9780470463635)
- **Book:** McHugh, M. L. (2013). The chi-square test of independence. *Biochemia Medica*, 23(2), 143-149. [DOI: 10.11613/BM.2013.018](https://doi.org/10.11613/BM.2013.018)

## Implementation

### Python

```python
import numpy as np
from scipy import stats
import pandas as pd

# ========== 1. GOODNESS OF FIT ==========
# Is this die fair?
observed = [12, 11, 15, 6, 8, 8]  # Actual rolls for sides 1-6
expected = [10, 10, 10, 10, 10, 10]  # Expected if fair (60 rolls / 6 sides)

chi2_stat, p_value = stats.chisquare(f_obs=observed, f_exp=expected)

print("=== GOODNESS OF FIT ===")
print(f"Chi-Square Statistic: {chi2_stat:.2f}")
print(f"P-value: {p_value:.4f}")
print(f"Degrees of Freedom: {len(observed) - 1}")

if p_value < 0.05:
    print("→ Die is likely biased (Reject H₀)")
else:
    print("→ Die appears fair (Fail to reject H₀)")

# ========== 2. TEST OF INDEPENDENCE ==========
# Is there a relationship between Gender and Product Preference?
#                Product A    Product B
# Men               30            20
# Women             10            40

contingency_table = np.array([
    [30, 20],  # Men
    [10, 40]   # Women
])

chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

print("\n=== TEST OF INDEPENDENCE ===")
print(f"Chi-Square Statistic: {chi2:.2f}")
print(f"P-value: {p:.4f}")
print(f"Degrees of Freedom: {dof}")
print(f"\nExpected Frequencies:\n{expected}")

if p < 0.05:
    print("→ Gender and Preference ARE related (Reject H₀)")
else:
    print("→ Gender and Preference are independent (Fail to reject H₀)")

# ========== 3. EFFECT SIZE: CRAMÉR'S V ==========
n = contingency_table.sum()
k = min(contingency_table.shape)
cramers_v = np.sqrt(chi2 / (n * (k - 1)))
print(f"\nCramér's V: {cramers_v:.3f}")

# ========== 4. CHECK EXPECTED FREQUENCIES ==========
if np.any(expected < 5):
    print("⚠️ Warning: Some expected frequencies < 5. Use Fisher's Exact Test.")
    # For 2x2 tables:
    odds_ratio, fisher_p = stats.fisher_exact(contingency_table)
    print(f"Fisher's Exact p-value: {fisher_p:.4f}")

# ========== 5. FROM PANDAS DATAFRAME ==========
df = pd.DataFrame({
    'Gender': ['M']*50 + ['F']*50,
    'Preference': ['A']*30 + ['B']*20 + ['A']*10 + ['B']*40
})

# Create contingency table
ct = pd.crosstab(df['Gender'], df['Preference'])
print("\n=== CROSSTAB ===")
print(ct)

chi2, p, dof, expected = stats.chi2_contingency(ct)
print(f"P-value: {p:.4f}")
```

### R

```r
# ========== 1. GOODNESS OF FIT ==========
observed <- c(12, 11, 15, 6, 8, 8)
expected_prob <- rep(1/6, 6)  # Equal probability

gof_test <- chisq.test(observed, p = expected_prob)
print(gof_test)

# ========== 2. TEST OF INDEPENDENCE ==========
# Contingency table
contingency <- matrix(c(30, 20, 10, 40), nrow = 2, byrow = TRUE,
                      dimnames = list(Gender = c("Men", "Women"),
                                     Product = c("A", "B")))
print(contingency)

chi_test <- chisq.test(contingency)
print(chi_test)

# Expected frequencies
print("Expected frequencies:")
print(chi_test$expected)

# Check if any expected < 5
if(any(chi_test$expected < 5)) {
  cat("Warning: Some expected < 5. Using Fisher's Exact Test:\n")
  print(fisher.test(contingency))
}

# ========== 3. EFFECT SIZE: CRAMÉR'S V ==========
library(rstatix)
cramer_v(contingency)

# Or manually:
n <- sum(contingency)
k <- min(dim(contingency))
V <- sqrt(chi_test$statistic / (n * (k - 1)))
cat("Cramér's V:", V, "\n")

# ========== 4. POST-HOC: STANDARDIZED RESIDUALS ==========
# Which cells contribute most to chi-square?
print("Standardized Residuals:")
print(chi_test$residuals)
# |residual| > 2 indicates significant deviation
```

---

## Interpretation Guide

| Output | Example Value | Interpretation | Edge Case/Warning |
|--------|---------------|----------------|-------------------|
| **χ² statistic** | 12.5 | Large deviation from expected. | Very large χ² may indicate outlier categories. |
| **p-value** | 0.002 | Strong evidence of association. | p < 0.05 doesn't mean large effect — check Cramér's V. |
| **p-value** | 0.35 | No significant association detected. | May be low power with small sample. |
| **Expected < 5** | Cell = 3.2 | Chi-square approximation invalid. | Use Fisher's Exact Test instead. |
| **Cramér's V** | 0.42 | Medium-to-large effect size. | Effect size matters more than p-value! |
| **df** | 2 | Indicates 3 categories (df = k-1). | Higher df = more categories compared. |

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Using Percentages Instead of Counts**
> - *Problem:* Input [50%, 30%, 20%] instead of [50, 30, 20].
> - *Result:* Completely wrong chi-square value.
> - *Solution:* Always use raw frequencies, never percentages.
>
> **2. Ignoring Expected Frequency Rule**
> - *Problem:* Running chi-square when expected cell count = 2.
> - *Result:* Chi-square approximation breaks down.
> - *Solution:* If expected < 5, use Fisher's Exact Test (2×2) or combine categories.
>
> **3. Testing Proportions Instead of Associations**
> - *Problem:* "Chi-square showed 60% prefer A, so A is better."
> - *Reality:* Chi-square tests if proportions differ from expected, not if one is "better."
> - *Solution:* Interpret as "association exists" or "distribution differs," not superiority.
>
> **4. Non-Independent Observations**
> - *Problem:* Same person counted in multiple categories (buys A AND B).
> - *Result:* Inflated significance due to violated independence.
> - *Solution:* Ensure each observation belongs to exactly one cell.

---

## Worked Numerical Example

> [!example] Testing if Survey Responses Differ by Age Group
> **Data:** 200 people surveyed about product preference (A, B, or C).
>
> | Age Group | Product A | Product B | Product C | Row Total |
> |-----------|-----------|-----------|-----------|-----------|
> | 18-35 | 30 | 25 | 45 | 100 |
> | 36-55 | 40 | 30 | 30 | 100 |
> | **Col Total** | 70 | 55 | 75 | 200 |
>
> **Step 1: Calculate Expected Frequencies**
> $$E_{11} = \frac{100 \times 70}{200} = 35$$
> $$E_{12} = \frac{100 \times 55}{200} = 27.5$$
> $$E_{13} = \frac{100 \times 75}{200} = 37.5$$
> (Same for row 2 since row totals are equal)
>
> **Step 2: Calculate Chi-Square**
> | Cell | O | E | (O-E)² | (O-E)²/E |
> |------|---|---|--------|----------|
> | 18-35, A | 30 | 35 | 25 | 0.71 |
> | 18-35, B | 25 | 27.5 | 6.25 | 0.23 |
> | 18-35, C | 45 | 37.5 | 56.25 | 1.50 |
> | 36-55, A | 40 | 35 | 25 | 0.71 |
> | 36-55, B | 30 | 27.5 | 6.25 | 0.23 |
> | 36-55, C | 30 | 37.5 | 56.25 | 1.50 |
>
> $$\chi^2 = 0.71 + 0.23 + 1.50 + 0.71 + 0.23 + 1.50 = 4.88$$
>
> **Step 3: Find Critical Value**
> - $df = (2-1)(3-1) = 2$
> - $\alpha = 0.05$
> - $\chi^2_{crit} = 5.99$
>
> **Step 4: Decision**
> - $\chi^2_{obs} = 4.88 < \chi^2_{crit} = 5.99$
> - **Fail to reject H₀**: No significant association between age group and product preference.
>
> **Effect Size:**
> $$V = \sqrt{\frac{4.88}{200 \times (2-1)}} = 0.156 \quad \text{(Small effect)}$$

---

## Related Concepts

**Alternatives:**
- [[stats/02_Hypothesis_Testing/Fisher's Exact Test\|Fisher's Exact Test]] — For small expected frequencies
- [[G-Test\|G-Test]] — Log-likelihood alternative

**Effect Size:**
- [[Cramér's V\|Cramér's V]] — Standardized effect size for χ²
- [[stats/01_Foundations/Odds Ratio\|Odds Ratio]] — For 2×2 tables

**Extensions:**
- [[stats/02_Hypothesis_Testing/McNemar's Test\|McNemar's Test]] — Paired categorical data
- [[Cochran-Mantel-Haenszel\|Cochran-Mantel-Haenszel]] — Stratified tables
