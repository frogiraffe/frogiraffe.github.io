---
{"dg-publish":true,"permalink":"/30-knowledge/stats/03-regression-analysis/heteroscedasticity/","tags":["regression","modeling"]}
---


## Definition

> [!abstract] Core Statement
> **Heteroscedasticity** occurs when the variance of residuals (errors) is **not constant** across all levels of the independent variable(s). In regression, we assume errors have constant spread (homoscedasticity); when they don't, standard errors become unreliable.

![Heteroscedasticity Visualization](https://upload.wikimedia.org/wikipedia/commons/a/a5/Heteroscedasticity.png)

**Intuition (ELI5):** Imagine predicting income from years of education. For people with 10 years of education, incomes might range from $30K–$50K (tight spread). For PhDs, incomes might range from $50K–$500K (huge spread). This "widening cone" of prediction errors is heteroscedasticity.

**Why It Matters:**
- Coefficients ($\beta$) remain **unbiased** — the "average" line is still correct.
- But **standard errors are wrong** — usually underestimated.
- This leads to **inflated t-statistics** and **false positives** (Type I errors).

---

## When to Check

> [!success] Check for Heteroscedasticity When...
> - Running **any regression** (OLS, logistic, etc.).
> - Modeling **financial data** (returns often have time-varying volatility).
> - Outcome variable has **natural bounds** or **proportional variance** (e.g., counts, proportions).
> - Residual plots show a **fan/cone** shape.

> [!failure] Heteroscedasticity is LESS of an Issue When...
> - Using **robust standard errors** (HC0, HC3) — standard errors are corrected.
> - Your model is for **prediction only** — inference (p-values) doesn't matter.
> - You're using **WLS or GLS** — variance structure is explicitly modeled.
> - Sample size is **very large** — some robustness due to CLT.

---

## Theoretical Background

### Homoscedasticity Assumption

In OLS regression, we assume:
$$
\text{Var}(\varepsilon_i | X_i) = \sigma^2 \quad \forall i
$$

This means the variance of errors is **constant** regardless of predictor values.

### What Heteroscedasticity Looks Like

$$
\text{Var}(\varepsilon_i | X_i) = \sigma_i^2 \neq \sigma^2
$$

Common patterns:
- **Funnel/Cone:** Variance increases with $X$ (common in financial data).
- **Bow-tie:** Variance high at extremes, low in middle.
- **Clusters:** Different groups have different variances.

### Why Standard Errors Break

OLS estimates SE as:
$$
\text{SE}(\hat{\beta}) = \sqrt{\frac{\hat{\sigma}^2}{n \cdot \text{Var}(X)}}
$$

This formula assumes constant $\sigma^2$. With heteroscedasticity, this estimate is **biased** (usually too small), making coefficients appear more significant than they are.

---

## Assumptions & Diagnostics

### Detection Checklist

- [ ] **Visual: Residuals vs Fitted Plot** — The gold standard first check.
- [ ] **Breusch-Pagan Test** — Regresses squared residuals on predictors.
- [ ] **White Test** — More general; doesn't assume linear form.
- [ ] **Goldfeld-Quandt Test** — Compares variance in two subsets.

### Visual Diagnostics Guide

| Plot Pattern | Meaning | Action |
|--------------|---------|--------|
| **Random cloud** around zero | ✅ Homoscedasticity. You're fine. | None needed. |
| **Funnel opening right** | Variance increases with fitted values. | Try log transform of Y. |
| **Funnel opening left** | Variance decreases (rare). | Consider square root transform. |
| **Bow-tie** | Variance high at extremes. | Check for non-linearity first. |
| **Distinct clusters** | Different groups have different variance. | Consider grouped models or WLS. |

---

## Implementation

### Python

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
import matplotlib.pyplot as plt

# Sample data
np.random.seed(42)
X = np.random.uniform(1, 10, 100)
# Heteroscedastic errors: variance increases with X
y = 2 + 3*X + np.random.normal(0, X/2, 100)  # SD proportional to X

X_const = sm.add_constant(X)
model = sm.OLS(y, X_const).fit()

# ========== VISUAL DIAGNOSTIC ==========
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(model.fittedvalues, model.resid, alpha=0.6)
ax.axhline(0, color='red', linestyle='--')
ax.set_xlabel('Fitted Values')
ax.set_ylabel('Residuals')
ax.set_title('Residuals vs Fitted (Check for Funnel Shape)')
plt.show()

# ========== BREUSCH-PAGAN TEST ==========
bp_test = het_breuschpagan(model.resid, model.model.exog)
labels = ['LM Statistic', 'LM p-value', 'F Statistic', 'F p-value']
print("\nBreusch-Pagan Test:")
print(dict(zip(labels, bp_test)))
# If p < 0.05: Heteroscedasticity detected!

# ========== WHITE TEST ==========
white_test = het_white(model.resid, model.model.exog)
print("\nWhite Test:")
print(dict(zip(labels, white_test)))

# ========== SOLUTION 1: ROBUST STANDARD ERRORS ==========
# HC3 is most recommended for small samples
robust_model = model.get_robustcov_results(cov_type='HC3')
print("\n=== OLS with Robust Standard Errors (HC3) ===")
print(robust_model.summary())

# Compare SE: Original vs Robust
print("\nSE Comparison:")
print(f"Original SE: {model.bse[1]:.4f}")
print(f"Robust SE:   {robust_model.bse[1]:.4f}")
# Robust SE is usually LARGER when heteroscedasticity exists
```

### R

```r
library(lmtest)   # bptest (Breusch-Pagan)
library(car)      # ncvTest
library(sandwich) # Robust SEs
library(ggplot2)

# Sample data with heteroscedastic errors
set.seed(42)
X <- runif(100, 1, 10)
y <- 2 + 3*X + rnorm(100, 0, X/2)  # Variance grows with X
df <- data.frame(X = X, y = y)

model <- lm(y ~ X, data = df)

# ========== VISUAL DIAGNOSTIC ==========
par(mfrow = c(2, 2))
plot(model)  # Check Residuals vs Fitted (top-left)

# ========== BREUSCH-PAGAN TEST ==========
bp <- bptest(model)
print(bp)
# p < 0.05 → Heteroscedasticity detected

# ========== NCV TEST (car package) ==========
ncvTest(model)

# ========== SOLUTION 1: ROBUST STANDARD ERRORS ==========
# Using sandwich estimator
library(lmtest)
coeftest(model, vcov = vcovHC(model, type = "HC3"))

# Compare with original
summary(model)$coefficients
coeftest(model, vcov = vcovHC(model, type = "HC3"))

# ========== SOLUTION 2: WEIGHTED LEAST SQUARES ==========
# If variance is proportional to X, weight by 1/X
wls_model <- lm(y ~ X, data = df, weights = 1/X)
summary(wls_model)

# ========== SOLUTION 3: LOG TRANSFORMATION ==========
# If Y has multiplicative errors
df$log_y <- log(df$y)
log_model <- lm(log_y ~ X, data = df)
bptest(log_model)  # Check if transformation helped
```

---

## Interpretation Guide

| Test Output | Example Value | Interpretation | Edge Case/Warning |
|-------------|---------------|----------------|-------------------|
| Breusch-Pagan p-value | **0.02** | $p < 0.05$ → Reject $H_0$: heteroscedasticity detected. | Sensitive to non-normality. Use White test if residuals are non-normal. |
| Breusch-Pagan p-value | **0.35** | $p > 0.05$ → Fail to reject. No evidence of heteroscedasticity. | Low power with small n. Visual check may still show patterns. |
| White test p-value | **0.008** | More general test confirms heteroscedasticity. | Also detects specification errors (omitted non-linear terms). |
| Original SE vs Robust SE | **0.15** vs **0.28** | Robust SE is larger → original understated uncertainty. | Coefficient p-value may flip from significant to non-significant! |
| Original SE vs Robust SE | **0.15** vs **0.14** | Similar → heteroscedasticity not severely affecting inference. | Both methods give similar conclusions. |
| WLS R² | Higher than OLS | WLS fits better by down-weighting noisy observations. | Requires knowing the variance structure (hard in practice). |

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Ignoring Visual Check, Only Using Tests**
> - *Problem:* Breusch-Pagan has low power in small samples. Test says "no heteroscedasticity" but plot shows obvious funnel.
> - *Solution:* Always plot Residuals vs Fitted first. Tests supplement, not replace.
>
> **2. Using Robust SEs as Default Without Checking**
> - *Problem:* Some tutorials say "always use robust SEs to be safe."
> - *Reality:* If homoscedasticity holds, robust SEs are less efficient (wider CIs, lower power).
> - *Solution:* Test first. Use robust SEs only when needed.
>
> **3. Confusing Heteroscedasticity with Non-linearity**
> - *Problem:* Residual plot shows a curve, not a cone. Researcher applies WLS.
> - *Reality:* The issue is a missing quadratic term, not variance structure.
> - *Solution:* Check for non-linearity first (add $X^2$, use RESET test).
>
> **4. Log Transforming Without Understanding Interpretation**
> - *Problem:* Log(Y) fixes heteroscedasticity, but now coefficients mean "% change in Y," not "unit change."
> - *Solution:* Be explicit in interpretation. $\beta = 0.05$ means 5% increase in Y per unit X.

---

## Worked Numerical Example

> [!example] Detecting Heteroscedasticity in Housing Data
> **Scenario:** Modeling house prices ($Y$) from square footage ($X$). Suspicion: expensive houses have more price variability.
>
> **Step 1: Fit OLS Model**
> ```
> Ŷ = $50,000 + $150 × SquareFeet
> SE(β₁) = $12 (OLS)
> t = 150/12 = 12.5, p < 0.001 ✓
> ```
>
> **Step 2: Visual Diagnostic**
> - Residuals vs Fitted plot shows clear **funnel shape** — residuals spread from ±$20K for small houses to ±$100K for mansions.
>
> **Step 3: Breusch-Pagan Test**
> ```
> LM Statistic = 28.4
> p-value = 0.0001 < 0.05
> → Reject H₀: Heteroscedasticity confirmed!
> ```
>
> **Step 4: Apply Robust Standard Errors (HC3)**
> ```
> SE(β₁) = $25 (Robust) — nearly DOUBLE the OLS SE!
> t = 150/25 = 6.0, p < 0.001 (still significant, but less extreme)
> ```
>
> **Step 5: Alternative — Log Transformation**
> ```
> log(Ŷ) = 10.8 + 0.0005 × SquareFeet
> Residual plot: No more funnel! Breusch-Pagan p = 0.42
> Interpretation: Each additional sqft increases price by ~0.05%.
> ```
>
> **Conclusion:** Both robust SEs and log transformation are valid fixes. Robust SEs keep the original interpretation ($150/sqft). Log transform changes interpretation to percentage terms but may fit the data better.

---

## Related Concepts

**Diagnostic Tests:**
- [[30_Knowledge/Stats/03_Regression_Analysis/Breusch-Pagan Test\|Breusch-Pagan Test]] — Formal test for linear heteroscedasticity
- [[30_Knowledge/Stats/03_Regression_Analysis/White Test\|White Test]] — More general test, detects non-linear forms
- [[30_Knowledge/Stats/03_Regression_Analysis/Residual Analysis\|Residual Analysis]] — Visual diagnostics overview

**Solutions:**
- [[30_Knowledge/Stats/03_Regression_Analysis/Weighted Least Squares (WLS)\|Weighted Least Squares (WLS)]] — Explicitly model variance structure
- [[30_Knowledge/Stats/01_Foundations/Robust Standard Errors\|Robust Standard Errors]] — Keep OLS, adjust standard errors
- Log Transformations — Stabilize variance through transformation

**Related Assumptions:**
- [[30_Knowledge/Stats/03_Regression_Analysis/Simple Linear Regression\|Simple Linear Regression]] — Where homoscedasticity is assumed
- [[30_Knowledge/Stats/03_Regression_Analysis/Multiple Linear Regression\|Multiple Linear Regression]] — Same assumption applies
- [[30_Knowledge/Stats/08_Time_Series_Analysis/GARCH Models\|GARCH Models]] — Time-varying volatility in time series

---

## When to Use

> [!success] Use Heteroscedasticity When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## References

- **Historical:** Breusch, T. S., & Pagan, A. R. (1979). A simple test for heteroscedasticity and random coefficient variation. *Econometrica*, 47(5), 1287-1294. [DOI: 10.2307/1911995](https://doi.org/10.2307/1911995)
- **Historical:** White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*, 48(4), 817-838. [DOI: 10.2307/1912934](https://doi.org/10.2307/1912934)
- **Book:** Greene, W. H. (2018). *Econometric Analysis* (8th ed.). Pearson. (Chapter 9) [Pearson Link](https://www.pearson.com/en-us/subject-catalog/p/econometric-analysis/P200000003056/9780134461366)
