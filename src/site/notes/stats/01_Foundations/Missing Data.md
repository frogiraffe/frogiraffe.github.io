---
{"dg-publish":true,"permalink":"/stats/01-foundations/missing-data/","tags":["Data-Preprocessing","EDA","Data-Quality"]}
---


## Definition

> [!abstract] Core Statement
> **Missing Data** occurs when no data value is stored for a variable in an observation. Understanding the *mechanism* behind missingness determines the appropriate handling strategy. Incorrect handling can introduce **bias** or **reduce statistical power**.

**Intuition (ELI5):** Imagine a survey where some people skip the "income" question. *Why* they skipped matters: Did they skip randomly (lost page)? Did high-earners skip to hide wealth? The reason dictates whether you can safely fill in guesses or if doing so would distort your conclusions.

---

## When to Use

> [!success] Use Imputation When...
> - Missing data is **< 20-30%** of total observations.
> - Missingness is **MCAR or MAR** (not dependent on the missing value itself).
> - You want to preserve sample size and statistical power.
> - Using algorithms that **cannot handle NaN** values (e.g., Scikit-Learn).

> [!failure] Do NOT Impute When...
> - Data is **MNAR** (Missing Not At Random) — imputation will introduce systematic bias.
> - Missing data is **> 50%** for a variable — consider dropping the variable entirely.
> - The variable is the **target/outcome** — imputing Y defeats the purpose.
> - You have sufficient sample size and missingness is truly random — listwise deletion may be simpler.

---

## Theoretical Background

### Types of Missing Data (Rubin's Classification)

Understanding *why* data is missing is critical for choosing the right strategy:

| Type | Definition | Example | Safe to Impute? |
|------|------------|---------|-----------------|
| **MCAR** | Missing Completely At Random. No pattern whatsoever. | Survey page lost in mail. | ✅ Yes (any method) |
| **MAR** | Missing At Random. Probability depends on *observed* data. | Women more likely to skip "weight" question. | ✅ Yes (use predictors) |
| **MNAR** | Missing Not At Random. Probability depends on *missing value itself*. | High earners hide income; sick patients drop out. | ⚠️ Dangerous — bias likely |

### Testing for MCAR: Little's Test

$$
\chi^2 = n \cdot \sum_{j=1}^{J} \hat{p}_j (\bar{x}_j - \bar{x})' \hat{\Sigma}^{-1} (\bar{x}_j - \bar{x})
$$

- $H_0$: Data is MCAR
- If $p < 0.05$: Reject $H_0$ → Missingness has a pattern (MAR or MNAR).

---

## Assumptions & Diagnostics

Before choosing a strategy, check:

- [ ] **Proportion of Missingness:** Use `df.isnull().mean()`. Variables with >50% missing may need removal.
- [ ] **Pattern of Missingness:** Visualize with `missingno` library. Look for correlated missingness.
- [ ] **Little's MCAR Test:** Statistical test for randomness of missingness.
- [ ] **Compare Distributions:** Compare observed vs. imputed values to check for bias.

**Visual Diagnostics:**
- **Missing Data Matrix:** Should look like random salt-and-pepper if MCAR. Structured patterns suggest MAR/MNAR.
- **Heatmap of Correlations:** High correlations between missingness indicators suggest MAR.

---

## Implementation

### Python

```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
import missingno as msno
import matplotlib.pyplot as plt

# Sample data with missing values
df = pd.DataFrame({
    'Age': [25, 30, np.nan, 35, 40],
    'Income': [50000, np.nan, 60000, np.nan, 80000],
    'Education': ['Bachelor', 'Master', 'PhD', np.nan, 'Bachelor']
})

# ========== STEP 1: DIAGNOSE ==========
# Check proportion of missing values
print("Missing %:\n", df.isnull().mean() * 100)

# Visualize missing patterns (requires: pip install missingno)
msno.matrix(df)
plt.show()

# ========== STEP 2: CHOOSE STRATEGY ==========

# --- Option A: Listwise Deletion (MCAR only, <5% missing) ---
df_complete = df.dropna()

# --- Option B: Simple Imputation (MCAR/MAR, quick fix) ---
# Mean for numeric, most frequent for categorical
num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')

df['Age'] = num_imputer.fit_transform(df[['Age']])
df['Education'] = cat_imputer.fit_transform(df[['Education']])

# --- Option C: KNN Imputation (MAR, uses multivariate info) ---
# KNN finds k similar rows and averages their values
knn_imputer = KNNImputer(n_neighbors=3)
df_numeric = pd.DataFrame(
    knn_imputer.fit_transform(df[['Age', 'Income']]),
    columns=['Age', 'Income']
)

# --- Option D: MICE - Multiple Imputation (Gold Standard) ---
# pip install sklearn-pandas miceforest
# from miceforest import ImputationKernel
# kernel = ImputationKernel(df, save_all_iterations=True)
# kernel.mice(3)  # 3 iterations
# df_imputed = kernel.complete_data()
```

### R

```r
library(mice)    # Multiple imputation
library(VIM)     # Visualization
library(naniar)  # Modern missing data tools

# Sample data
df <- data.frame(
  Age = c(25, 30, NA, 35, 40),
  Income = c(50000, NA, 60000, NA, 80000),
  Education = c("Bachelor", "Master", "PhD", NA, "Bachelor")
)

# ========== STEP 1: DIAGNOSE ==========
# Summary of missing values
summary(df)
md.pattern(df)  # Missing data pattern

# Visualize
aggr(df, col = c('navyblue', 'red'),
     numbers = TRUE, sortVars = TRUE)

# Little's MCAR Test (requires BaylorEdPsych package)
# library(BaylorEdPsych)
# LittleMCAR(df)

# ========== STEP 2: IMPUTE ==========

# --- Simple Mean Imputation ---
df$Age[is.na(df$Age)] <- mean(df$Age, na.rm = TRUE)

# --- MICE: Multiple Imputation by Chained Equations ---
# Gold standard for MAR data
imp <- mice(df, m = 5, method = 'pmm', seed = 123)
# m = 5 imputed datasets
# pmm = predictive mean matching (preserves distribution)

df_imputed <- complete(imp, 1)  # Get first imputed dataset

# Pool results across imputations for regression
fit <- with(imp, lm(Income ~ Age + Education))
pooled <- pool(fit)
summary(pooled)
```

---

## Interpretation Guide

| Scenario | Strategy | Interpretation | Edge Case/Warning |
|----------|----------|----------------|-------------------|
| <5% missing, MCAR confirmed | **Listwise Deletion** | Drop rows with any NA. Simple, unbiased if truly random. | If pattern exists, you're losing non-random data → bias. |
| Numeric variable, symmetric distribution | **Mean Imputation** | Replace NA with column mean. Quick but reduces variance. | Underestimates variability. Don't use for skewed data. |
| Numeric variable, skewed distribution | **Median Imputation** | Replace NA with column median. More robust to outliers. | Still reduces variance artificially. |
| Categorical variable | **Mode Imputation** | Replace NA with most frequent category. | If distribution is uniform, mode is arbitrary. |
| MAR pattern detected | **KNN Imputation** | Use k similar observations to predict missing value. | Sensitive to scale — always standardize first! |
| MAR, need valid inference | **MICE** | Create m imputed datasets, analyze each, pool results. | Computationally expensive. Requires domain knowledge for imputation models. |
| MNAR suspected | **Sensitivity Analysis** | Compare results under different assumptions. | No statistical fix for MNAR. Report limitations. |

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Assuming MCAR Without Testing**
> - *Problem:* Researchers often assume missingness is random because it's "probably just people forgetting."
> - *Reality:* In health data, sicker patients miss follow-ups. In surveys, sensitive questions have systematic non-response.
> - *Solution:* Always run Little's MCAR test and visualize patterns.
>
> **2. Mean Imputation Destroying Variance**
> - *Problem:* Replacing 20% of values with the mean artificially shrinks the standard deviation.
> - *Result:* Confidence intervals become too narrow → false positives.
> - *Solution:* Use MICE with `m > 1` to preserve uncertainty.
>
> **3. Data Leakage in Imputation**
> - *Problem:* Fitting imputer on entire dataset (train + test), then applying to test set.
> - *Result:* Test set statistics leak into training → overoptimistic performance.
> - *Solution:* Fit imputer ONLY on training data, then `transform()` both train and test.
>
> **4. Imputing the Target Variable**
> - *Problem:* Using imputation for the outcome variable Y.
> - *Result:* You're literally making up the answers you're trying to predict.
> - *Solution:* Never impute Y. Drop rows where Y is missing.

---

## Worked Numerical Example

> [!example] Employee Salary Analysis
> **Scenario:** HR dataset with 1000 employees. Variable `Bonus` has 15% missing values.
>
> **Step 1: Diagnose Missingness**
> - Visual inspection shows employees in "Sales" department have more missing bonuses.
> - Little's MCAR test: $\chi^2 = 24.3$, $p = 0.002$ → Reject MCAR.
> - **Conclusion:** Missingness is MAR (depends on Department, an observed variable).
>
> **Step 2: Choose Strategy**
> - Since MAR: KNN or MICE imputation is appropriate.
> - Simple mean imputation would assign same bonus to all missing → ignores department differences.
>
> **Step 3: KNN Imputation (k=5)**
> ```
> Employee 47: Bonus = NA, Department = Sales, Experience = 3 years
> 5 Nearest Neighbors (same dept, similar experience):
>   - Employee 12: Bonus = $4,500
>   - Employee 23: Bonus = $5,000
>   - Employee 38: Bonus = $4,200
>   - Employee 55: Bonus = $4,800
>   - Employee 71: Bonus = $4,500
> Imputed Bonus = mean([4500, 5000, 4200, 4800, 4500]) = $4,600
> ```
>
> **Step 4: Validate**
> - Compare distribution of imputed vs. observed bonuses.
> - Imputed mean: $4,580 | Observed mean: $4,620 → Similar, good sign.
> - Imputed SD: $820 | Observed SD: $890 → Slightly lower (expected with imputation).

---

## Related Concepts

**Prerequisites:**
- [[Exploratory Data Analysis (EDA)\|Exploratory Data Analysis (EDA)]]
- [[Feature Engineering\|Feature Engineering]]

**Advanced Topics:**
- [[MICE (Multiple Imputation)\|MICE (Multiple Imputation)]]
- [[Expectation Maximization (EM)\|Expectation Maximization (EM)]]
- [[stats/01_Foundations/Selection Bias\|Selection Bias]]
- [[stats/06_Causal_Inference/Survival Analysis\|Survival Analysis]] (Censoring is a form of missingness)
