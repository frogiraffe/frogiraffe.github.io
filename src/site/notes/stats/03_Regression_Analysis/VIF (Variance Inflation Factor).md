---
{"dg-publish":true,"permalink":"/stats/03-regression-analysis/vif-variance-inflation-factor/","tags":["probability","diagnostics","regression","multicollinearity"]}
---

## Definition

> [!abstract] Core Statement
> **Variance Inflation Factor (VIF)** quantifies how much the variance of an estimated regression coefficient is ==inflated due to multicollinearity== with other predictors. It diagnoses whether predictors in [[stats/03_Regression_Analysis/Multiple Linear Regression\|Multiple Linear Regression]] are too similar to each other.

---

## Purpose

1.  Detect **multicollinearity** (high inter-correlation among predictors).
2.  Identify which predictors are redundant.
3.  Justify removing or combining correlated variables.

---

## When to Use

> [!success] Calculate VIF When...
> - Building a [[stats/03_Regression_Analysis/Multiple Linear Regression\|Multiple Linear Regression]] model with many predictors.
> - Coefficients have unexpected signs or magnitudes.
> - Standard errors are unusually large.

---

## Theoretical Background

### Formula

For predictor $X_j$:
$$
VIF_j = \frac{1}{1 - R_j^2}
$$
where $R_j^2$ is the $R^2$ from regressing $X_j$ on **all other predictors**.

### Interpretation

| VIF | Interpretation |
|-----|----------------|
| **1** | No correlation with other predictors. (Ideal). |
| **1 - 5** | Moderate correlation. (Usually acceptable). |
| **5 - 10** | High correlation. (Concerning). |
| **> 10** | Severe multicollinearity. (Action required). |

### Why Multicollinearity is Bad

1.  **Unstable Coefficients:** Small changes in data cause large swings in $\beta$.
2.  **Inflated Standard Errors:** P-values become unreliable; real effects appear insignificant.
3.  **Interpretation Ambiguity:** Which variable is "really" responsible?

---

## Assumptions

VIF is a diagnostic, not a test. It has no formal assumptions but is only meaningful in the context of OLS regression.

---

## Limitations

> [!warning] Pitfalls
> 1.  **Threshold is arbitrary:** VIF > 5 is a guideline, not a law. Context matters.
> 2.  **VIF for the intercept is meaningless.** Ignore it.
> 3.  **Structural multicollinearity:** Interaction terms (e.g., $X$ and $X^2$) naturally have high VIF; center variables first.

---

## Solutions for High VIF

1.  **Remove one variable:** If $X_1$ and $X_2$ measure the same thing, drop one.
2.  **Combine variables:** Create an index or use [[stats/05_Unsupervised_Learning/PCA (Principal Component Analysis)\|PCA (Principal Component Analysis)]].
3.  **Regularization:** Use [[stats/03_Regression_Analysis/Ridge Regression\|Ridge Regression]] or [[stats/03_Regression_Analysis/Lasso Regression\|Lasso Regression]].

---

## Python Implementation

```python
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Prepare Data (Must include constant for correct calculation)
X = add_constant(df[['Age', 'Income', 'Education']])

# Calculate VIF
vif_data = pd.DataFrame()
vif_data['Variable'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif_data[vif_data['Variable'] != 'const'])
```

---

## R Implementation

```r
library(car)

model <- lm(Y ~ Age + Income + Education, data = df)

# Calculate VIF
vif(model)

# Generalized VIF (for categorical variables with >2 levels)
# Output: GVIF^(1/(2*Df)) > 2 is concerning
```

---

## Interpretation Guide

| Output | Interpretation |
|--------|----------------|
| VIF(Income) = 1.2 | No multicollinearity issue with Income. |
| VIF(Education) = 8.5 | Education is highly correlated with other predictors. Investigate. |
| All VIF < 5 | Model is reasonably free from multicollinearity. |

---

## Related Concepts

- [[stats/03_Regression_Analysis/Multiple Linear Regression\|Multiple Linear Regression]]
- [[stats/03_Regression_Analysis/Ridge Regression\|Ridge Regression]] - Handles multicollinearity.
- [[stats/05_Unsupervised_Learning/PCA (Principal Component Analysis)\|PCA (Principal Component Analysis)]] - Dimension reduction.
- [[stats/01_Foundations/Correlation Matrix\|Correlation Matrix]]

---

## References

- **Historical:** Marquardt, D. W. (1970). Generalized inverses, ridge regression, biased linear estimation, and nonlinear estimation. *Technometrics*, 12(3), 591-612. [DOI Link](https://doi.org/10.1080/00401706.1970.10488691)
- **Book:** Kutner, M. H., Nachtsheim, C. J., Neter, J., & Li, W. (2004). *Applied Linear Statistical Models* (5th ed.). McGraw-Hill. [McGraw-Hill Link](https://www.mheducation.com/highered/product/applied-linear-statistical-models-kutner-nachtsheim/M9780073108742.html)
- **Book:** Fox, J. (2016). *Applied Regression Analysis and Generalized Linear Models* (3rd ed.). Sage. [Sage Link](https://us.sagepub.com/en-us/nam/applied-regression-analysis-and-generalized-linear-models/book237240)