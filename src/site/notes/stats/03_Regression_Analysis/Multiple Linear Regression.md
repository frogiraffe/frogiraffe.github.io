---
{"dg-publish":true,"permalink":"/stats/03-regression-analysis/multiple-linear-regression/","tags":["Regression","Linear-Models","OLS","Multiple-Predictors"]}
---

## Definition

> [!abstract] Core Statement
> **Multiple Linear Regression (MLR)** extends simple linear regression to model the relationship between a single ==continuous dependent variable ($Y$)== and ==two or more independent variables ($X_1, X_2, \dots, X_k$)==. It determines the unique contribution of each predictor while **controlling for** the others.

---

## Purpose

1.  **Isolate Effects:** Understand the effect of $X_1$ on $Y$, *holding $X_2, X_3, \dots$ constant* (Ceteris Paribus).
2.  **Control for Confounders:** Reduce bias by including variables that might otherwise distort the relationship.
3.  **Build Predictive Models:** Create more accurate predictions than SLR.

---

## When to Use

> [!success] Use MLR When...
> - You have **multiple continuous predictors** for one continuous outcome.
> - You want to **control for confounders**.
> - You believe the relationship between each predictor and the outcome is **linear**.

> [!failure] Do NOT Use MLR When...
> - The outcome is categorical (use [[stats/03_Regression_Analysis/Logistic Regression\|Logistic Regression]]).
> - Predictors are highly correlated (Multicollinearity problem).

---

## Theoretical Background

### The Model Equation

$$
Y_i = \beta_0 + \beta_1 X_{1i} + \beta_2 X_{2i} + \dots + \beta_k X_{ki} + \varepsilon_i
$$

**Interpretation of $\beta_j$:**
$\beta_j$ represents the expected change in $Y$ for a ==one-unit increase in $X_j$, holding all other predictors constant==.

### Adjusted $R^2$

> [!important] Always use Adjusted $R^2$
> Standard $R^2$ **always increases** when you add variables, even useless ones. **Adjusted $R^2$** penalizes for adding variables that don't improve the model.
> $$
> R^2_{adj} = 1 - \frac{(1 - R^2)(n - 1)}{n - k - 1}
> $$
> Use Adjusted $R^2$ for model comparison.

### The F-Test (Overall Model Significance)

Tests whether the model as a whole explains significantly more variance than a model with just the intercept.
- $H_0$: $\beta_1 = \beta_2 = \dots = \beta_k = 0$ (Model has no explanatory power).
- If $p < 0.05$: Reject $H_0$. The model is useful.

---

## Assumptions (LINE + No Multicollinearity)

All assumptions from [[stats/03_Regression_Analysis/Simple Linear Regression\|Simple Linear Regression]] apply, plus one critical addition:

- [ ] **Linearity**
- [ ] **Independence**
- [ ] **Normality of Residuals**
- [ ] **Equality of Variance (Homoscedasticity)**
- [ ] **No Multicollinearity:** Predictors ($X$s) should not be highly correlated with each other. (Check: [[stats/03_Regression_Analysis/VIF (Variance Inflation Factor)\|VIF (Variance Inflation Factor)]]).

---

## Limitations

> [!warning] Pitfalls
> 1.  **Multicollinearity:** If $X_1$ and $X_2$ are highly correlated (e.g., $r > 0.8$), the model cannot determine which one is responsible for the effect. Standard errors inflate, p-values become unreliable.
> 2.  **Overfitting:** Adding too many predictors can make the model fit noise rather than signal. Use [[stats/03_Regression_Analysis/Ridge Regression\|Ridge Regression]] or [[stats/03_Regression_Analysis/Lasso Regression\|Lasso Regression]] for regularization.
> 3.  **Specification Errors:** Including irrelevant variables or omitting relevant ones biases estimates.

---

## Python Implementation

```python
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 1. Prepare Data
X = df[['Age', 'Experience', 'Education']]
y = df['Salary']
X = sm.add_constant(X)

# 2. Check VIF Before Fitting
vif = pd.DataFrame()
vif['Variable'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("VIF:\n", vif[vif['Variable'] != 'const'])
# Rule: VIF > 5 is concerning; VIF > 10 is severe.

# 3. Fit Model
model = sm.OLS(y, X).fit()
print(model.summary())
```

---

## R Implementation

```r
# 1. Fit Model
model <- lm(Salary ~ Age + Experience + Education, data = df)

# 2. Results
summary(model)

# 3. Check VIF
library(car)
vif(model)
# GVIF^(1/(2*Df)) > 2 is concerning

# 4. Confidence Intervals
confint(model)

# 5. Compare Models (ANOVA)
model_reduced <- lm(Salary ~ Age, data = df)
anova(model_reduced, model)
# Tests if additional variables significantly improve fit
```

---

## Worked Numerical Example

> [!example] Salary Prediction Model
> **Data:** 100 employees
> **Model:** Salary = β₀ + β₁(Age) + β₂(Experience) + β₃(Education)
>
> **Results:**
> - β₀ (Intercept) = 20,000
> - β₁ (Age) = 500, p < 0.05
> - β₂ (Experience) = 1,200, p < 0.001
> - β₃ (Education) = 3,000, p < 0.01
> - Adjusted R² = 0.78
> - F-statistic: p < 0.001
>
> **Interpretation:**
> - For a 30-year-old with 5 years experience and bachelor's degree (Education=4):
>   Predicted Salary = 20,000 + 500(30) + 1,200(5) + 3,000(4) = $53,000
> - If experience increases to 6 years (holding age and education constant):
>   New Salary = $53,000 + $1,200 = $54,200

---

## Interpretation Guide

| Output | Example Value | Interpretation | Edge Case Notes |
|--------|---------------|----------------|------------------|
| $\beta_{Age}$ | 500 | Each additional year increases Salary by $500, **holding Experience and Education constant**. | If Age and Experience are correlated (r=0.9), this estimate is unstable. Check VIF. |
| $\beta_{Age}$ | -200 | Counterintuitive negative sign suggests **Simpson's Paradox** or multicollinearity. | Investigate: Age may proxy for obsolete skills when controlling for Education. |
| VIF for Experience | 8.5 | High multicollinearity. Standard errors inflated. | Consider: Remove Experience, or combine Age+Experience into "Career Length". |
| VIF for Experience | 1.2 | No multicollinearity concern. | Coefficient estimate is reliable. |
| Adjusted $R^2$ | 0.78 | 78% of variance explained (penalized for # predictors). | Compare to R² (0.82): penalty is small, model complexity justified. |
| Adjusted $R^2$ | 0.15 | Model explains little variance even after penalty. | Predictors may be irrelevant, or relationship is non-linear. |
| F-statistic p < 0.001 | Model is significant overall. | At least one predictor has non-zero effect. | Individual p-values may still be >0.05 due to multicollinearity. |
| F-statistic p = 0.30 | Model has no explanatory power. | All predictors together don't predict Y better than intercept-only model. |

---

## Common Pitfall Example

> [!warning] Real-World Trap: Income and Years of Education
> **Scenario:** Regressing Income on Years_of_Education and Years_of_Experience.
>
> **Problem:** Education and Experience are highly correlated (r = 0.85) because:
> - High education → late career start → less experience
> - Low education → early career start → more experience
>
> **Result:**
> - VIF(Education) = 7.2, VIF(Experience) = 7.2
> - β_Education = $2,000, p = 0.08 (not significant!)
> - β_Experience = $1,500, p = 0.12 (not significant!)
>
> **But:** When you remove one variable:
> - Model with only Education: β = $4,000, p < 0.001
> - Model with only Experience: β = $3,500, p < 0.001
>
> **Lesson:** Both variables ARE important, but multicollinearity makes them appear non-significant when included together. Use Ridge regression or create composite variable (e.g., "Career Investment Score").

---

## Related Concepts

- [[stats/03_Regression_Analysis/Simple Linear Regression\|Simple Linear Regression]] - The single-predictor case.
- [[stats/03_Regression_Analysis/VIF (Variance Inflation Factor)\|VIF (Variance Inflation Factor)]] - Multicollinearity diagnostic.
- [[stats/03_Regression_Analysis/Ridge Regression\|Ridge Regression]] - Regularization to handle multicollinearity.
- [[stats/03_Regression_Analysis/Lasso Regression\|Lasso Regression]] - Feature selection via L1 penalty.
- [[stats/01_Foundations/Adjusted R-squared\|Adjusted R-squared]]

---

## References

- **Book:** Draper, N. R., & Smith, H. (1998). *Applied Regression Analysis* (3rd ed.). Wiley. [Wiley Link](https://www.wiley.com/en-us/Applied+Regression+Analysis%2C+3rd+Edition-p-9780471170822)
- **Book:** James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.). Springer. [Springer Link](https://link.springer.com/book/10.1007/978-1-0716-1418-1)
- **Book:** Faraway, J. J. (2015). *Linear Models with R* (2nd ed.). CRC Press. [CRC Press Link](https://www.routledge.com/Linear-Models-with-R/Faraway/p/book/9781439887332)
