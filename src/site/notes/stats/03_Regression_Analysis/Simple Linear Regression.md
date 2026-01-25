---
{"dg-publish":true,"permalink":"/stats/03-regression-analysis/simple-linear-regression/","tags":["Regression","Linear-Models","OLS","Prediction"]}
---

## Definition

> [!abstract] Core Statement
> **Simple Linear Regression (SLR)** models the linear relationship between a single ==continuous independent variable ($X$)== and a ==continuous dependent variable ($Y$)==. It fits a straight line that minimizes the sum of squared residuals (Ordinary Least Squares - OLS).

---

## Purpose

1.  **Prediction:** Estimate $Y$ for a given $X$.
2.  **Explanation:** Quantify the strength and direction of the association between $X$ and $Y$.
3.  **Testing:** Determine if the relationship is statistically significant ($\beta_1 \neq 0$).

---

## When to Use

> [!success] Use SLR When...
> - You have **one continuous predictor** and **one continuous outcome**.
> - You believe the relationship is **linear**.
> - You want to understand/test the bivariate association.

> [!failure] Do NOT Use SLR When...
> - You have multiple predictors (use [[stats/03_Regression_Analysis/Multiple Linear Regression\|Multiple Linear Regression]]).
> - The outcome is categorical (use [[stats/03_Regression_Analysis/Logistic Regression\|Logistic Regression]]).
> - The relationship is clearly non-linear (consider polynomial or GAM).

---

## Theoretical Background

### The Model Equation

$$
Y_i = \beta_0 + \beta_1 X_i + \varepsilon_i
$$

| Term | Name | Interpretation |
|------|------|----------------|
| $Y_i$ | Dependent Variable | The outcome we are predicting. |
| $X_i$ | Independent Variable | The predictor. |
| $\beta_0$ | Intercept | Expected value of $Y$ when $X = 0$. |
| $\beta_1$ | Slope | ==Change in $Y$ for a 1-unit increase in $X$.== |
| $\varepsilon_i$ | Error Term | Random noise; captures unexplained variation. |

### OLS Estimation

The OLS method finds $\beta_0$ and $\beta_1$ that minimize the Residual Sum of Squares (RSS):
$$
\text{RSS} = \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2 = \sum_{i=1}^{n} (Y_i - \beta_0 - \beta_1 X_i)^2
$$

**Closed-Form Solutions:**
$$
\hat{\beta}_1 = \frac{\sum(X_i - \bar{X})(Y_i - \bar{Y})}{\sum(X_i - \bar{X})^2} = \frac{Cov(X, Y)}{Var(X)}
$$
$$
\hat{\beta}_0 = \bar{Y} - \hat{\beta}_1 \bar{X}
$$

### Coefficient of Determination ($R^2$)

$$
R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = \frac{\text{Variance Explained}}{\text{Total Variance}}
$$

**Interpretation:** The proportion of variance in $Y$ explained by $X$.
- $R^2 = 0.70$: 70% of the variability in $Y$ is explained by $X$.

---

## Assumptions (LINE)

> [!important] The LINE Mnemonic
> - [ ] **L**inearity: The relationship between $X$ and $Y$ is linear. (Check: Scatter plot, Residual vs Fitted plot).
> - [ ] **I**ndependence: Observations are independent. (Check: [[stats/05_Time_Series/Durbin-Watson Test\|Durbin-Watson Test]] for time series).
> - [ ] **N**ormality: ==Residuals== (not $Y$!) are normally distributed. (Check: [[stats/08_Visualization/Q-Q Plot\|Q-Q Plot]], [[stats/02_Hypothesis_Testing/Shapiro-Wilk Test\|Shapiro-Wilk Test]]).
> - [ ] **E**qual Variance (Homoscedasticity): The variance of residuals is constant across all $X$. (Check: [[stats/03_Regression_Analysis/Breusch-Pagan Test\|Breusch-Pagan Test]], Residual vs Fitted plot).

---

## Limitations

> [!warning] Pitfalls
> 1.  **Correlation $\neq$ Causation:** SLR identifies association, not causation. Confounders may exist.
> 2.  **Sensitive to Outliers:** Extreme points can disproportionately influence the slope and $R^2$. Check [[stats/03_Regression_Analysis/Cook's Distance\|Cook's Distance]].
> 3.  **Extrapolation is Dangerous:** Predicting $Y$ for $X$ values outside the observed range is unreliable.
> 4.  **Omitted Variable Bias:** If a relevant variable is missing, $\beta_1$ may be biased.

---

## Python Implementation

```python
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt

# 1. Prepare Data
X = df['Ad_Spend']
y = df['Sales']
X = sm.add_constant(X)  # Add intercept

# 2. Fit Model
model = sm.OLS(y, X).fit()

# 3. Results
print(model.summary())

# 4. Visualization
plt.scatter(df['Ad_Spend'], df['Sales'], alpha=0.6)
plt.plot(df['Ad_Spend'], model.predict(X), color='red', linewidth=2)
plt.xlabel("Ad Spend ($)")
plt.ylabel("Sales ($)")
plt.title(f"SLR: R-squared = {model.rsquared:.3f}")
plt.show()

# 5. Diagnostics
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sm.graphics.plot_regress_exog(model, 'Ad_Spend', fig=fig)
plt.show()
```

---

## R Implementation

```r
# 1. Fit Model
model <- lm(Sales ~ Ad_Spend, data = df)

# 2. Results
summary(model)

# 3. Confidence Intervals for Coefficients
confint(model)

# 4. Diagnostics (Standard R Plots)
par(mfrow = c(2, 2))
plot(model)
# Plot 1: Residuals vs Fitted (Linearity, Homoscedasticity)
# Plot 2: Normal Q-Q (Normality of Residuals)
# Plot 3: Scale-Location (Homoscedasticity)
# Plot 4: Residuals vs Leverage (Influential Points)

# 5. Prediction
new_data <- data.frame(Ad_Spend = c(100, 200, 300))
predict(model, newdata = new_data, interval = "confidence")
```

---

## Worked Numerical Example

> [!example] predicting Sales from Ad Spend
> **Data:** 10 months of data.
> **Result:** Sales = 1000 + 2.5(Ad_Spend)
> 
> **Interpretation:**
> - **Intercept (1000):** If Ad Spend is \$0, we expect \$1000 in baseline sales (brand awareness, organic).
> - **Slope (2.5):** For every additional \$1 spent on ads, Sales increase by \$2.50 on average.
> 
> **Prediction:**
> - If Ad Spend = \$500:
> - Sales = 1000 + 2.5(500) = 1000 + 1250 = \$2250.

---

## Interpretation Guide

| Output | Example Value | Interpretation | Edge Case Notes |
|--------|---------------|----------------|-----------------|
| $\beta_1$ (Slope) | 2.5 | Positive association. Y increases with X. | If 0, no linear relationship. |
| $\beta_0$ (Intercept) | -50 | Value of Y when X=0. | Negative sales? **Impossible.** Intercept may have no physical meaning if X=0 is outside data range. |
| $p$-value for $\beta_1$ | 0.001 | Slope $\neq$ 0. Relationship is significant. | Does not mean relationship is *strong*, just *reliable*. |
| **R-squared** | 0.65 | 65% of variance in Sales is explained. | 35% is noise/error. |
| **R-squared** | 0.01 | X explains almost nothing about Y. | Check for non-linear U-shape! |

---

## Common Pitfall Example

> [!warning] Extrapolation Danger
> **Scenario:** Modeling child height vs age (Age 2-10).
> **Model:** Height = 80cm + 6cm(Age). (Fits well for data).
> 
> **Prediction Error:**
> - Predict height for a 30-year-old:
> - H = 80 + 6(30) = 260cm (8.5 feet!).
> 
> **Lesson:** Linear trends rarely continue forever. **Never** extrapolate far outside the range of your training data.

---

## Related Concepts

- [[stats/03_Regression_Analysis/Multiple Linear Regression\|Multiple Linear Regression]] - More than one predictor.
- [[stats/02_Hypothesis_Testing/Pearson Correlation\|Pearson Correlation]] - Related but different (Correlation $\neq$ Slope).
- [[stats/03_Regression_Analysis/Residual Analysis\|Residual Analysis]] - Checking assumptions.
- [[stats/03_Regression_Analysis/Cook's Distance\|Cook's Distance]] - Identifying influential observations.