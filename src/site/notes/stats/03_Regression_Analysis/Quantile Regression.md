---
{"dg-publish":true,"permalink":"/stats/03-regression-analysis/quantile-regression/","tags":["Regression","Robust  - Advanced"]}
---

## Definition

> [!abstract] Core Statement
> **Quantile Regression** extends linear regression to model the conditional **quantiles** (e.g., median, 90th percentile) of the response variable, rather than just the conditional **mean**. It provides a more complete view of possible causal relationships between variables.

---

## Purpose

1.  **Robustness (Median Regression):** The 50th percentile (Median) is robust to extreme outliers, unlike the Mean (OLS).
2.  **Heteroscedasticity Analysis:** Understand how factors affect the *spread* or *tails* of the distribution.
3.  **Risk Analysis:** Model the "Worst Case" (e.g., 99th percentile of Loss) rather than the average case.

---

## When to Use

> [!success] Use Quantile Regression When...
> - **Data has Outliers:** OLS results are distorted by extreme values.
> - **Heteroscedasticity:** Variance is not constant (OLS assumption violated).
> - **Interest in Tails:** You care about the high-performers or the at-risk group, not the average. (e.g., "What effects birth weight in *low-weight* infants?").

---

## Worked Example: Income Inequality

> [!example] Problem
> Does Education increase Income equally for everyone?
> Run OLS vs Quantile Regression ($\tau=0.1, 0.5, 0.9$).

**Results:**
-   **OLS (Mean):** Each year of education adds **\$5,000**.
-   **QR ($\tau=0.1$ - Low Earners):** Adds **\$1,000**.
-   **QR ($\tau=0.5$ - Median):** Adds **\$4,000**.
-   **QR ($\tau=0.9$ - High Earners):** Adds **\$15,000**.

**Interpretation:**
Education has a much higher payoff for high-earners (perhaps due to elite schools/networks). OLS missed this nuance by averaging everything into a single number.

---

## Theoretical Background

### Loss Function

-   **OLS** minimizes sum of **squared residuals**: $\sum (y - \hat{y})^2$.
-   **Quantile Regression** minimizes sum of **weighted absolute residuals**:
    $$ \sum \rho_\tau (y - \hat{y}) $$
    Where $\rho_\tau$ is the "Check Function" (tilted absolute value). For Median ($\tau=0.5$), this is just Mean Absolute Error (MAE).

---

## Assumptions

- [ ] **Independence:** Observations are independent.
- [ ] **Linearity:** The relationship is linear *at the specific quantile*.
- [ ] **No Homoscedasticity Required:** Unlike OLS, variance can change.

---

## Limitations

> [!warning] Pitfalls
> 1.  **Crossing Quantiles:** Sometimes lines can cross (e.g., predicting 90th percentile < 50th percentile) due to lack of constraints. This is mathematically impossible and indicates insufficient data in that region.
> 2.  **Computation:** Slower than OLS (requires Linear Programming, not simple Matrix Algebra).
> 3.  **Sample Size:** Tails ($\tau=0.99$) require very large samples to estimate stable coefficients.

---

## Python Implementation

```python
import statsmodels.formula.api as smf
import pandas as pd

# Load Data
df = pd.read_csv('salary_data.csv')

# 1. OLS (Mean)
model_ols = smf.ols('Income ~ Education', df).fit()
print(f"OLS Coeff: {model_ols.params['Education']:.2f}")

# 2. Median Regression (tau=0.5)
model_med = smf.quantreg('Income ~ Education', df).fit(q=0.5)
print(f"Median Coeff: {model_med.params['Education']:.2f}")

# 3. 90th Percentile
model_90 = smf.quantreg('Income ~ Education', df).fit(q=0.9)
print(f"90th %ile Coeff: {model_90.params['Education']:.2f}")

# Comparison gives insight into heterogeneity.
```

---

## R Implementation

```r
# install.packages("quantreg")
library(quantreg)

data(stackloss)

# Median Regression (tau = 0.5)
model_median <- rq(stack.loss ~ stack.x, tau = 0.5, data = stackloss)
summary(model_median)

# Compare with OLS
model_ols <- lm(stack.loss ~ stack.x, data = stackloss)

# Plot
plot(stackloss$stack.loss ~ stackloss$Water.Temp)
abline(rq(stack.loss ~ Water.Temp, tau = 0.5), col="blue")
abline(lm(stack.loss ~ Water.Temp), col="red", lty=2)
```

---

## Related Concepts

- [[stats/03_Regression_Analysis/Simple Linear Regression\|Simple Linear Regression]] - The baseline (Mean).
- [[stats/01_Foundations/Descriptive Statistics\|Descriptive Statistics]] - Median vs Mean.
- [[stats/03_Regression_Analysis/Heteroscedasticity\|Heteroscedasticity]] - The problem QR solves.
- [[stats/01_Foundations/Loss Function\|Loss Function]] - L1 (Absolute) vs L2 (Squared).

---

## References

- **Historical:** Koenker, R., & Bassett, G. (1978). Regression quantiles. *Econometrica*, 46(1), 33-50. [DOI: 10.2307/1913643](https://doi.org/10.2307/1913643)
- **Book:** Koenker, R. (2005). *Quantile Regression*. Cambridge University Press. [Cambridge Link](https://www.cambridge.org/9780521845731)
- **Book:** Hao, L., & Naiman, D. Q. (2007). *Quantile Regression*. Sage. [Sage Link](https://us.sagepub.com/en-us/nam/quantile-regression/book227144)
