---
{"dg-publish":true,"permalink":"/stats/03-regression-analysis/binary-logistic-regression/","tags":["Regression","Classification","GLM","Binary-Outcome"]}
---

## Definition

> [!abstract] Core Statement
> **Binary Logistic Regression** is the standard model for a ==binary (dichotomous) outcome variable== (e.g., Yes/No, Success/Failure, 0/1). It estimates the probability $P(Y=1 | X)$ using the logistic (sigmoid) function.

---

> [!tip] Intuition (ELI5): The Rain Chance
> You want to know: **"Will it rain today?"** (Yes or No). Regular regression predicts a number (like 5 inches). Logistic Regression predicts a **Probability** (like an 80% chance). It takes info like "cloudiness" and squashes it between 0 and 1. If it's $>0.5$, the model says "Yes!"

> [!example] Real-Life Example: Spam Filters
> Email filters look for features like "unknown sender" or "free money." The model calculates the probability of **Spam**. If that probability is $>0.99$, it moves the email to the Junk folder automatically.

---

## Purpose

1.  **Predict** the probability of a binary event.
2.  **Understand** which factors increase or decrease the likelihood of the event via Odds Ratios.
3.  **Classify** observations into one of two groups based on a probability threshold.

---

## When to Use

> [!success] Use Binary Logistic Regression When...
> - Outcome is **binary** (two mutually exclusive categories).
> - Predictors can be continuous, categorical, or a mix.
> - You need interpretable effects in terms of **Odds Ratios**.

> [!failure] Do NOT Use When...
> - Outcome has more than 2 categories (use [[stats/03_Regression_Analysis/Multinomial Logistic Regression (MNLogit)\|Multinomial Logistic Regression (MNLogit)]] or [[stats/03_Regression_Analysis/Ordinal Logistic Regression\|Ordinal Logistic Regression]]).
> - Outcome is a count (use [[stats/03_Regression_Analysis/Poisson Regression\|Poisson Regression]]).

---

## Theoretical Background

### The Model

$$
P(Y=1 | X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \dots)}} = \frac{e^{\beta_0 + \beta_1 X_1 + \dots}}{1 + e^{\beta_0 + \beta_1 X_1 + \dots}}
$$

This is the **Sigmoid Function**, which squashes the linear predictor into the range $[0, 1]$.

### Logit Form (Link Function)

$$
\text{logit}(P) = \ln\left(\frac{P}{1-P}\right) = \beta_0 + \beta_1 X_1 + \dots
$$

### Coefficients and Odds Ratios

| Term | Meaning |
|------|---------|
| $\beta_j$ | Change in **log-odds** for a 1-unit increase in $X_j$ |
| $e^{\beta_j}$ | **Odds Ratio (OR)**: Multiplicative change in odds for a 1-unit increase in $X_j$ |

**Example:**
If $\beta_{age} = 0.05$, then $OR = e^{0.05} \approx 1.05$.
*"For every additional year of age, the odds of the event increase by 5%."*

---

## Assumptions

- [ ] **Binary Outcome:** Dependent variable must be dichotomous.
- [ ] **Independence of Observations:** Data points should not be clustered or dependent.
- [ ] **Linearity of Logit:** The relationship between continuous predictors and the **log-odds** of the outcome must be linear. (Test with Box-Tidwell).
- [ ] **No Multicollinearity:** Predictors should not be highly correlated.
- [ ] **Large Sample Size:** MLE requires sufficient events. Rule of thumb: **10-20 events per predictor**.

> [!warning] Events Per Variable (EPV)
> If you have only 50 events (Y=1) and 10 predictors, you have EPV = 5, which is dangerously low. Coefficients will be unstable, and p-values unreliable. Reduce predictors or gather more data.

---

## Limitations

> [!warning] Pitfalls
> 1.  **Odds Ratio $\neq$ Relative Risk:** OR overestimates Relative Risk when the outcome is common (>10%). For rare diseases, OR $\approx$ RR.
> 2.  **Perfect Separation:** If a predictor perfectly predicts the outcome (e.g., all Y=1 when X>5), MLE fails (coefficients go to infinity).
> 3.  **Threshold Selection:** Classification accuracy depends on the chosen probability threshold (default 0.5), which may not be optimal to maximize AUC or balance classes.

---

## Model Evaluation

Unlike OLS ($R^2$), Logistic Regression uses different metrics:
| Metric | Purpose |
|--------|---------|
| **Pseudo $R^2$ (McFadden's)** | Explained variance analog. 0.2-0.4 is excellent. |
| **[[stats/04_Supervised_Learning/ROC-AUC\|ROC-AUC]]** | Discrimination ability. Area under curve; 0.7-0.8 acceptable, >0.8 excellent. |

---

## Related Concepts

- [[stats/01_Foundations/Odds Ratio\|Odds Ratio]] - Key interpretation metric.
- [[stats/04_Supervised_Learning/Confusion Matrix\|Confusion Matrix]] - For classification performance.
- [[stats/03_Regression_Analysis/Generalized Linear Models (GLM)\|Generalized Linear Models (GLM)]] - The parent framework.
- [[stats/03_Regression_Analysis/Multinomial Logistic Regression (MNLogit)\|Multinomial Logistic Regression (MNLogit)]] - For >2 categories.

---


---

## Python Implementation

```python
import statsmodels.api as sm
import numpy as np
import pandas as pd

# 1. Fit Model
X = sm.add_constant(df[['Age', 'Income']])
y = df['Purchased']

model = sm.Logit(y, X).fit()
print(model.summary())

# 2. Odds Ratios with Confidence Intervals
params = model.params
conf = model.conf_int()
conf['OR'] = params
conf.columns = ['2.5%', '97.5%', 'Log-Odds']
conf['Odds Ratio'] = np.exp(conf['Log-Odds'])
conf['OR Lower'] = np.exp(conf['2.5%'])
conf['OR Upper'] = np.exp(conf['97.5%'])
print(conf[['Odds Ratio', 'OR Lower', 'OR Upper']])

# 3. Predict Probabilities
df['pred_prob'] = model.predict(X)
```

---

## R Implementation

```r
# 1. Fit Model (GLM with Binomial Family)
model <- glm(Purchased ~ Age + Income, data = df, family = "binomial")
summary(model)

# 2. Odds Ratios with CI
exp(cbind(OR = coef(model), confint(model)))

# 3. Predict Probabilities
df$pred_prob <- predict(model, type = "response")

# 4. Hosmer-Lemeshow Test
library(ResourceSelection)
hoslem.test(df$Purchased, fitted(model), g = 10)
```

---

## Worked Numerical Example

> [!example] Heart Disease Prediction
> **Scenario:** Predict 10-year risk of heart disease (1=Yes, 0=No).
> **Predictor:** Smoker (1=Yes, 0=No)
> 
> **Results:**
> - $\beta_{smoker}$ = 0.85
> - $p$-value < 0.001
> 
> **Calculations:**
> - **Log-Odds:** Being a smoker increases the log-odds of heart disease by 0.85.
> - **Odds Ratio (OR):** $e^{0.85} \approx 2.34$.
>   - *"Smokers have 2.34 times higher odds of heart disease compared to non-smokers."*
> 
> **Probability Prediction (Intercept $\beta_0 = -2.0$):**
> - **Non-Smoker ($X=0$):** $P = 1 / (1 + e^{-(-2.0)}) \approx 0.12$ (12% risk)
> - **Smoker ($X=1$):** $z = -2.0 + 0.85 = -1.15$
>   - $P = 1 / (1 + e^{-(-1.15)}) \approx 0.24$ (24% risk)
> 
> **Key Insight:** OR stays constant (2.34), but absolute risk depends on the baseline.

---

## Interpretation Guide

| Output | Example | Interpretation | Edge Case Notes |
|--------|---------|----------------|-----------------|
| Coef (Age) | 0.03 | Each additional year increases log-odds by 0.03. | Positive $\beta$ $\to$ OR > 1. |
| OR (Age) | 1.03 | Each additional year increases odds of event by 3%. | If OR contains 1 in CI, not significant. |
| OR | 0.50 | Exposure **halves** the odds of the event (Protection). | $0.50$ is equivalent to $2.0$ in opposite direction ($1/0.5 = 2$). |
| OR | 25.0 | Extremely large effect or **Perfect Separation**. | Check if predictor perfectly predicts outcome (e.g., all X>10 have Y=1). |
| Predicted Prob | 0.72 | Model predicts 72% chance of event. | If using threshold 0.5, classify as Y=1. |

---

## Common Pitfall Example

> [!warning] The Relative Risk Trap
> **Mistake:** Interpreting Odds Ratios (OR) as Relative Risk (RR).
> 
> **Scenario:**
> - Study on a common outcome (e.g., "Recovery", rate = 50%).
> - Treatment OR = 2.0.
> 
> **Incorrect Interpretation:** "Treatment doubles the probability of recovery." (Implies Risk Ratio = 2.0).
> 
> **Reality:**
> - If Baseline Risk = 50% (Odds = 1:1 = 1.0).
> - Treatment Odds = $1.0 \times 2.0 = 2.0$ (Risk $\approx$ 67%).
> - **Actual Risk Ratio:** $67\% / 50\% = 1.34$.
> - The OR (2.0) massive exaggerates the RR (1.34) because outcome is common.
> 
> **Rule:** OR $\approx$ RR only when the outcome is **rare** (<10%). Otherwise, report predicted probabilities.

---

## Related Concepts

- [[stats/03_Regression_Analysis/Logistic Regression\|Logistic Regression]] - Family overview.
- [[stats/01_Foundations/Maximum Likelihood Estimation (MLE)\|Maximum Likelihood Estimation (MLE)]]
- [[stats/04_Supervised_Learning/ROC-AUC\|ROC-AUC]] - Performance metric.
- [[stats/04_Supervised_Learning/Confusion Matrix\|Confusion Matrix]] - Classification metrics.
- [[stats/03_Regression_Analysis/Probit Regression\|Probit Regression]] - Alternative using normal CDF.

---

## References

- **Book:** Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression* (3rd ed.). Wiley. [Wiley Link](https://www.wiley.com/en-us/Applied+Logistic+Regression%2C+3rd+Edition-p-9780470582473)
- **Book:** Agresti, A. (2013). *Categorical Data Analysis* (3rd ed.). Wiley. [Wiley Link](https://www.wiley.com/en-us/Categorical+Data+Analysis%2C+3rd+Edition-p-9780470463635)
- **Book:** Menard, S. (2002). *Applied Logistic Regression Analysis* (2nd ed.). Sage. [Sage Link](https://doi.org/10.4135/9781412983433)
- **Book:** James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.). Springer. [Springer Link](https://link.springer.com/book/10.1007/978-1-0716-1418-1)
- **Article:** Cox, D. R. (1958). The regression analysis of binary sequences. *Journal of the Royal Statistical Society B*, 20(2), 215-242. [DOI Link](https://doi.org/10.1093/biomet/45.1-2.215)