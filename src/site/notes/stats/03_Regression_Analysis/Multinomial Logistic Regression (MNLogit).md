---
{"dg-publish":true,"permalink":"/stats/03-regression-analysis/multinomial-logistic-regression-mn-logit/","tags":["Regression","GLM","Classification","Multiclass"]}
---

## Definition

> [!abstract] Core Statement
> **Multinomial Logistic Regression (MNLogit)** extends [[stats/03_Regression_Analysis/Binary Logistic Regression\|Binary Logistic Regression]] to handle outcomes with ==more than two unordered categories== (e.g., Transportation Mode: Car, Bus, Bike). It models the probability of each category relative to a **reference category**.

---

## Purpose

1.  Model nominal (unordered) multi-class outcomes.
2.  Understand which factors predict category membership.
3.  Calculate predicted probabilities for each class.

---

## When to Use

> [!success] Use MNLogit When...
> - Outcome has **3+ unordered categories**.
> - Predictors can be continuous, categorical, or mixed.
> - Categories are mutually exclusive and exhaustive.

> [!failure] Do NOT Use When...
> - Outcome is **ordinal** (use Ordinal Logistic Regression).
> - Outcome is binary (use [[stats/03_Regression_Analysis/Binary Logistic Regression\|Binary Logistic Regression]]).

---

## Theoretical Background

### The Model

For $J$ categories, MNLogit estimates $J-1$ sets of coefficients (one category is the **reference**).
$$
\ln\left(\frac{P(Y=j)}{P(Y=\text{ref})}\right) = \beta_{j0} + \beta_{j1} X_1 + \dots
$$

### Relative Risk Ratio (RRR)

$$
RRR = e^{\beta_j}
$$
*"A 1-unit increase in $X$ multiplies the odds of being in category $j$ versus the reference by $RRR$."*

### Independence of Irrelevant Alternatives (IIA)

> [!warning] Critical Assumption
> MNLogit assumes that the odds of choosing between any two categories are independent of other categories. (e.g., Preference for Car vs Bus is unaffected by adding a new "Train" option).
> If violated, use **Nested Logit** or **Mixed Logit**.

---

## Assumptions

- [ ] **Nominal Outcome:** Categories are unordered.
- [ ] **Independence of Observations.**
- [ ] **IIA:** Independence of Irrelevant Alternatives.
- [ ] **Linearity** between log-odds and predictors.
- [ ] **Sufficient Sample Size:** Each category needs enough observations.

---

## Limitations

> [!warning] Pitfalls
> 1.  **IIA Violation:** If adding a new category changes the relative odds of existing categories, the model is misspecified.
> 2.  **Complexity:** Interpretation requires $J-1$ sets of coefficients.
> 3.  **Sparse Categories:** Rare categories can cause estimation problems.

---

## Python Implementation

```python
import statsmodels.api as sm

# Fit MNLogit
X = sm.add_constant(df[['income', 'distance']])
y = df['transport_mode']  # Categories: 'car', 'bus', 'bike'

model = sm.MNLogit(y, X).fit()
print(model.summary())

# Relative Risk Ratios
import numpy as np
print("\n--- Relative Risk Ratios ---")
print(np.exp(model.params))
```

---

## R Implementation

```r
library(nnet)

# Fit MNLogit (multinom function)
# Relevel to set reference category
df$transport_mode <- relevel(factor(df$transport_mode), ref = "car")

model <- multinom(transport_mode ~ income + distance, data = df)
summary(model)

# Relative Risk Ratios
exp(coef(model))

# Confidence Intervals (z-test)
z <- summary(model)$coefficients / summary(model)$standard.errors
p <- (1 - pnorm(abs(z), 0, 1)) * 2
print(p)
```

---

## Worked Numerical Example

> [!example] Transportation Choice Model
> **Outcome:** How do students commute? (Walk, Bike, Car)
> **Predictors:** Distance_to_School (miles), Income ($1000s)
> **Base Category:** Walk
> 
> **Results:**
> 
> | Outcome | Predictor | Coefficient | RRR | Interpretation |
> |---------|-----------|-------------|-----|----------------|
> | **Walk** (base) | - | 0 | 1.0 | Reference category |
> | Bike vs Walk | Distance | 0.15 | 1.16 | Each mile → 16% more likely to bike than walk |
> | Bike vs Walk | Income | -0.02 | 0.98 | Higher income → slightly less likely to bike |
> | Car vs Walk | Distance | 0.40 | 1.49 | Each mile → 49% more likely to drive than walk |
> | Car vs Walk | Income | 0.08 | 1.08 | Each $1K income → 8% more likely to drive |
> 
> **Predictions for a student 5 miles away, income $40K:**
> - P(Walk) = 0.15
> - P(Bike) = 0.25  
> - P(Car) = 0.60
> - Most likely choice: Car

---

## Interpretation Guide

| Output | Interpretation | Edge Case Notes |
|--------|----------------|------------------|
| RRR (Car vs Walk) = 2.5 | Car 2.5× more likely than Walk per unit increase. | Relative to base only! Doesn't tell you Car vs Bike. |
| RRR = 1.0 | No effect on relative odds between categories. | Coefficient ≈ 0 (exactly 0 if no effect). |
| RRR = 0.5 | Outcome 50% **less** likely than base. | RRR < 1 means negative association. |
| All RRRs > 1 for one predictor | Predictor increases all non-base categories. | Common for variables like "service quality" in choice models. |
| Base category has P = 0.90 | Model may be unstable (rare outcomes). | Consider combining rare categories or multinomial ordered logit. |

---

## Common Pitfall Example

> [!warning] IIA Violation: The Red Bus / Blue Bus Problem
> **Classic Example:**
> - Initial choices: Car (70%), Bus (30%)
> - MNLogit assumes: P(Car)/P(Bus) = constant regardless of other options
> 
> **IIA Assumption:** If you add a "Red Bus" option:
> - IIA predicts: Car (70%), Blue Bus (15%), Red Bus (15%)
> - Reality: Car (70%), Blue Bus (20%), Red Bus (10%)
> - **Why?** Red/Blue buses are close substitutes, violate independence!
> 
> **Real Scenario:**
> - Brand choice: Coke, Pepsi, Store Brand
> - Add "Diet Coke" → should draw more from Coke than Pepsi
> - But MNLogit assumes all alternatives equally affected
> 
> **Test IIA:**
> - Hausman test: Fit model with all options, then drop one
> - If coefficients change substantially → IIA violated
> 
> **Solutions if IIA violated:**
> 1. Nested Logit (group similar alternatives)
> 2. Mixed Logit (random coefficients)
> 3. Multinomial Probit (no IIA assumption)
> 
> **When IIA is OK:**
> - Alternatives are truly distinct (Car, Bike, Walk)
> - No obvious substitution patterns

---

## Interpretation Guide

| Comparison | Coef | RRR | Interpretation |
|------------|------|-----|----------------|
| Bus vs Car (Income) | -0.5 | 0.61 | Higher income decreases odds of choosing Bus over Car by 39%. |
| Bike vs Car (Distance) | -1.2 | 0.30 | Longer distance decreases odds of Bike over Car by 70%. |

---

## Related Concepts

- [[stats/03_Regression_Analysis/Binary Logistic Regression\|Binary Logistic Regression]]
- [[stats/03_Regression_Analysis/Ordinal Logistic Regression\|Ordinal Logistic Regression]] - For ordered outcomes.
- [[stats/03_Regression_Analysis/Generalized Linear Models (GLM)\|Generalized Linear Models (GLM)]]

---

## References

- **Book:** Agresti, A. (2019). *An Introduction to Categorical Data Analysis* (3rd ed.). Wiley. [Wiley Link](https://www.wiley.com/en-us/An+Introduction+to+Categorical+Data+Analysis%2C+3rd+Edition-p-9781119405269)
- **Book:** Long, J. S., & Freese, J. (2014). *Regression Models for Categorical Dependent Variables Using Stata* (3rd ed.). Stata Press. [Stata Press Link](https://www.stata.com/bookstore/regression-models-categorical-dependent-variables/)
- **Book:** McFadden, D. (1974). Conditional logit analysis of qualitative choice behavior. *Frontiers in Econometrics*. [Stable Link](https://ideas.repec.org/h/nbe/nbechp/5341.html)