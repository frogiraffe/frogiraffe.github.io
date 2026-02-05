---
{"dg-publish":true,"permalink":"/stats/03-regression-analysis/ordinal-logistic-regression/","tags":["probability","regression","glm","ordinal-outcome"]}
---


## Definition

> [!abstract] Core Statement
> **Ordinal Logistic Regression** is a regression model for ==ordinal outcomes==—categorical variables with a natural ordering but unknown distances between categories (e.g., "Poor < Fair < Good < Excellent"). It extends [[stats/03_Regression_Analysis/Binary Logistic Regression\|Binary Logistic Regression]] using cumulative probabilities and the **proportional odds assumption**.

**Intuition (ELI5):** Imagine rating a movie 1-5 stars. The distance between 1 and 2 stars isn't the same as between 4 and 5. But we know 5 > 4 > 3 > 2 > 1. Ordinal regression uses this ordering without assuming equal spacing.

---

## Purpose

1.  **Model Ordinal Outcomes:** Survey responses (Likert scales), disease severity, satisfaction ratings.
2.  **Preserve Ordering:** Unlike multinomial regression, utilizes the rank information.
3.  **Calculate Odds Ratios:** Interpret effects on cumulative probabilities.

---

## When to Use

> [!success] Use Ordinal Logistic Regression When...
> - Outcome is **ordered categorical** (3+ levels with natural ranking).
> - You want to **preserve rank information** (not treat as nominal).
> - The **proportional odds assumption** holds (same OR across cutpoints).

> [!failure] Do NOT Use When...
> - Outcome is **nominal** without ordering (use [[stats/03_Regression_Analysis/Multinomial Logistic Regression (MNLogit)\|Multinomial Logistic Regression (MNLogit)]]).
> - Outcome is **binary** (use [[stats/03_Regression_Analysis/Binary Logistic Regression\|Binary Logistic Regression]]).
> - Outcome is **continuous** (use [[stats/03_Regression_Analysis/Multiple Linear Regression\|Linear Regression]] or [[stats/03_Regression_Analysis/Multiple Linear Regression\|Multiple Linear Regression]]).
> - Proportional odds assumption is **severely violated**.

---

## Theoretical Background

### The Cumulative Logit Model

For an ordinal outcome $Y$ with $J$ categories $(1, 2, \dots, J)$, we model the **cumulative probability**:

$$
P(Y \leq j | X) = \frac{1}{1 + e^{-(\alpha_j - X\beta)}}
$$

Or equivalently, the **cumulative logit**:

$$
\text{logit}[P(Y \leq j | X)] = \alpha_j - \beta_1 X_1 - \beta_2 X_2 - \dots
$$

Where:
- $\alpha_j$ = **threshold (cutpoint)** for category $j$ (there are $J-1$ thresholds)
- $\beta$ = **common slope** for all thresholds (proportional odds)

### Proportional Odds Assumption

The key assumption is that the effect of $X$ is **constant across all thresholds**:

$$
\frac{\text{Odds}(Y \leq j | X = x + 1)}{\text{Odds}(Y \leq j | X = x)} = e^{-\beta} \quad \text{(same for all } j\text{)}
$$

**Interpretation of $e^{\beta}$:** The odds ratio for being in a **higher category** (or equal) for a 1-unit increase in $X$.

### Visual Intuition

```
Category:      1    |    2    |    3    |    4    |    5
                    α₁       α₂       α₃       α₄
              <--- Lower ----|---- Higher --->
              
Cumulative P: P(Y≤1) P(Y≤2)  P(Y≤3)  P(Y≤4)   = 1
```

### Probability Calculations

$$
P(Y = j | X) = P(Y \leq j | X) - P(Y \leq j-1 | X)
$$

---

## Assumptions

- [ ] **Ordinal Outcome:** Categories have meaningful order.
- [ ] **Proportional Odds:** Effect of predictors is uniform across all thresholds.
- [ ] **Independence:** Observations are independent.
- [ ] **No Multicollinearity:** Predictors are not highly correlated.
- [ ] **Sufficient Sample Size:** ~10-20 observations per category per predictor.

### Testing Proportional Odds

1. **Brant Test** (R: `brant()`)
2. **Score Test** (SAS, Stata)
3. **Graphical Check:** Plot predicted probabilities by category

> [!warning] What if PO Fails?
> - **Partial Proportional Odds Model:** Allow some coefficients to vary by threshold.
> - **Multinomial Logit:** Treat categories as unordered.
> - **Continuation Ratio Model:** Alternative ordinal model.

---

## Limitations

> [!warning] Pitfalls
> 1. **PO Violation:** If proportional odds fails, estimates are biased. Always test!
> 2. **Interpretation Complexity:** Cumulative ORs are harder to explain than binary.
> 3. **Sparse Categories:** Categories with few observations cause instability.
> 4. **Arbitrary Thresholds:** Combining or splitting categories changes results.

---

## Python Implementation

```python
import numpy as np
import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel
import matplotlib.pyplot as plt

# ========== EXAMPLE DATA ==========
# Patient satisfaction survey (1=Very Dissatisfied to 5=Very Satisfied)
np.random.seed(42)
n = 500

df = pd.DataFrame({
    'age': np.random.normal(50, 15, n),
    'wait_time': np.random.exponential(30, n),  # minutes
    'treatment': np.random.choice(['A', 'B'], n)
})

# Generate ordinal outcome based on predictors
linear_pred = -0.02 * df['age'] - 0.05 * df['wait_time'] + 0.8 * (df['treatment'] == 'A')
probs = 1 / (1 + np.exp(-linear_pred))
df['satisfaction'] = pd.cut(probs, bins=5, labels=[1, 2, 3, 4, 5]).astype(int)

print("Outcome Distribution:")
print(df['satisfaction'].value_counts().sort_index())

# ========== FIT ORDINAL LOGISTIC REGRESSION ==========
# Convert treatment to numeric
df['treatment_A'] = (df['treatment'] == 'A').astype(int)

# Fit model
model = OrderedModel(df['satisfaction'], 
                     df[['age', 'wait_time', 'treatment_A']], 
                     distr='logit')
result = model.fit(method='bfgs', disp=False)
print("\n=== Model Summary ===")
print(result.summary())

# ========== ODDS RATIOS ==========
print("\n=== Odds Ratios ===")
params = result.params[:-4]  # Exclude thresholds
or_values = np.exp(-params)  # Note: sign convention
print(pd.DataFrame({
    'Coefficient': params,
    'Odds Ratio': or_values
}))

# ========== PREDICTED PROBABILITIES ==========
# Predict for a specific patient
new_patient = pd.DataFrame({
    'age': [45],
    'wait_time': [20],
    'treatment_A': [1]
})

pred_probs = result.predict(new_patient)
print("\n=== Predicted Probabilities ===")
print(f"P(Satisfaction = 1): {pred_probs[0, 0]:.3f}")
print(f"P(Satisfaction = 2): {pred_probs[0, 1]:.3f}")
print(f"P(Satisfaction = 3): {pred_probs[0, 2]:.3f}")
print(f"P(Satisfaction = 4): {pred_probs[0, 3]:.3f}")
print(f"P(Satisfaction = 5): {pred_probs[0, 4]:.3f}")

# ========== VISUALIZATION ==========
# Predicted probabilities by wait time
wait_range = np.linspace(5, 90, 50)
pred_data = pd.DataFrame({
    'age': [50] * 50,
    'wait_time': wait_range,
    'treatment_A': [1] * 50
})
pred_probs = result.predict(pred_data)

fig, ax = plt.subplots(figsize=(10, 6))
for i in range(5):
    ax.plot(wait_range, pred_probs[:, i], label=f'Satisfaction = {i+1}')
ax.set_xlabel('Wait Time (minutes)')
ax.set_ylabel('Probability')
ax.set_title('Predicted Probabilities by Wait Time')
ax.legend()
ax.grid(alpha=0.3)
plt.show()
```

---

## R Implementation

```r
library(MASS)      # polr function
library(brant)     # Brant test for proportional odds
library(effects)   # Effect plots
library(broom)
library(ggplot2)

# ========== EXAMPLE DATA ==========
set.seed(42)
n <- 500

df <- data.frame(
  age = rnorm(n, 50, 15),
  wait_time = rexp(n, 1/30),
  treatment = factor(sample(c("A", "B"), n, replace = TRUE))
)

# Generate ordinal outcome
linear_pred <- -0.02 * df$age - 0.05 * df$wait_time + 0.8 * (df$treatment == "A")
probs <- plogis(linear_pred)
df$satisfaction <- factor(cut(probs, breaks = 5, labels = 1:5), ordered = TRUE)

table(df$satisfaction)

# ========== FIT ORDINAL LOGISTIC REGRESSION ==========
model <- polr(satisfaction ~ age + wait_time + treatment, 
              data = df, 
              Hess = TRUE)
summary(model)

# ========== ODDS RATIOS WITH CI ==========
# Extract coefficients and compute ORs
coef_table <- coef(summary(model))
p_values <- pnorm(abs(coef_table[, "t value"]), lower.tail = FALSE) * 2

# Odds Ratios (for higher satisfaction)
ORs <- exp(coef(model))
CI <- exp(confint(model))

cat("\n=== Odds Ratios ===\n")
print(cbind(OR = ORs, CI, "p-value" = p_values[1:length(ORs)]))

# ========== TEST PROPORTIONAL ODDS ASSUMPTION ==========
cat("\n=== Brant Test for Proportional Odds ===\n")
brant_test <- brant(model)
print(brant_test)
# p > 0.05 for each variable means PO assumption holds

# ========== PREDICTED PROBABILITIES ==========
new_patient <- data.frame(
  age = 45,
  wait_time = 20,
  treatment = factor("A", levels = c("A", "B"))
)

pred_probs <- predict(model, newdata = new_patient, type = "probs")
cat("\n=== Predicted Probabilities ===\n")
print(round(pred_probs, 3))

# ========== VISUALIZATION ==========
# Effect plot
plot(Effect("wait_time", model))

# Manual probability curves
wait_seq <- seq(5, 90, length.out = 50)
pred_data <- data.frame(
  age = 50,
  wait_time = wait_seq,
  treatment = factor("A", levels = c("A", "B"))
)
pred_probs <- predict(model, newdata = pred_data, type = "probs")

pred_long <- data.frame(
  wait_time = rep(wait_seq, 5),
  satisfaction = factor(rep(1:5, each = 50)),
  probability = as.vector(pred_probs)
)

ggplot(pred_long, aes(x = wait_time, y = probability, color = satisfaction)) +
  geom_line(size = 1) +
  labs(title = "Predicted Probabilities by Wait Time",
       x = "Wait Time (minutes)", y = "Probability") +
  theme_minimal()
```

---

## Worked Numerical Example

> [!example] Job Satisfaction Survey
> **Scenario:** HR analyzes factors affecting job satisfaction (1=Low, 2=Medium, 3=High).
> **Predictors:** Salary (in $10K), Years of Experience.
> 
> **Model Output:**
> 
> | Parameter | Coefficient | SE | Odds Ratio |
> |-----------|-------------|-----|------------|
> | Salary | 0.35 | 0.08 | 1.42 |
> | Experience | 0.12 | 0.05 | 1.13 |
> | α₁ (Low\|Medium) | -2.5 | - | - |
> | α₂ (Medium\|High) | 0.8 | - | - |
> 
> **Interpretation:**
> 
> 1. **Salary OR = 1.42:**
>    - For each $10K increase in salary, the odds of being in a **higher satisfaction category** (vs. lower or equal) increase by **42%**.
>    - This effect is the same whether comparing Low vs. (Medium+High) or (Low+Medium) vs. High.
> 
> 2. **Experience OR = 1.13:**
>    - Each additional year of experience increases odds of higher satisfaction by 13%.
> 
> 3. **Thresholds:**
>    - $\alpha_1 = -2.5$: Logit of P(Y ≤ Low) when all X = 0.
>    - $\alpha_2 = 0.8$: Logit of P(Y ≤ Medium) when all X = 0.
> 
> **Predicted Probability for Employee with Salary = $60K, Experience = 5 years:**
> 
> $$\text{logit}[P(Y \leq \text{Low})] = -2.5 - 0.35(6) - 0.12(5) = -5.2$$
> $$P(Y \leq \text{Low}) = \frac{1}{1+e^{5.2}} = 0.005 \quad (0.5\%)$$
> 
> $$\text{logit}[P(Y \leq \text{Medium})] = 0.8 - 0.35(6) - 0.12(5) = -1.9$$
> $$P(Y \leq \text{Medium}) = \frac{1}{1+e^{1.9}} = 0.13 \quad (13\%)$$
> 
> Therefore:
> - P(Low) = 0.5%
> - P(Medium) = 13% - 0.5% = 12.5%
> - P(High) = 100% - 13% = **87%**

---

## Interpretation Guide

| Output | Example | Interpretation | Edge Case |
|--------|---------|----------------|-----------|
| OR | 1.5 | 50% higher odds of being in higher category | OR very close to 1 → minimal effect |
| OR | 0.7 | 30% lower odds of higher category | OR < 1 → predictor decreases outcome |
| Threshold α₁ | -2.0 | Log-odds of Y ≤ 1 at baseline | Lower threshold → fewer in low category |
| Threshold α₂ | 1.5 | Log-odds of Y ≤ 2 at baseline | Higher threshold → more in low+medium |
| Brant p-value | 0.02 | PO assumption violated for this variable! | Consider partial proportional odds |

---

## Common Pitfall Example

> [!warning] Ignoring Proportional Odds Violation
> **Scenario:** Analyzing pain severity (None < Mild < Moderate < Severe) by drug dosage.
> 
> **Brant Test Results:**
> | Variable | p-value |
> |----------|---------|
> | Dosage | 0.01 (violated!) |
> | Age | 0.45 (OK) |
> 
> **Problem:** The effect of dosage differs across thresholds:
> - OR for None vs. (Mild+Moderate+Severe) = 0.8
> - OR for (None+Mild) vs. (Moderate+Severe) = 0.4
> 
> **Consequence:** Single OR is misleading. Report category-specific effects or use:
> - Partial proportional odds model
> - Multinomial logistic regression
> - Separate binary logits

---

## Related Concepts

**Prerequisites:**
- [[stats/03_Regression_Analysis/Binary Logistic Regression\|Binary Logistic Regression]] - Two-category version
- [[stats/03_Regression_Analysis/Generalized Linear Models (GLM)\|Generalized Linear Models (GLM)]] - Framework

**Alternatives:**
- [[stats/03_Regression_Analysis/Multinomial Logistic Regression (MNLogit)\|Multinomial Logistic Regression (MNLogit)]] - Unordered categories
- [[stats/03_Regression_Analysis/Poisson Regression\|Poisson Regression]] - Count outcomes

**Extensions:**
- [[stats/03_Regression_Analysis/Linear Mixed Models (LMM)\|Mixed Effects Models]] - Clustered ordinal data
- Continuation Ratio Model - Different parameterization

---

## References

- **Book:** Agresti, A. (2010). *Analysis of Ordinal Categorical Data* (2nd ed.). Wiley. (Chapters 4-6) [Wiley Link](https://www.wiley.com/en-us/Analysis+of+Ordinal+Categorical+Data%2C+2nd+Edition-p-9780470082898)
- **Book:** Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression* (3rd ed.). Wiley. (Chapter 8) [Wiley Link](https://www.wiley.com/en-us/Applied+Logistic+Regression,+3rd+Edition-p-9780470582473)
- **Book:** Long, J. S., & Freese, J. (2014). *Regression Models for Categorical Dependent Variables Using Stata* (3rd ed.). Stata Press. (Chapter 7) [Stata Press Link](https://www.stata.com/bookstore/regression-models-categorical-dependent-variables/)
- **Article:** Brant, R. (1990). Assessing Proportionality in the Proportional Odds Model for Ordinal Logistic Regression. *Biometrics*, 46(4), 1171-1178. [DOI: 10.2307/2532457](https://doi.org/10.2307/2532457)
- **Article:** McCullagh, P. (1980). Regression Models for Ordinal Data. *Journal of the Royal Statistical Society: Series B*, 42(2), 109-142. [JSTOR Link](https://www.jstor.org/stable/2984952)
