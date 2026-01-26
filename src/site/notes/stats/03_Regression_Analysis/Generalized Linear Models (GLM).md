---
{"dg-publish":true,"permalink":"/stats/03-regression-analysis/generalized-linear-models-glm/","tags":["Regression","Advanced","Statistics","Exponential-Family"]}
---


## Definition

> [!abstract] Core Statement
> **Generalized Linear Models (GLM)** extend ordinary linear regression to handle response variables that follow **any distribution from the Exponential Family** (not just Normal). A GLM uses a **link function** to connect the linear predictor to the expected value of the response.

**Intuition (ELI5):** Regular regression assumes your outcome is a bell curve. But what if you're predicting counts (can't be negative) or yes/no outcomes? GLM stretches or squishes the predictions through a "link" function to make the math work for these different data types.

**The GLM Framework:**
$$
g(E[Y]) = X\beta
$$

Where:
- $E[Y]$ = Expected value of response
- $g(\cdot)$ = Link function
- $X\beta$ = Linear predictor

---

## When to Use

> [!success] Use GLM When...
> - Response variable is **not continuous/normal** (counts, binary, proportions).
> - Relationship between predictors and response is **non-linear** on original scale.
> - Variance **depends on the mean** (heteroscedasticity by design).
> - You need a unified framework for **Logistic, Poisson, Gamma** regressions.

> [!failure] Do NOT Use GLM When...
> - Response is **continuous and normally distributed** — use OLS.
> - Data has **zero-inflation** beyond what Poisson/NegBin handles — use [[stats/03_Regression_Analysis/Zero-Inflated Models\|Zero-Inflated Models]].
> - You need **mixed effects** — use [[stats/03_Regression_Analysis/Linear Mixed Models (LMM)\|Linear Mixed Models (LMM)]].
> - Relationships are **highly non-linear** — consider GAM or ML methods.

---

## Theoretical Background

### Three Components of GLM

| Component | Description | Example (Logistic) |
|-----------|-------------|-------------------|
| **Random** | Distribution of Y | Y ~ Bernoulli |
| **Systematic** | Linear predictor | $\eta = \beta_0 + \beta_1 X_1 + ...$ |
| **Link** | Connects μ to η | $\text{logit}(p) = \eta$ |

### Common GLM Families

| Distribution | Link Function | Use Case | Coefficient Interpretation |
|--------------|---------------|----------|---------------------------|
| **Gaussian** | Identity: $\mu = \eta$ | Continuous Y (OLS) | $\beta$ = change in Y |
| **Binomial** | Logit: $\log\frac{p}{1-p} = \eta$ | Binary Y (0/1) | $e^\beta$ = odds ratio |
| **Poisson** | Log: $\log(\lambda) = \eta$ | Count Y (0, 1, 2...) | $e^\beta$ = rate ratio |
| **Gamma** | Inverse: $\frac{1}{\mu} = \eta$ | Positive, skewed Y | Complex |
| **Inverse Gaussian** | $\frac{1}{\mu^2} = \eta$ | Positive, right-skewed | Complex |

### The Exponential Family

All GLM distributions share this form:
$$
f(y|\theta, \phi) = \exp\left(\frac{y\theta - b(\theta)}{a(\phi)} + c(y, \phi)\right)
$$

This structure enables:
- Unified estimation via **IRLS** (Iteratively Reweighted Least Squares)
- Automatic **variance functions**
- Consistent **deviance** calculations

### Variance Functions

| Family | Variance | $\text{Var}(Y)$ depends on $\mu$ |
|--------|----------|----------------------------------|
| Gaussian | $\sigma^2$ | Constant (homoscedastic) |
| Binomial | $n \cdot p(1-p)$ | Highest at p=0.5 |
| Poisson | $\lambda$ | Equal to mean |
| Gamma | $\mu^2 / \phi$ | Proportional to mean² |

---

## Assumptions & Diagnostics

- [ ] **Correct Distribution:** Response follows assumed family.
- [ ] **Correct Link:** Link function appropriate for data.
- [ ] **Independence:** Observations independent.
- [ ] **Linearity (on link scale):** Linear relationship holds after transformation.

### Diagnostics

| Diagnostic | Purpose | Check |
|------------|---------|-------|
| **Deviance residuals** | Overall fit | Should be ~N(0,1) |
| **Pearson residuals** | Outliers | No pattern vs fitted values |
| **Dispersion parameter** | Overdispersion | Should be ~1 (Poisson/Binomial) |
| **AIC/BIC** | Model comparison | Lower is better |

---

## Implementation

### Python

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# ========== EXAMPLE 1: POISSON REGRESSION (Count Data) ==========
# Predicting number of customer complaints per day
np.random.seed(42)
n = 200
temperature = np.random.uniform(60, 100, n)
weekend = np.random.binomial(1, 0.3, n)
# True model: complaints = exp(-2 + 0.03*temp + 0.5*weekend)
true_lambda = np.exp(-2 + 0.03*temperature + 0.5*weekend)
complaints = np.random.poisson(true_lambda)

df = pd.DataFrame({
    'complaints': complaints,
    'temperature': temperature,
    'weekend': weekend
})

# Fit Poisson GLM
poisson_model = smf.glm(
    'complaints ~ temperature + weekend',
    data=df,
    family=sm.families.Poisson()
).fit()

print("=== POISSON GLM ===")
print(poisson_model.summary())

# Interpretation: Coefficients are log(rate ratio)
print("\nRate Ratios (exp of coefficients):")
print(np.exp(poisson_model.params))
# e.g., exp(0.5) = 1.65 → weekends have 65% more complaints

# ========== EXAMPLE 2: LOGISTIC REGRESSION (Binary Data) ==========
# Predicting customer churn
np.random.seed(42)
tenure = np.random.uniform(1, 72, n)
monthly_charges = np.random.uniform(20, 100, n)
log_odds = -2 + 0.02*monthly_charges - 0.05*tenure
prob = 1 / (1 + np.exp(-log_odds))
churn = np.random.binomial(1, prob)

df2 = pd.DataFrame({
    'churn': churn,
    'tenure': tenure,
    'monthly_charges': monthly_charges
})

# Fit Logistic GLM (Binomial with logit link)
logit_model = smf.glm(
    'churn ~ tenure + monthly_charges',
    data=df2,
    family=sm.families.Binomial()
).fit()

print("\n=== LOGISTIC GLM ===")
print(logit_model.summary())

print("\nOdds Ratios:")
print(np.exp(logit_model.params))

# ========== EXAMPLE 3: GAMMA REGRESSION (Positive Continuous) ==========
# Predicting insurance claim amounts (always positive, right-skewed)
np.random.seed(42)
age = np.random.uniform(18, 70, n)
claim_amount = np.random.gamma(shape=2, scale=100 + 5*age)

df3 = pd.DataFrame({'claim_amount': claim_amount, 'age': age})

gamma_model = smf.glm(
    'claim_amount ~ age',
    data=df3,
    family=sm.families.Gamma(link=sm.families.links.Log())
).fit()

print("\n=== GAMMA GLM ===")
print(gamma_model.summary())

# ========== DIAGNOSTICS ==========
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Residuals vs Fitted
axes[0].scatter(poisson_model.fittedvalues, poisson_model.resid_deviance)
axes[0].axhline(0, color='red', linestyle='--')
axes[0].set_xlabel('Fitted Values')
axes[0].set_ylabel('Deviance Residuals')
axes[0].set_title('Poisson: Residuals vs Fitted')

# Q-Q plot of residuals
sm.qqplot(poisson_model.resid_deviance, line='45', ax=axes[1])
axes[1].set_title('Q-Q Plot of Deviance Residuals')

plt.tight_layout()
plt.show()
```

### R

```r
# ========== EXAMPLE 1: POISSON REGRESSION ==========
set.seed(42)
n <- 200
temperature <- runif(n, 60, 100)
weekend <- rbinom(n, 1, 0.3)
lambda <- exp(-2 + 0.03*temperature + 0.5*weekend)
complaints <- rpois(n, lambda)

df <- data.frame(complaints, temperature, weekend)

poisson_model <- glm(complaints ~ temperature + weekend,
                     family = poisson(link = "log"),
                     data = df)

summary(poisson_model)

# Rate ratios
cat("\nRate Ratios:\n")
exp(coef(poisson_model))

# Check overdispersion
cat("\nDispersion parameter:", 
    sum(residuals(poisson_model, type = "pearson")^2) / poisson_model$df.residual, "\n")
# If >> 1, use quasipoisson or negative binomial

# ========== EXAMPLE 2: LOGISTIC REGRESSION ==========
tenure <- runif(n, 1, 72)
monthly_charges <- runif(n, 20, 100)
log_odds <- -2 + 0.02*monthly_charges - 0.05*tenure
prob <- plogis(log_odds)
churn <- rbinom(n, 1, prob)

df2 <- data.frame(churn, tenure, monthly_charges)

logit_model <- glm(churn ~ tenure + monthly_charges,
                   family = binomial(link = "logit"),
                   data = df2)

summary(logit_model)
cat("\nOdds Ratios:\n")
exp(coef(logit_model))

# ========== EXAMPLE 3: GAMMA REGRESSION ==========
age <- runif(n, 18, 70)
claim_amount <- rgamma(n, shape = 2, rate = 1/(100 + 5*age))

df3 <- data.frame(claim_amount, age)

gamma_model <- glm(claim_amount ~ age,
                   family = Gamma(link = "log"),
                   data = df3)

summary(gamma_model)

# ========== DIAGNOSTICS ==========
par(mfrow = c(2, 2))
plot(poisson_model)
```

---

## Interpretation Guide

| Model | Coefficient | Interpretation | Example |
|-------|-------------|----------------|---------|
| **Gaussian (Identity)** | $\beta = 2$ | Y increases by 2 for each unit increase in X | Salary +$2K per year of experience |
| **Binomial (Logit)** | $\beta = 0.5$ | Odds multiply by $e^{0.5} = 1.65$ | 65% higher odds of churn |
| **Poisson (Log)** | $\beta = 0.3$ | Rate multiplies by $e^{0.3} = 1.35$ | 35% more complaints per hour |
| **Gamma (Log)** | $\beta = 0.1$ | Mean multiplies by $e^{0.1} = 1.11$ | 11% higher claim amount |

### Key Outputs

| Output | Example | Check |
|--------|---------|-------|
| **Dispersion** | 3.2 (Poisson) | If > 1, overdispersion — use quasipoisson or NegBin |
| **Null deviance** | 150 | Deviance of intercept-only model |
| **Residual deviance** | 95 | Deviance of full model — lower = better fit |
| **AIC** | 320 | Compare across models — lower is better |

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Ignoring Overdispersion**
> - *Problem:* Using Poisson when variance >> mean.
> - *Result:* Standard errors too small → false significance.
> - *Solution:* Check dispersion; use Negative Binomial if > 1.5.
>
> **2. Wrong Link Function**
> - *Problem:* Using identity link for proportions (can give p > 1).
> - *Result:* Nonsensical predictions.
> - *Solution:* Match link to data type (logit for binary, log for counts).
>
> **3. Interpreting Coefficients on Wrong Scale**
> - *Problem:* Saying "β = 0.5 means Y increases by 0.5."
> - *Reality:* For log link, β = 0.5 means Y multiplies by exp(0.5) = 1.65.
> - *Solution:* Exponentiate coefficients for rate/odds ratios.
>
> **4. Complete Separation in Logistic**
> - *Problem:* Predictor perfectly separates outcomes.
> - *Result:* Coefficients go to infinity; algorithm fails.
> - *Solution:* Use Firth's penalized likelihood or combine categories.

---

## Worked Numerical Example

> [!example] Poisson Regression for Website Clicks
> **Data:** Clicks per ad placement, varying by position and ad size.
>
> | Observation | Position | Size | Clicks |
> |-------------|----------|------|--------|
> | 1 | Top | Large | 15 |
> | 2 | Top | Small | 8 |
> | 3 | Side | Large | 6 |
> | 4 | Side | Small | 3 |
>
> **Model:** $\log(\lambda) = \beta_0 + \beta_1 \cdot \text{Top} + \beta_2 \cdot \text{Large}$
>
> **Fitted Coefficients:**
> - $\beta_0 = 1.1$ (baseline: Side, Small)
> - $\beta_1 = 0.7$ (Top vs Side)
> - $\beta_2 = 0.5$ (Large vs Small)
>
> **Predictions:**
> - Side, Small: $\lambda = e^{1.1} = 3.0$ clicks
> - Top, Large: $\lambda = e^{1.1 + 0.7 + 0.5} = e^{2.3} = 10.0$ clicks
>
> **Interpretation:**
> - Top position: $e^{0.7} = 2.0$× more clicks (Rate Ratio)
> - Large size: $e^{0.5} = 1.65$× more clicks

---

## GLM Family Decision Tree

```
Is Y binary (0/1)?
├─ Yes → Binomial (Logistic Regression)
│
└─ No → Is Y a count (0, 1, 2, ...)?
         ├─ Yes → Is variance ≈ mean?
         │         ├─ Yes → Poisson
         │         └─ No (variance > mean) → Negative Binomial
         │
         └─ No → Is Y always positive and skewed?
                  ├─ Yes → Gamma
                  └─ No → Gaussian (OLS)
```

---

## Related Concepts

**Specific GLMs:**
- [[stats/03_Regression_Analysis/Logistic Regression\|Logistic Regression]] — Binary/Binomial
- [[stats/03_Regression_Analysis/Poisson Regression\|Poisson Regression]] — Count data
- [[stats/03_Regression_Analysis/Negative Binomial Regression\|Negative Binomial Regression]] — Overdispersed counts

**Extensions:**
- [[stats/03_Regression_Analysis/Zero-Inflated Models\|Zero-Inflated Models]] — Excess zeros
- [[stats/03_Regression_Analysis/Linear Mixed Models (LMM)\|Linear Mixed Models (LMM)]] — Random effects
- [[GAMs\|GAMs]] — Non-linear relationships
