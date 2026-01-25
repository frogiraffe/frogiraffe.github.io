---
{"dg-publish":true,"permalink":"/stats/03-regression-analysis/probit-regression/","tags":["Regression","GLM","Classification"]}
---

## Definition

> [!abstract] Core Statement
> **Probit Regression** is a type of Generalized Linear Model used for binary outcomes, similar to [[stats/03_Regression_Analysis/Binary Logistic Regression\|Binary Logistic Regression]]. Instead of the logit (log-odds) link, it uses the ==cumulative distribution function (CDF) of the standard normal distribution== ($\Phi$) as the link function.

---

## Purpose

1.  Model binary outcomes with an alternative to the logit link.
2.  Commonly used in **Econometrics** and **biometric assays**.
3.  Provides coefficients interpretable in terms of **standard deviation units**.

---

## When to Use

> [!success] Use Probit When...
> - You have a **binary outcome**.
> - Theoretical reasons suggest an underlying **normal latent variable** (e.g., threshold models in psychometrics).
> - Disciplinary conventions favor Probit (e.g., labor economics).

> [!tip] Logit vs Probit
> In practice, **Logit** and **Probit** give very similar predictions. Logit is more common because Odds Ratios are intuitive. Probit is used when a normal latent variable interpretation is desired.

---

## Theoretical Background

### The Model

$$
P(Y=1 | X) = \Phi(\beta_0 + \beta_1 X_1 + \dots)
$$

where $\Phi(z)$ is the CDF of the standard normal distribution:
$$
\Phi(z) = \int_{-\infty}^{z} \frac{1}{\sqrt{2\pi}} e^{-t^2/2} dt
$$

### Latent Variable Interpretation

Probit assumes there is an unobserved continuous variable $Y^*$:
$$
Y^* = \beta_0 + \beta_1 X + \varepsilon, \quad \varepsilon \sim N(0,1)
$$
$$
Y = 1 \text{ if } Y^* > 0, \text{ else } Y = 0
$$

### Logit vs Probit Comparison

| Feature | Logit | Probit |
|---------|-------|--------|
| **Link Function** | Logistic (Sigmoid) | Normal CDF ($\Phi$) |
| **Coefficient Interpretation** | Log-Odds / Odds Ratio | Change in Z-score |
| **Tail Behavior** | Heavier tails | Lighter tails |
| **Common In** | Medical, Social Sciences | Econometrics |

---

## Assumptions

Same as [[stats/03_Regression_Analysis/Binary Logistic Regression\|Binary Logistic Regression]]:
- [ ] Binary outcome.
- [ ] Independence of observations.
- [ ] Correct specification (linear in $X$).
- [ ] No perfect separation.

---

## Limitations

> [!warning] Pitfalls
> 1.  **No Odds Ratio:** Coefficients are in "Z-score" units, which are less intuitive than Odds Ratios.
> 2.  **Almost identical to Logit:** Differences are negligible in most applications. Choose based on convention or interpretation needs.

---

## Python Implementation

```python
import statsmodels.api as sm

# Fit Probit
X = sm.add_constant(df[['age', 'income']])
y = df['purchased']

model_probit = sm.Probit(y, X).fit()
print(model_probit.summary())

# Marginal Effects (Change in P for 1-unit change in X)
mfx = model_probit.get_margeff()
print(mfx.summary())
```

---

## R Implementation

```r
# Fit Probit (GLM with binomial family and probit link)
model_probit <- glm(purchased ~ age + income, data = df, 
                    family = binomial(link = "probit"))
summary(model_probit)

# Marginal Effects (mfx package)
library(mfx)
probitmfx(purchased ~ age + income, data = df, atmean = TRUE)
```

---

## Worked Numerical Example

> [!example] College Admission: Probit vs Logit
> **Outcome:** Admitted (1) or Rejected (0)
> **Predictor:** Exam Score (0-100)
> 
> **Probit Results:**
> - β_score = 0.04
> - At Score=70: P(Admission) = Φ(β₀ + 0.04×70) = Φ(-1 + 2.8) = Φ(1.8) = 0.964
> - Marginal effect at mean: 0.015 (1 point → 1.5% higher probability)
> 
> **Logit Results (same data):**
> - β_score = 0.07 (≈ 1.75 × Probit coefficient)
> - At Score=70: P(Admission) = 1/(1+e^-(β₀+0.07×70)) = 0.962
> - Almost identical predictions!
> 
> **Key Insight:**
> - Probit and Logit give nearly identical fitted probabilities
> - Coefficients differ by ~1.6-1.8× factor
> - Marginal effects are almost the same
> - Choice barely matters for prediction; convention is use Logit

---

## Interpretation Guide

| Output | Interpretation | Edge Case Notes |
|--------|----------------|------------------|
| Probit β = 0.8 | 1-unit increase in X increases z-score by 0.8. | Not directly interpretable! Must compute marginal effects. |
| Logit β ≈ 1.6 × Probit β | Rough conversion rule. | Ratio varies slightly (1.6-1.8) depending on data. |
| Marginal Effect = 0.25 | At mean, 1-unit ↑ in X → 25pp ↑ in P(Y=1). | **Percentage points**, not percent! 0.25 = 25pp, not 25%. |
| Marginal Effect = 0.01 | Very small effect (1 percentage point). | May be practically unimportant even if significant. |
| Predicted P > 1 or < 0 | **Impossible!** Model error or coding bug. | Probit automatically constrains to [0,1] via Φ. If you see this, check code. |

---

## Common Pitfall Example

> [!warning] Misinterpreting Probit Coefficients Directly
> **Wrong Interpretation:**
> - β_income = 0.0002
> - Analyst says: "Each $1 increase in income increases probability of purchase by 0.02%"
> 
> **Why Wrong:**
> - Probit coefficients are on the **z-score scale**, not probability scale
> - The effect on probability depends on **where you are on the curve**
> 
> **Correct Approach:**
> 1. Calculate marginal effects: `marginal = β × φ(Xβ)` where φ is normal PDF
> 2. Or use software: `margins` in Stata, `marginaleffects()` in R
> 3. Report: "At mean income, $1000 increase → 3% higher purchase probability"
> 
> **Example showing non-linearity:**
> - At P=0.5 (middle): $1000 income → +5% probability
> - At P=0.95 (tail): $1000 income → +0.5% probability (much smaller!)
> - Effect depends on baseline probability (S-curve shape)

---

## Related Concepts

- [[stats/03_Regression_Analysis/Binary Logistic Regression\|Binary Logistic Regression]] - The more common alternative.
- [[stats/01_Foundations/Maximum Likelihood Estimation (MLE)\|Maximum Likelihood Estimation (MLE)]]
- [[stats/03_Regression_Analysis/Generalized Linear Models (GLM)\|Generalized Linear Models (GLM)]]
