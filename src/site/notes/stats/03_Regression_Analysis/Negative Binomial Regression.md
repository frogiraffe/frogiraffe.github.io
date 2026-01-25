---
{"dg-publish":true,"permalink":"/stats/03-regression-analysis/negative-binomial-regression/","tags":["Regression","GLM","Count-Data","Overdispersion"]}
---

## Definition

> [!abstract] Core Statement
> **Negative Binomial Regression** is an extension of [[stats/03_Regression_Analysis/Poisson Regression\|Poisson Regression]] used when count data exhibits ==overdispersion== (Variance > Mean). It introduces an extra parameter to model the heterogeneity in event rates across observations.

---

## Purpose

1.  Provide valid inference for count data when **Poisson's equidispersion assumption fails**.
2.  Model counts where unobserved heterogeneity causes extra variance (e.g., some customers are inherently more active than others).

---

## When to Use

> [!success] Use Negative Binomial When...
> - Outcome is a **count**.
> - The [[stats/03_Regression_Analysis/Poisson Regression\|Poisson Regression]] model shows **overdispersion** (Dispersion statistic > 1.5).
> - Data has **unobserved heterogeneity** in rates.

> [!failure] Do NOT Use NegBin When...
> - Equidispersion holds (Poisson is simpler and sufficient).
> - Excess zeros are the main problem (use [[stats/03_Regression_Analysis/Zero-Inflated Models\|Zero-Inflated Models]]).

---

## Theoretical Background

### Mean-Variance Relationship

| Model | Variance | Implication |
|-------|----------|-------------|
| Poisson | $Var(Y) = \mu$ | Mean = Variance |
| Negative Binomial | $Var(Y) = \mu + \alpha \mu^2$ | Variance > Mean |

The parameter $\alpha$ (dispersion parameter) controls the extra variance. When $\alpha = 0$, NegBin reduces to Poisson.

### The Model

Like Poisson, uses a log link:
$$
\ln(\mu) = \beta_0 + \beta_1 X_1 + \dots
$$

Coefficients are interpreted as **Rate Ratios** ($e^{\beta}$), same as Poisson.

---

## Assumptions

- [ ] **Count Data**.
- [ ] **Independence**.
- [ ] **Log-linearity** of expected count.
- [ ] **Overdispersion is due to unobserved heterogeneity** (not model misspecification).

---

## Limitations

> [!warning] Pitfalls
> 1.  **Still sensitive to excess zeros.** If you have structural zeros (e.g., people who will *never* use a product), consider [[stats/03_Regression_Analysis/Zero-Inflated Models\|Zero-Inflated Models]].
> 2.  **Model selection:** Use likelihood ratio tests or AIC to compare Poisson vs NegBin.

---

## Python Implementation

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Fit Negative Binomial (uses NegativeBinomial family in GLM)
# Alternative: statsmodels.discrete.discrete_model.NegativeBinomial

model_nb = smf.negativebinomial("count ~ age + income", data=df).fit()
print(model_nb.summary())

# Rate Ratios
import numpy as np
print("\n--- Rate Ratios ---")
print(np.exp(model_nb.params))
```

---

## R Implementation

```r
library(MASS)

# Fit Negative Binomial
model_nb <- glm.nb(count ~ age + income, data = df)
summary(model_nb)

# Check Theta (Dispersion Parameter)
# Smaller theta = more overdispersion
cat("Theta:", model_nb$theta, "\n")

# Rate Ratios
exp(coef(model_nb))
exp(confint(model_nb))

# Compare to Poisson via LRT
model_pois <- glm(count ~ age + income, data = df, family = poisson)
anova(model_pois, model_nb, test = "LRT")
```

---

## Worked Numerical Example

> [!example] Hospital Visits: Poisson vs Negative Binomial
> **Outcome:** Number of hospital visits per year (count data)
> **Predictor:** Chronic illness severity (0-10 scale)
> 
> **Poisson Results:**
> - β_severity = 0.22, RR = e^0.22 = 1.25
> - Dispersion = Residual Deviance/df = 156/98 = **1.59** (overdispersion!)
> - Many p-values < 0.001 (likely inflated)
> 
> **Negative Binomial Results:**
> - β_severity = 0.18, RR = e^0.18 = 1.20
> - α (dispersion parameter) = 2.3
> - Variance = μ + 2.3μ² (much larger than Poisson's Var = μ)
> - Same predictors, but p-values more conservative (some now p = 0.04 instead of p < 0.001)
> - AIC: Poisson = 892, NegBin = 765 (NegBin much better)
> 
> **Interpretation:**
> - Severity increases visits by 20% per unit increase
> - Poisson underestimated SEs → inflated significance
> - NegBin accounts for extra variability (unobserved heterogeneity)

---

## Interpretation Guide

| Output | Example | Interpretation | Edge Case Notes |
|--------|---------|----------------|------------------|
| α (dispersion) | 2.5 | High overdispersion. Var = μ + 2.5μ². | If α→0, NegBin converges to Poisson. |
| α (dispersion) | 0.1 | Mild overdispersion. NegBin barely differs from Poisson. | May not need NegBin; Poisson sufficient. |
| RR (predictor) | 1.15 | Each unit increase multiplies count by 1.15. | Same interpretation as Poisson RR. |
| AIC_NB << AIC_Poisson | NegBin fits much better. Overdispersion confirmed. | Difference > 10 is substantial. Use NegBin. |
| AIC_NB ≈ AIC_Poisson | Models similar. Overdispersion marginal. | May stick with simpler Poisson model. |
| LRT p < 0.05 | NegBin is significantly better than Poisson; overdispersion is real. |

---

## Common Pitfall Example

> [!warning] Ignoring Overdispersion and Reporting Poisson
> **Scenario:** Analyzing number of defects in manufacturing
> 
> **Analyst's Mistake:**
> 1. Fits Poisson model
> 2. Sees dispersion = 3.2 (severe overdispersion!)
> 3. Ignores it, reports Poisson results anyway
> 4. Conclusion: "Quality training significantly reduces defects (p < 0.001)"
> 
> **Problem:**
> - Standard errors are **too small** (underestimated by ~√3.2 = 1.8×)
> - p-values are **artificially low**
> - True p-value might be 0.08 (not significant!)
> 
> **Correct Practice:**
> - Always check dispersion: Residual Deviance/df
> - If > 1.5, use Negative Binomial
> - Or at minimum, report quasi-Poisson with corrected SEs
> 
> **Real Result with NegBin:**
> - Same β coefficient
> - SE increased by 1.8×
> - p = 0.07 (no longer significant at α=0.05)
> - Conclusion changes: "Marginally significant trend, needs more data"

---

## Related Concepts

- [[stats/03_Regression_Analysis/Poisson Regression\|Poisson Regression]] - The baseline count model.
- [[stats/03_Regression_Analysis/Zero-Inflated Models\|Zero-Inflated Models]] - For excess zeros.
- [[stats/01_Foundations/Overdispersion\|Overdispersion]]