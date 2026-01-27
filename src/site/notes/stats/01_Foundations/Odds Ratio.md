---
{"dg-publish":true,"permalink":"/stats/01-foundations/odds-ratio/","tags":["Probability","Effect-Size","Epidemiology","Logistic-Regression"]}
---


## Definition

> [!abstract] Core Statement
> The **Odds Ratio (OR)** is a measure of association that quantifies the ==strength of relationship== between an exposure and an outcome. It compares the odds of an event occurring in one group to the odds of it occurring in another group.

**Intuition (ELI5):** If you're betting on a horse race, "odds" tell you how likely the horse is to win vs. lose. The Odds Ratio compares betting odds between two horses—if Horse A has OR = 3 compared to Horse B, Horse A is "3 times more likely" to win (in odds terms, not probability!).

---

## Purpose

1.  **Quantify Effect Size:** Measure how strongly an exposure is associated with an outcome.
2.  **Logistic Regression:** Interpret regression coefficients ($e^{\beta}$ = OR).
3.  **Case-Control Studies:** The only valid measure of association when sampling is based on outcome.
4.  **Meta-Analysis:** Combine results across studies using log(OR).

---

## When to Use

> [!success] Use Odds Ratio When...
> - Outcome is **binary** (Yes/No, Disease/Healthy).
> - Study design is **case-control** (OR is the only valid measure).
> - Interpreting **logistic regression** coefficients.
> - Outcome is **rare** (<10%), where OR ≈ Relative Risk.

> [!failure] Do NOT Use When...
> - You want to communicate risk to a general audience (use Relative Risk or Absolute Risk instead).
> - Outcome is **common** (>10%)—OR will exaggerate the effect.
> - Study is a **cohort or RCT** with complete follow-up (use Risk Ratio directly).

---

## Theoretical Background

### Odds vs. Probability

| Measure | Formula | Example (80% probability) |
|---------|---------|---------------------------|
| **Probability** | $P = \frac{\text{Events}}{\text{Total}}$ | $P = 0.80$ |
| **Odds** | $\text{Odds} = \frac{P}{1-P}$ | $\text{Odds} = \frac{0.80}{0.20} = 4$ |

**Key Insight:** Probability is bounded [0, 1], but Odds range from [0, ∞).

### Odds Ratio Formula

For a 2×2 contingency table:

|  | Outcome (+) | Outcome (−) |
|--|-------------|-------------|
| Exposed | a | b |
| Unexposed | c | d |

$$
\text{OR} = \frac{a/b}{c/d} = \frac{a \cdot d}{b \cdot c}
$$

### Interpretation Scale

| OR Value | Interpretation |
|----------|----------------|
| OR = 1 | No association (exposure doesn't affect odds) |
| OR > 1 | Positive association (exposure increases odds) |
| OR < 1 | Negative association (exposure decreases odds / protective) |
| OR = 2 | Exposed group has **2× the odds** of the outcome |
| OR = 0.5 | Exposed group has **half the odds** (OR = 1/2) |

### Relationship to Logistic Regression

In [[stats/03_Regression_Analysis/Binary Logistic Regression\|Binary Logistic Regression]]:

$$
\ln\left(\frac{P}{1-P}\right) = \beta_0 + \beta_1 X
$$

The coefficient $\beta_1$ is the **log-odds ratio**. Exponentiating gives:

$$
\text{OR} = e^{\beta_1}
$$

**Example:** If $\beta_{\text{smoking}} = 0.85$, then $\text{OR} = e^{0.85} \approx 2.34$.
*"Smokers have 2.34 times the odds of lung cancer compared to non-smokers."*

### Confidence Interval for OR

Since OR is skewed, we compute CI on the log scale:

$$
\ln(\text{OR}) \pm z_{\alpha/2} \cdot \text{SE}(\ln(\text{OR}))
$$

Where:
$$
\text{SE}(\ln(\text{OR})) = \sqrt{\frac{1}{a} + \frac{1}{b} + \frac{1}{c} + \frac{1}{d}}
$$

Then exponentiate the bounds to get the CI for OR.

---

## Assumptions

- [ ] **Independence:** Observations are independent (no clustering).
- [ ] **Correct Classification:** Exposure and outcome are accurately measured.
- [ ] **No Confounding:** Or confounders are controlled via stratification/regression.

---

## Limitations

> [!warning] Critical Pitfalls
> 1. **OR ≠ Relative Risk:** When outcome is common (>10%), OR exaggerates the true risk ratio. See [[stats/01_Foundations/Correlation vs Causation\|Correlation vs Causation]].
> 2. **Non-Collapsibility:** Adjusted OR ≠ Marginal OR even without confounding (mathematical property, not bias).
> 3. **Interpretation Difficulty:** Public and clinicians often misinterpret OR as probability increase.
> 4. **Zero Cells:** If any cell is 0, OR is undefined. Use Haldane-Anscombe correction (+0.5).

---

## Python Implementation

```python
import numpy as np
from scipy.stats import fisher_exact
import statsmodels.api as sm

# ========== METHOD 1: 2x2 Table ==========
# Example: Smoking and Lung Cancer
#          Cancer+  Cancer-
# Smoker      40      60
# Non-smoker  10      90

table = np.array([[40, 60], [10, 90]])
a, b = table[0]
c, d = table[1]

# Calculate OR
odds_ratio = (a * d) / (b * c)
print(f"Odds Ratio: {odds_ratio:.2f}")

# Confidence Interval (95%)
log_or = np.log(odds_ratio)
se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
ci_lower = np.exp(log_or - 1.96 * se_log_or)
ci_upper = np.exp(log_or + 1.96 * se_log_or)
print(f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")

# Fisher's Exact Test (preferred for small samples)
or_fisher, p_value = fisher_exact(table)
print(f"Fisher's OR: {or_fisher:.2f}, p-value: {p_value:.4f}")

# ========== METHOD 2: Logistic Regression ==========
import pandas as pd

# Create dataset
df = pd.DataFrame({
    'smoking': [1]*100 + [0]*100,
    'cancer': [1]*40 + [0]*60 + [1]*10 + [0]*90
})

X = sm.add_constant(df['smoking'])
model = sm.Logit(df['cancer'], X).fit(disp=0)

# Extract OR with CI
params = model.params
conf = model.conf_int()
or_logit = np.exp(params['smoking'])
or_ci = np.exp(conf.loc['smoking'])
print(f"\nLogistic Regression OR: {or_logit:.2f}")
print(f"95% CI: [{or_ci[0]:.2f}, {or_ci[1]:.2f}]")
```

---

## R Implementation

```r
# ========== METHOD 1: 2x2 Table ==========
# Example: Smoking and Lung Cancer
table <- matrix(c(40, 60, 10, 90), nrow = 2, byrow = TRUE,
                dimnames = list(c("Smoker", "Non-smoker"),
                               c("Cancer+", "Cancer-")))
print(table)

# Calculate OR manually
a <- table[1,1]; b <- table[1,2]
c <- table[2,1]; d <- table[2,2]
OR <- (a * d) / (b * c)
cat("Odds Ratio:", round(OR, 2), "\n")

# Confidence Interval
log_OR <- log(OR)
SE <- sqrt(1/a + 1/b + 1/c + 1/d)
CI_lower <- exp(log_OR - 1.96 * SE)
CI_upper <- exp(log_OR + 1.96 * SE)
cat("95% CI: [", round(CI_lower, 2), ",", round(CI_upper, 2), "]\n")

# Fisher's Exact Test
fisher.test(table)

# ========== METHOD 2: Logistic Regression ==========
library(broom)

df <- data.frame(
  smoking = c(rep(1, 100), rep(0, 100)),
  cancer = c(rep(1, 40), rep(0, 60), rep(1, 10), rep(0, 90))
)

model <- glm(cancer ~ smoking, data = df, family = "binomial")

# Odds Ratios with CI
exp(cbind(OR = coef(model), confint(model)))

# Tidy output
tidy(model, exponentiate = TRUE, conf.int = TRUE)
```

---

## Worked Numerical Example

> [!example] Case-Control Study: Coffee and Heart Disease
> **Study Design:** 200 heart disease cases, 200 healthy controls.
> **Exposure:** Heavy coffee drinking (≥5 cups/day).
> 
> **Data:**
> 
> |  | Heart Disease | Healthy |
> |--|---------------|---------|
> | Heavy Coffee | 80 | 40 |
> | Light/No Coffee | 120 | 160 |
> 
> **Step 1: Calculate OR**
> $$\text{OR} = \frac{80 \times 160}{40 \times 120} = \frac{12800}{4800} = 2.67$$
> 
> **Step 2: 95% Confidence Interval**
> $$\text{SE}(\ln \text{OR}) = \sqrt{\frac{1}{80} + \frac{1}{40} + \frac{1}{120} + \frac{1}{160}} = 0.245$$
> $$\ln(\text{OR}) = \ln(2.67) = 0.982$$
> $$\text{CI} = e^{0.982 \pm 1.96 \times 0.245} = [1.65, 4.31]$$
> 
> **Interpretation:**
> - Heavy coffee drinkers have **2.67 times the odds** of heart disease compared to light/non-drinkers.
> - 95% CI [1.65, 4.31] excludes 1 → **statistically significant** at α = 0.05.
> 
> **Caution:** This does NOT mean coffee drinkers have 2.67× the *probability* of heart disease!

---

## Interpretation Guide

| Output | Example | Interpretation | Edge Case |
|--------|---------|----------------|-----------|
| OR | 2.67 | Exposed group has 2.67× the odds of outcome | OR > 10 may indicate sparse data or confounding |
| OR | 0.37 | Exposed group has 63% lower odds (protective) | Reciprocal: 1/0.37 = 2.7× lower odds |
| 95% CI | [1.2, 5.9] | True OR likely between 1.2 and 5.9. Significant (excludes 1). | Wide CI → small sample or high variability |
| 95% CI | [0.8, 3.1] | CI includes 1 → NOT significant at α = 0.05 | May still be clinically meaningful |
| p-value | 0.002 | Strong evidence against H₀ (OR = 1) | Small p ≠ large effect. Check OR magnitude. |

---

## Common Pitfall Example

> [!warning] The Relative Risk Trap
> **Scenario:** Study on a common outcome (50% baseline rate).
> 
> **Data:**
> - Control group: 50% develop outcome (Odds = 1.0)
> - Treatment group OR = 2.0
> 
> **Incorrect Interpretation:** "Treatment doubles the risk of outcome."
> 
> **Correct Calculation:**
> - Treatment Odds = 1.0 × 2.0 = 2.0
> - Treatment Probability = 2.0 / (1 + 2.0) = 66.7%
> - **Actual Risk Ratio = 66.7% / 50% = 1.33** (not 2.0!)
> 
> **Lesson:** OR exaggerates the effect when outcome is common. Only when outcome is rare (<10%) does OR ≈ RR.

---

## Related Concepts

- [[stats/03_Regression_Analysis/Binary Logistic Regression\|Binary Logistic Regression]] - OR from regression coefficients
- [[stats/02_Statistical_Inference/Chi-Square Test of Independence\|Chi-Square Test of Independence]] - Tests if OR ≠ 1
- [[stats/02_Statistical_Inference/Fisher's Exact Test\|Fisher's Exact Test]] - Exact test for 2×2 tables
- [[Relative Risk\|Relative Risk]] - Alternative measure for cohort studies
- [[stats/02_Statistical_Inference/Confidence Intervals\|Confidence Intervals]] - Uncertainty quantification

---

## References

- **Book:** Agresti, A. (2013). *Categorical Data Analysis* (3rd ed.). Wiley. (Chapter 3) [Wiley Link](https://www.wiley.com/en-us/Categorical+Data+Analysis,+3rd+Edition-p-9780470463635)
- **Book:** Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression* (3rd ed.). Wiley. (Chapter 1) [Wiley Link](https://www.wiley.com/en-us/Applied+Logistic+Regression,+3rd+Edition-p-9780470582473)
- **Book:** Rothman, K. J., Greenland, S., & Lash, T. L. (2008). *Modern Epidemiology* (3rd ed.). Lippincott Williams & Wilkins. (Chapter 4) [Publisher Link](https://shop.lww.com/Modern-Epidemiology/p/9781451190052)
- **Article:** Bland, J. M., & Altman, D. G. (2000). The odds ratio. *BMJ*, 320(7247), 1468. [DOI: 10.1136/bmj.320.7247.1468](https://doi.org/10.1136/bmj.320.7247.1468)
