---
{"dg-publish":true,"permalink":"/stats/03-regression-analysis/interaction-effects/","tags":["Regression","Analysis","Moderation"]}
---


## Definition

> [!abstract] Core Statement
> An **Interaction Effect** occurs when the effect of one predictor on the outcome **depends on the level of another predictor**. The combined effect is different from the sum of individual effects.

$$
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 (X_1 \times X_2) + \epsilon
$$

**Intuition (ELI5):** Coffee improves alertness. Exercise improves alertness. But coffee + exercise together might make you MORE alert than the sum of each alone (synergy) or LESS alert because you're jittery (antagonism). That extra effect beyond the sum is the interaction.

**Key Insight:** Without interaction, effects are **additive**. With interaction, effects are **conditional**.

---

## When to Use

> [!success] Include Interaction Terms When...
> - Theory suggests the effect of X depends on Z (**moderation hypothesis**).
> - Subgroup analyses show different effects (effect heterogeneity).
> - Residual plots show patterns unexplained by main effects.
> - You're testing if a treatment works differently for different groups.

> [!failure] Be Cautious When...
> - Sample size is small (interactions need more data).
> - Too many interactions lead to overfitting.
> - Main effects are not significant (usually include them anyway).

---

## Theoretical Background

### Without Interaction (Additive Model)

$$
Y = \beta_0 + \beta_1 X + \beta_2 Z
$$

- Effect of X on Y is always $\beta_1$, regardless of Z
- Lines for different Z values are **parallel**

### With Interaction

$$
Y = \beta_0 + \beta_1 X + \beta_2 Z + \beta_3 (X \times Z)
$$

The effect of X on Y is now:
$$
\frac{\partial Y}{\partial X} = \beta_1 + \beta_3 Z
$$

- Effect of X **depends on Z**
- Lines for different Z values **cross** or diverge

### Types of Interaction

| Type | Description | Visual |
|------|-------------|--------|
| **Ordinal (Synergistic)** | Lines diverge but don't cross | Both effects in same direction |
| **Disordinal (Crossover)** | Lines cross | Effect reverses at some point |
| **No Interaction** | Lines are parallel | Additive effects |

---

## Implementation

### Python

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# Generate data with interaction
np.random.seed(42)
n = 200
experience = np.random.uniform(0, 20, n)
education = np.random.choice([0, 1], n)  # 0=No degree, 1=Degree

# True model: education MODERATES effect of experience
salary = (30000 + 
          1000 * experience + 
          20000 * education + 
          1500 * experience * education +  # INTERACTION
          np.random.normal(0, 5000, n))

df = pd.DataFrame({
    'salary': salary,
    'experience': experience,
    'education': education
})

# ========== MODEL WITHOUT INTERACTION ==========
model_no_int = smf.ols('salary ~ experience + education', data=df).fit()
print("=== Model WITHOUT Interaction ===")
print(model_no_int.summary().tables[1])

# ========== MODEL WITH INTERACTION ==========
model_int = smf.ols('salary ~ experience * education', data=df).fit()
print("\n=== Model WITH Interaction ===")
print(model_int.summary().tables[1])

# ========== INTERPRETATION ==========
print("\n=== Interpretation ===")
b = model_int.params
print(f"Effect of experience (no degree): ${b['experience']:.0f}/year")
print(f"Effect of experience (with degree): ${b['experience'] + b['experience:education']:.0f}/year")

# ========== VISUALIZATION ==========
plt.figure(figsize=(10, 6))

for edu in [0, 1]:
    subset = df[df['education'] == edu]
    plt.scatter(subset['experience'], subset['salary'], 
                label=f"{'Degree' if edu else 'No Degree'}")
    
    # Fitted line
    x_line = np.linspace(0, 20, 100)
    if model_int:
        y_line = (b['Intercept'] + b['experience']*x_line + 
                  b['education']*edu + b['experience:education']*x_line*edu)
        plt.plot(x_line, y_line, linestyle='--')

plt.xlabel('Experience (years)')
plt.ylabel('Salary')
plt.legend()
plt.title('Interaction: Education Moderates Experience Effect')
plt.show()
```

### R

```r
# Generate data
set.seed(42)
n <- 200
experience <- runif(n, 0, 20)
education <- sample(c(0, 1), n, replace = TRUE)

salary <- 30000 + 1000*experience + 20000*education + 
          1500*experience*education + rnorm(n, 0, 5000)

df <- data.frame(salary, experience, education = factor(education))

# ========== MODEL WITHOUT INTERACTION ==========
model_no_int <- lm(salary ~ experience + education, data = df)
summary(model_no_int)

# ========== MODEL WITH INTERACTION ==========
model_int <- lm(salary ~ experience * education, data = df)
summary(model_int)

# ========== COMPARE MODELS ==========
anova(model_no_int, model_int)
# If p < 0.05, interaction is significant

# ========== VISUALIZATION ==========
library(ggplot2)
ggplot(df, aes(x = experience, y = salary, color = education)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  labs(title = "Interaction: Non-Parallel Lines",
       x = "Experience", y = "Salary")
```

---

## Interpretation Guide

| Coefficient | Interpretation |
|-------------|----------------|
| $\beta_1$ (main effect X) | Effect of X when Z = 0 |
| $\beta_2$ (main effect Z) | Effect of Z when X = 0 |
| $\beta_3$ (interaction) | How much the effect of X CHANGES for each unit increase in Z |

### Example Interpretation

Model: $\text{Salary} = 30000 + 1000 \times \text{Exp} + 20000 \times \text{Edu} + 1500 \times (\text{Exp} \times \text{Edu})$

| Group | Effect of 1 Year Experience |
|-------|----------------------------|
| No Degree (Edu=0) | $1000 + 1500(0) = \$1,000$ |
| With Degree (Edu=1) | $1000 + 1500(1) = \$2,500$ |

**Interpretation:** Having a degree increases the return on experience by \$1,500/year.

---

## Common Pitfalls

> [!warning] Traps to Avoid
>
> **1. Removing Main Effects When Interaction is Present**
> - Including X×Z without X and Z violates hierarchy
> - Solution: Always include main effects if including interaction
>
> **2. Centering Continuous Variables**
> - Without centering, main effects change meaning
> - Solution: Center continuous predictors before creating interaction
>
> **3. Over-Interpreting Non-Significant Interactions**
> - Interactions need MORE power to detect
> - Solution: Use adequate sample size; consider pre-registration
>
> **4. Multicollinearity**
> - X, Z, and X×Z are often highly correlated
> - Solution: Center variables to reduce correlation

---

## Worked Example

> [!example] Drug Effectiveness by Age
> **Question:** Does a new drug work differently for young vs. old patients?
>
> **Model:** Recovery ~ Drug + Age + Drug×Age
>
> **Results:**
> | Term | Coefficient | p-value |
> |------|-------------|---------|
> | Intercept | 50 | <0.001 |
> | Drug | 15 | <0.001 |
> | Age | -0.5 | 0.02 |
> | Drug×Age | -0.3 | 0.04 |
>
> **Interpretation:**
> - Drug effect for Age=0 (extrapolated): +15 recovery points
> - Drug effect for Age=60: $15 - 0.3(60) = -3$ recovery points
>
> **Conclusion:** Drug helps young patients but may harm elderly patients. The positive main effect is misleading without considering the interaction!

---

## Related Concepts

- [[stats/03_Regression_Analysis/Multiple Linear Regression\|Multiple Linear Regression]] — Base framework
- [[Moderation vs Mediation\|Moderation vs Mediation]] — Related concepts
- [[stats/02_Statistical_Inference/One-Way ANOVA\|ANOVA]] — Factorial interactions
- [[stats/03_Regression_Analysis/VIF (Variance Inflation Factor)\|Multicollinearity]] — Issue with interactions

---

## References

- **Book:** Aiken, L. S., & West, S. G. (1991). *Multiple Regression: Testing and Interpreting Interactions*. Sage. [Sage Link](https://us.sagepub.com/en-us/nam/multiple-regression/book3578)
- **Book:** Cohen, J., & Cohen, P. (1983). *Applied Multiple Regression/Correlation Analysis for the Behavioral Sciences*. Erlbaum. [Publisher Link](https://www.routledge.com/Applied-Multiple-RegressionCorrelation-Analysis-for-the-Behavioral-Sciences/Cohen-Cohen-West-Aiken/p/book/9780805822239)
- **Article:** Brambor, T., Clark, W. R., & Golder, M. (2006). Understanding interaction models: Improving empirical analyses. *Political Analysis*, 14(1), 63-82. [DOI: 10.1093/pan/mpi043](https://doi.org/10.1093/pan/mpi043)
