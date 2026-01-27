---
{"dg-publish":true,"permalink":"/stats/03-regression-analysis/linear-mixed-models-lmm/","tags":["Regression","Hierarchical-Data","Mixed-Effects","Panel-Data","Repeated-Measures"]}
---


## Definition

> [!abstract] Core Statement
> **Linear Mixed Models (LMM)** are regression models that include both ==fixed effects== (population-level parameters) and ==random effects== (group-level variation). They handle **hierarchical/clustered data** where observations are not independent—such as students within schools, repeated measurements within subjects, or patients within hospitals.

![Hierarchical Data Structure: Units nested within groups](https://commons.wikimedia.org/wiki/Special:FilePath/Hierarchical_Model.svg)

**Intuition (ELI5):** Imagine measuring student test scores. Each school has different baseline performance (some are "better" schools). A regular regression ignores this—treating all students as independent. LMM says: "Hey, students in the same school are similar. Let me account for that by giving each school its own baseline."

---

## Purpose

1.  **Model Clustered Data:** Account for non-independence within groups.
2.  **Repeated Measures:** Analyze longitudinal data with multiple measurements per subject.
3.  **Partial Pooling:** Borrow strength across groups (shrinkage estimation).
4.  **Handle Unbalanced Data:** Works with different sample sizes per group.

---

## When to Use

> [!success] Use LMM When...
> - Data has **hierarchical structure** (students in schools, employees in companies).
> - You have **repeated measurements** on the same subjects.
> - Sample sizes **vary across groups**.
> - You want to **generalize** to new groups (not just those in the sample).
> - [[stats/02_Statistical_Inference/Repeated Measures ANOVA\|Repeated Measures ANOVA]] assumptions (sphericity) are violated.

> [!failure] Do NOT Use When...
> - Data is **truly independent** (use standard regression).
> - You only care about **specific groups** in your sample (use fixed effects).
> - Outcome is **non-normal** (use [[stats/03_Regression_Analysis/Generalized Linear Models (GLM)\|Generalized Linear Models (GLM)]] with random effects: GLMM).
> - Groups have **very few levels** (<5 groups)—random effects poorly estimated.

---

## Theoretical Background

### Model Specification

The general LMM can be written as:

$$
Y_{ij} = \underbrace{\beta_0 + \beta_1 X_{ij} + \dots}_{\text{Fixed Effects}} + \underbrace{u_{0j} + u_{1j} X_{ij} + \dots}_{\text{Random Effects}} + \varepsilon_{ij}
$$

Where:
- $Y_{ij}$ = outcome for observation $i$ in group $j$
- $\beta$ = fixed effects (population averages)
- $u_{0j}$ = random intercept for group $j$ ~ $N(0, \sigma^2_{u0})$
- $u_{1j}$ = random slope for group $j$ ~ $N(0, \sigma^2_{u1})$
- $\varepsilon_{ij}$ = residual error ~ $N(0, \sigma^2)$

### Types of Random Effects

| Type | Notation | Meaning |
|------|----------|---------|
| **Random Intercept** | $(1 | \text{group})$ | Each group has different baseline |
| **Random Slope** | $(X | \text{group})$ | Effect of X varies by group |
| **Both** | $(1 + X | \text{group})$ | Both intercept and slope vary |

### Fixed vs. Random Effects

| Fixed Effects | Random Effects |
|---------------|----------------|
| Specific levels of interest | Levels are sample from population |
| Estimate separate parameter for each | Estimate variance of distribution |
| Treatment groups, Gender | Schools, Patients, Companies |
| Few levels (<10) | Many levels (>5-10) |

### Intraclass Correlation Coefficient (ICC)

$$
\text{ICC} = \frac{\sigma^2_{u0}}{\sigma^2_{u0} + \sigma^2}
$$

- **ICC = 0:** No clustering (observations are independent).
- **ICC = 0.1:** 10% of variance is between groups.
- **ICC > 0.05:** Ignoring clustering leads to inflated Type I error.

### Partial Pooling (Shrinkage)

LMM produces estimates between:
1. **No pooling:** Separate regression for each group (unstable for small groups).
2. **Complete pooling:** Ignore groups (biased if groups differ).
3. **Partial pooling (LMM):** Weighted average—small groups shrink toward overall mean.

---

## Assumptions

- [ ] **Linearity:** Relationship between X and Y is linear (within and between groups).
- [ ] **Normality of Random Effects:** $u_{0j}, u_{1j}$ are normally distributed.
- [ ] **Normality of Residuals:** $\varepsilon_{ij}$ are normally distributed.
- [ ] **Homoscedasticity:** Constant residual variance across groups and fitted values.
- [ ] **Independence of Errors:** Residuals are independent *within* groups (after accounting for random effects).

### Checking Assumptions

1. **Residual plots:** Check for patterns.
2. **Q-Q plots:** For residuals and random effects.
3. **Random effects distribution:** Should be approximately normal.

---

## Limitations

> [!warning] Pitfalls
> 1. **Few Groups:** With <5-10 groups, random effects variance is poorly estimated. Use fixed effects instead.
> 2. **Convergence Issues:** Complex models may fail to converge. Simplify random structure.
> 3. **Interpretation:** Random effect coefficients are BLUPs (predictions), not parameter estimates.
> 4. **Missing Data:** Assumes Missing at Random (MAR). MNAR causes bias.
> 5. **Crossed vs. Nested:** Ensure correct specification of hierarchy.

---

## Python Implementation

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# ========== EXAMPLE DATA ==========
# Students nested within schools
np.random.seed(42)

n_schools = 20
students_per_school = np.random.randint(15, 40, n_schools)
n_total = sum(students_per_school)

# Generate hierarchical data
data = []
for school in range(n_schools):
    n_students = students_per_school[school]
    school_effect = np.random.normal(0, 5)  # Random intercept
    school_slope = np.random.normal(0, 0.3)  # Random slope
    
    for student in range(n_students):
        study_hours = np.random.uniform(1, 10)
        score = (50 + school_effect +  # Random intercept
                (3 + school_slope) * study_hours +  # Fixed + random slope
                np.random.normal(0, 5))  # Residual
        data.append({
            'school': f'School_{school+1}',
            'study_hours': study_hours,
            'score': score
        })

df = pd.DataFrame(data)
print(f"Data shape: {df.shape}")
print(f"Schools: {df['school'].nunique()}")
print(df.head())

# ========== FIT MIXED MODEL ==========
# Random intercept model
model_ri = smf.mixedlm("score ~ study_hours", df, groups=df["school"])
result_ri = model_ri.fit()
print("\n=== Random Intercept Model ===")
print(result_ri.summary())

# Random intercept + random slope model
model_ris = smf.mixedlm("score ~ study_hours", df, 
                         groups=df["school"],
                         re_formula="~study_hours")
result_ris = model_ris.fit()
print("\n=== Random Intercept + Slope Model ===")
print(result_ris.summary())

# ========== CALCULATE ICC ==========
# From random intercept model
var_between = result_ri.cov_re.iloc[0, 0]  # Group variance
var_within = result_ri.scale  # Residual variance
icc = var_between / (var_between + var_within)
print(f"\nICC: {icc:.3f}")
print(f"Interpretation: {icc*100:.1f}% of variance is between schools")

# ========== RANDOM EFFECTS (BLUPs) ==========
random_effects = result_ri.random_effects
re_df = pd.DataFrame({
    'school': random_effects.keys(),
    'random_intercept': [v['Group'] for v in random_effects.values()]
})
print("\n=== Random Effects (Top 5 schools) ===")
print(re_df.head())

# ========== VISUALIZATION ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Data by school
for school in df['school'].unique()[:5]:
    subset = df[df['school'] == school]
    axes[0].scatter(subset['study_hours'], subset['score'], alpha=0.5, label=school)
axes[0].set_xlabel('Study Hours')
axes[0].set_ylabel('Test Score')
axes[0].set_title('Score vs Study Hours (5 Schools)')
axes[0].legend()

# Plot 2: Random effects distribution
axes[1].hist(re_df['random_intercept'], bins=10, edgecolor='black')
axes[1].axvline(0, color='red', linestyle='--')
axes[1].set_xlabel('Random Intercept')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of Random Intercepts')

plt.tight_layout()
plt.show()

# ========== MODEL COMPARISON ==========
# Compare with OLS (ignoring clustering)
import statsmodels.api as sm
ols_model = smf.ols("score ~ study_hours", df).fit()
print("\n=== OLS (ignoring clustering) ===")
print(f"study_hours coef: {ols_model.params['study_hours']:.3f}")
print(f"SE: {ols_model.bse['study_hours']:.3f}")

print("\n=== LMM (accounting for clustering) ===")
print(f"study_hours coef: {result_ri.params['study_hours']:.3f}")
print(f"SE: {result_ri.bse['study_hours']:.3f}")
print("\nNote: LMM SE is typically larger (correct) due to clustering!")
```

---

## R Implementation

```r
library(lme4)
library(lmerTest)  # For p-values
library(performance)  # For ICC
library(broom.mixed)
library(ggplot2)

# ========== EXAMPLE DATA ==========
set.seed(42)

n_schools <- 20
data_list <- list()

for (school in 1:n_schools) {
  n_students <- sample(15:40, 1)
  school_effect <- rnorm(1, 0, 5)
  school_slope <- rnorm(1, 0, 0.3)
  
  data_list[[school]] <- data.frame(
    school = paste0("School_", school),
    study_hours = runif(n_students, 1, 10),
    score = 50 + school_effect + 
            (3 + school_slope) * runif(n_students, 1, 10) + 
            rnorm(n_students, 0, 5)
  )
}

df <- do.call(rbind, data_list)
str(df)

# ========== FIT MIXED MODELS ==========
# Random intercept model
model_ri <- lmer(score ~ study_hours + (1 | school), data = df)
summary(model_ri)

# Random intercept + slope model
model_ris <- lmer(score ~ study_hours + (1 + study_hours | school), data = df)
summary(model_ris)

# ========== CALCULATE ICC ==========
icc_value <- icc(model_ri)
print(icc_value)

# Manual calculation
var_components <- as.data.frame(VarCorr(model_ri))
var_between <- var_components$vcov[1]
var_within <- var_components$vcov[2]
icc_manual <- var_between / (var_between + var_within)
cat("\nManual ICC:", round(icc_manual, 3), "\n")

# ========== FIXED EFFECTS ==========
cat("\n=== Fixed Effects ===\n")
print(fixef(model_ri))

# With confidence intervals
print(confint(model_ri))

# Tidy output
tidy(model_ri, effects = "fixed", conf.int = TRUE)

# ========== RANDOM EFFECTS (BLUPs) ==========
random_effects <- ranef(model_ri)$school
print(head(random_effects))

# Caterpillar plot
dotplot(ranef(model_ri))

# ========== MODEL COMPARISON ==========
# Likelihood ratio test: is random slope needed?
anova(model_ri, model_ris)

# Compare with OLS
model_ols <- lm(score ~ study_hours, data = df)
cat("\n=== Comparison: OLS vs LMM ===\n")
cat("OLS SE:", summary(model_ols)$coefficients["study_hours", "Std. Error"], "\n")
cat("LMM SE:", summary(model_ri)$coefficients["study_hours", "Std. Error"], "\n")

# ========== CHECK ASSUMPTIONS ==========
# Residual plot
plot(model_ri)

# Q-Q plot of residuals
qqnorm(resid(model_ri))
qqline(resid(model_ri))

# Q-Q plot of random effects
qqnorm(ranef(model_ri)$school[,1])
qqline(ranef(model_ri)$school[,1])

# ========== VISUALIZATION ==========
# Spaghetti plot
ggplot(df, aes(x = study_hours, y = score, group = school, color = school)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "lm", se = FALSE, alpha = 0.5) +
  theme_minimal() +
  theme(legend.position = "none") +
  labs(title = "Score vs Study Hours by School",
       subtitle = "Each line = one school")

# Fixed effect with random intercepts
df$fitted <- predict(model_ri)
ggplot(df, aes(x = study_hours, y = fitted, color = school)) +
  geom_line() +
  theme_minimal() +
  theme(legend.position = "none") +
  labs(title = "LMM Predictions by School")
```

---

## Worked Numerical Example

> [!example] Hospital Patient Satisfaction
> **Structure:** 5 patients measured 3 times each in one of 3 hospitals.
> **Outcome:** Satisfaction score (0-100).
> **Predictor:** Wait time (minutes).
> 
> **Data Summary:**
> 
> | Hospital | Mean Satisfaction | N Patients |
> |----------|-------------------|------------|
> | A | 72 | 5 |
> | B | 65 | 5 |
> | C | 58 | 5 |
> 
> **Model:** `Satisfaction ~ WaitTime + (1 | Hospital) + (1 | Patient:Hospital)`
> 
> **Results:**
> 
> | Parameter | Estimate | SE |
> |-----------|----------|-----|
> | Intercept (β₀) | 80.5 | 4.2 |
> | WaitTime (β₁) | -0.45 | 0.08 |
> | σ²_hospital | 52.3 | - |
> | σ²_patient | 28.1 | - |
> | σ² (residual) | 18.4 | - |
> 
> **Interpretation:**
> 
> 1. **Fixed Effect (WaitTime):** Each minute of wait time decreases satisfaction by 0.45 points.
> 
> 2. **Variance Components:**
>    - Hospital-level variance = 52.3 (hospitals differ substantially)
>    - Patient-level variance = 28.1 (patients within hospitals differ)
>    - Residual variance = 18.4 (measurement error)
> 
> 3. **ICC (Hospital):**
>    $$\text{ICC}_{hospital} = \frac{52.3}{52.3 + 28.1 + 18.4} = 0.53$$
>    53% of variance is between hospitals!
> 
> 4. **Random Intercepts:**
>    - Hospital A: +7.2 (above average)
>    - Hospital B: +1.1 (near average)
>    - Hospital C: -8.3 (below average)

---

## Interpretation Guide

| Output | Example | Interpretation | Edge Case |
|--------|---------|----------------|-----------|
| Fixed Effect β | -0.45 | Population-average effect | Very small β may be practically insignificant |
| Random Intercept Variance | 52.3 | Groups differ in baseline | Very small → use OLS |
| Random Slope Variance | 12.1 | Effect of X varies by group | 0 → drop random slope |
| ICC | 0.35 | 35% of variance is between groups | ICC > 0.05 → clustering matters |
| BLUP (Hospital A) | +7.2 | Hospital A is 7.2 units above average | Shrunken toward mean |
| LRT p-value | 0.002 | Random effect significantly improves fit | Compare nested models |

---

## Common Pitfall Example

> [!warning] Ignoring Clustering Inflates Type I Error
> **Scenario:** Analyzing test scores of 1000 students in 50 classrooms.
> 
> **OLS Result (ignoring clustering):**
> - β_treatment = 2.5, SE = 0.3, p = 0.001 ✓ Significant!
> 
> **LMM Result (accounting for classrooms):**
> - β_treatment = 2.4, SE = 0.8, p = 0.12 ✗ Not significant
> 
> **What happened?**
> - OLS assumes 1000 independent observations.
> - Reality: Only ~50 independent clusters (classrooms).
> - Effective sample size is closer to 50 than 1000.
> - OLS underestimates SE → inflates Type I error to 15-25%!
> 
> **Rule:** If ICC > 0.05, always use multilevel models.

---

## Related Concepts

**Prerequisites:**
- [[stats/03_Regression_Analysis/Multiple Linear Regression\|Multiple Linear Regression]] - Fixed effects only
- [[stats/02_Statistical_Inference/Repeated Measures ANOVA\|Repeated Measures ANOVA]] - Simpler repeated measures

**Extensions:**
- [[stats/03_Regression_Analysis/Generalized Linear Models (GLM)\|Generalized Linear Models (GLM)]] - For non-normal outcomes (GLMM)
- [[stats/01_Foundations/Bayesian Statistics\|Bayesian Statistics]] - Bayesian multilevel models

**Applications:**
- [[Panel Data Analysis\|Panel Data Analysis]] - Econometric perspective
- [[Growth Curve Models\|Growth Curve Models]] - Longitudinal trajectories

---

## References

- **Book:** Gelman, A., & Hill, J. (2007). *Data Analysis Using Regression and Multilevel/Hierarchical Models*. Cambridge University Press. (Chapters 11-13) [Cambridge Link](https://www.cambridge.org/9780521686891)
- **Book:** Snijders, T. A. B., & Bosker, R. J. (2012). *Multilevel Analysis* (2nd ed.). SAGE. [Publisher Link](https://uk.sagepub.com/en-gb/eur/multilevel-analysis/book233959)
- **Book:** Pinheiro, J. C., & Bates, D. M. (2000). *Mixed-Effects Models in S and S-PLUS*. Springer. [Springer Link](https://link.springer.com/book/10.1007/b98882)
- **Article:** Bates, D., Mächler, M., Bolker, B., & Walker, S. (2015). Fitting Linear Mixed-Effects Models Using lme4. *Journal of Statistical Software*, 67(1), 1-48. [DOI: 10.18637/jss.v067.i01](https://doi.org/10.18637/jss.v067.i01)
- **Article:** Barr, D. J., Levy, R., Scheepers, C., & Tily, H. J. (2013). Random effects structure for confirmatory hypothesis testing: Keep it maximal. *Journal of Memory and Language*, 68(3), 255-278. [DOI: 10.1016/j.jml.2012.11.001](https://doi.org/10.1016/j.jml.2012.11.001)
