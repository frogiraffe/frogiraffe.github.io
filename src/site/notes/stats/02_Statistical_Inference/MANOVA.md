---
{"dg-publish":true,"permalink":"/stats/02-statistical-inference/manova/","tags":["Multivariate","Hypothesis-Testing","ANOVA"]}
---

## Overview

> [!abstract] Definition
> **MANOVA (Multivariate Analysis of Variance)** is an extension of ANOVA that tests for the difference in two or more vectors of means. It is used when there are **multiple dependent variables** that are correlated.

---

## 1. Why Not Multiple ANOVAs?

Running separate ANOVAs for each dependent variable causes two problems:
1. **Type I Error Inflation:** Multiple testing problem.
2. **Ignoring Correlations:** Separate ANOVAs ignore the relationship between dependent variables. MANOVA might detect a difference in the *joint* distribution that individual ANOVAs miss.

---

## 2. Hypotheses

- $H_0$: The mean vectors across groups are equal.
  $$ \mathbf{\mu}_1 = \mathbf{\mu}_2 = \dots = \mathbf{\mu}_k $$
- $H_1$: At least one group differs on at least one dependent variable.

---

## Worked Example: Plant Growth

> [!example] Problem
> You test 3 fertilizers (A, B, C) on plants.
> **Outcomes (DVs):**
> 1.  **Height** (cm)
> 2.  **Weight** (g)
> 
> If you run two separate ANOVAs:
> -   Height: $p=0.06$ (Not Sig).
> -   Weight: $p=0.06$ (Not Sig).
> 
> **Run MANOVA:**
> -   Plants with Fertilizer A are **Tall and Skinny**.
> -   Plants with Fertilizer B are **Short and Fat**.
> -   Plants with Fertilizer C are **Tall and Fat**.
> 
> **MANOVA Result:** $p < 0.01$ (Significant!).
> **Reasoning:** Even though the *marginal* distributions (Height alone, Weight alone) overlap, the *joint* clusters in 2D space are distinct. MANOVA captures this correlation.

---

## 3. Test Statistics

Since we are comparing matrices (Variance-Covariance matrices), there is no single F-value. Common statistics include:
1.  **Wilks' Lambda:** Most common. Ratio of determinants ($|\mathbf{W}| / |\mathbf{T}|$). Values close to 0 indicate groups are distinct.
2.  **Pillai's Trace:** Consider the most **robust** to violations (e.g., small sample, unequal variance). Use this if assumptions are shaky.
3.  **Hotelling-Lawley Trace:** Good for two groups (generalizes T-test).
4.  **Roy's Largest Root:** Upper bound; most powerful if difference is on only one dimension, but sensitive to assumptions.

---

## 4. Assumptions

1. **Multivariate Normality:** Dependent variables should be normally distributed within groups.
2. **Homogeneity of Covariance Matrices:** Box's M Test (sensitive).
3. **Linearity:** Linear relationships between DVs.
4. **No Multicollinearity:** DVs shouldn't be too highly correlated ($r > 0.9$).

---

## 5. Python Implementation Example

```python
from statsmodels.multivariate.manova import MANOVA
import pandas as pd

# Formula interface: DV1 + DV2 ~ Group
model = MANOVA.from_formula('Sepal_Length + Sepal_Width + Petal_Length ~ Species', data=df)
print(model.mv_test())
```

---

## 6. Related Concepts

- [[stats/02_Statistical_Inference/One-Way ANOVA\|One-Way ANOVA]] - Univariate version.
- [[stats/02_Statistical_Inference/Hotelling's T-Squared\|Hotelling's T-Squared]] - Two-group multivariate test (Multivariate t-test).
- [[Principal Component Analysis (PCA)\|Principal Component Analysis (PCA)]] - Can be used to reduce DVs before analysis.

---

## References

- **Book:** Tabachnick, B. G., & Fidell, L. S. (2019). *Using Multivariate Statistics* (7th ed.). Pearson. [Pearson Link](https://www.pearson.com/en-us/subject-catalog/p/using-multivariate-statistics/P200000006275/)
- **Book:** Johnson, R. A., & Wichern, D. W. (2007). *Applied Multivariate Statistical Analysis* (6th ed.). Pearson. [Pearson Link](https://www.pearson.com/en-us/subject-catalog/p/applied-multivariate-statistical-analysis/P200000006276/)
- **Book:** Hair, J. F., et al. (2018). *Multivariate Data Analysis* (8th ed.). Cengage. [Cengage Link](https://www.cengage.com/c/multivariate-data-analysis-8e-hair/9781473756540/)
