---
{"dg-publish":true,"permalink":"/stats/01-foundations/factor-analysis-efa-and-cfa/","tags":["Multivariate","Psychometrics"]}
---

## Overview

> [!abstract] Definition
> **Factor Analysis** is a statistical method used to describe variability among observed, correlated variables in terms of a potentially lower number of unobserved variables called **factors**.

---

## 1. PCA vs. Factor Analysis

Though often confused, they have different goals:

| Feature | PCA | Factor Analysis (EFA) |
|---------|-----|-----------------------|
| **Goal** | Data Reduction | Identifying Latent Structures |
| **Model** | Linear combination of variables | $X = \mu + \Lambda F + e$ (Measurement Model) |
| **Variance** | Analyzes **Total** variance | Analyzes **Shared** (Common) variance |
| **Use Case** | Image compression, Pre-processing | Psychometrics, Survey validation |

---

## 2. Exploratory vs. Confirmatory

### Exploratory Factor Analysis (EFA)
- **Goal:** Discover the underlying structure of a relatively large set of variables.
- **Process:** No prior hypothesis about which items belong to which factor. The data dictates the structure.
- **Interpretation:** Loadings matrix is rotated (e.g., Varimax rotation) to make interpretation easier.

### Confirmatory Factor Analysis (CFA)
- **Goal:** Verify if data fits a proposed measurement model.
- **Process:** You specify exactly which items measure which factor based on theory.
- **Evaluation:** Goodness-of-fit indices (RMSEA, CFI, TLI).

---

## 3. Python Implementation Example (EFA)

*Note: `factor_analyzer` is the standard library for EFA in Python.*

```python
from factor_analyzer import FactorAnalyzer
import pandas as pd

# 1. Adequacy Test (Bartlett's & KMO)
from factor_analyzer.factor_analyzer import calculate_kmo
kmo_all, kmo_model = calculate_kmo(df)
print(f"KMO Score: {kmo_model:.2f}") # Should be > 0.6

# 2. Fit EFA
fa = FactorAnalyzer(n_factors=3, rotation='varimax')
fa.fit(df)

# 3. Loadings
loadings = pd.DataFrame(fa.loadings_, index=df.columns)
print(loadings)

# High loading (>0.4) indicates variable belongs to that factor
```

---

## 4. Related Concepts

- [[stats/05_Unsupervised_Learning/PCA (Principal Component Analysis)\|PCA (Principal Component Analysis)]] - Often used as a preliminary step.
- [[stats/01_Foundations/Structural Equation Modeling (SEM)\|Structural Equation Modeling (SEM)]] - Extension of CFA.
- [[stats/01_Foundations/Cronbach's Alpha\|Cronbach's Alpha]] - Reliability metric often paired with FA.

---

## References

- **Historical:** Spearman, C. (1904). "General Intelligence," Objectively Determined and Measured. *The American Journal of Psychology*, 15(2), 201-292. [JSTOR](https://doi.org/10.2307/1412107)
- **Book:** Gorsuch, R. L. (1983). *Factor Analysis* (2nd ed.). Routledge. [Routledge Link](https://www.routledge.com/Factor-Analysis/Gorsuch/p/book/9780805801088)
- **Book:** Brown, T. A. (2015). *Confirmatory Factor Analysis for Applied Research* (2nd ed.). Guilford Publications. [Guilford Press](https://www.guilford.com/books/Confirmatory-Factor-Analysis-for-Applied-Research/Timothy-Brown/9781462515363)
