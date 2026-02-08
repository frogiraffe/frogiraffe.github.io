---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/factor-analysis-efa-and-cfa/","tags":["probability","foundations"]}
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

- [[30_Knowledge/Stats/05_Unsupervised_Learning/PCA (Principal Component Analysis)\|PCA (Principal Component Analysis)]] - Often used as a preliminary step.
- [[30_Knowledge/Stats/01_Foundations/Structural Equation Modeling (SEM)\|Structural Equation Modeling (SEM)]] - Extension of CFA.
- [[30_Knowledge/Stats/01_Foundations/Cronbach's Alpha\|Cronbach's Alpha]] - Reliability metric often paired with FA.

---

## Definition

> [!abstract] Core Statement
> **Factor Analysis (EFA & CFA)** ... Refer to standard documentation

---

> [!tip] Intuition (ELI5)
> Refer to standard documentation

---

## When to Use

> [!success] Use Factor Analysis (EFA & CFA) When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## Python Implementation

```python
import numpy as np
import pandas as pd

# Example implementation of Factor Analysis (EFA & CFA)
# See documentation for details

data = np.random.randn(100)
print(f"Mean: {np.mean(data):.3f}")
print(f"Std: {np.std(data):.3f}")
```

---

## R Implementation

```r
# Factor Analysis (EFA & CFA) in R
set.seed(42)

# Example implementation
data <- rnorm(100)
summary(data)
```

---

## Related Concepts

- [[30_Knowledge/Stats/01_Foundations/Normal Distribution\|Normal Distribution]]
- [[30_Knowledge/Stats/01_Foundations/Central Limit Theorem (CLT)\|Central Limit Theorem (CLT)]]
- [[30_Knowledge/Stats/01_Foundations/Variance\|Variance]]

---

## References

- **Historical:** Spearman, C. (1904). "General Intelligence," Objectively Determined and Measured. *The American Journal of Psychology*, 15(2), 201-292. [JSTOR](https://doi.org/10.2307/1412107)
- **Book:** Gorsuch, R. L. (1983). *Factor Analysis* (2nd ed.). Routledge. [Routledge Link](https://www.routledge.com/Factor-Analysis/Gorsuch/p/book/9780805801088)
- **Book:** Brown, T. A. (2015). *Confirmatory Factor Analysis for Applied Research* (2nd ed.). Guilford Publications. [Guilford Press](https://www.guilford.com/books/Confirmatory-Factor-Analysis-for-Applied-Research/Timothy-Brown/9781462515363)
