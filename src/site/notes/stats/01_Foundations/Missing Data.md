---
{"dg-publish":true,"permalink":"/stats/01-foundations/missing-data/","tags":["Data-Preprocessing","EDA"]}
---


## Definition

> [!abstract] Overview
> **Missing Data** occurs when no data value is stored for the variable in an observation. Handling it correctly is vital because most algorithms (Scikit-Learn) cannot handle NaNs.

---

## 1. Types of Missing Data

Understanding *why* data is missing dictates how to fix it.

1.  **MCAR (Missing Completely At Random):** No pattern. Rolled a die to decide whether to delete data.
    - *Safe to drop rows (if few).*
2.  **MAR (Missing At Random):** Probability of missingness depends on valid observed data (e.g., Women are more likely to leave "Age" blank).
    - *Safe to Impute.*
3.  **MNAR (Missing Not At Random):** Probability depends on the missing value itself (e.g., Rich people hiding their "Income").
    - *Dangerous. Imputation introduces bias.*

---

## 2. Strategies

### Deletion
- **Listwise Deletion:** Drop entire row. (Good if $<5\%$ data missing and MCAR).
- **Pairwise Deletion:** Use available data for specific stats (Correlation).

### Imputation
- **Univariate:** Fill with Mean (Normal data), Median (Skewed data), or Mode (Categorical).
- **Multivariate (Advanced):**
    - **KNN Imputation:** Find 5 similar users and take their average.
    - **MICE (Multivariate Imputation by Chained Equations):** Model missing column as target variable using other columns as features.

---

## 3. Python Implementation

```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer

df = pd.DataFrame({'A': [1, 2, np.nan], 'B': [5, np.nan, 7]})

# 1. Simple Mean Imputation
imputer = SimpleImputer(strategy='mean')
print(imputer.fit_transform(df))

# 2. KNN Imputation
knn_imputer = KNNImputer(n_neighbors=2)
print(knn_imputer.fit_transform(df))
```

---

## Related Concepts

- [[Feature Engineering\|Feature Engineering]]
- [[Bias-Variance Tradeoff\|Bias-Variance Tradeoff]]
- [[Exploratory Data Analysis (EDA)\|Exploratory Data Analysis (EDA)]]
