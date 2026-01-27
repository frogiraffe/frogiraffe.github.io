---
{"dg-publish":true,"permalink":"/stats/01-foundations/feature-selection/","tags":["Machine-Learning","Modeling","Dimensionality-Reduction"]}
---


## Definition

> [!abstract] Core Statement
> **Feature Selection** is the process of ==selecting a subset of relevant features== to improve model performance, reduce overfitting, and enhance interpretability.

---

## Categories

| Approach | Method | Examples |
|----------|--------|----------|
| **Filter** | Statistical tests | Correlation, χ², mutual info |
| **Wrapper** | Model evaluation | Forward/backward selection, RFE |
| **Embedded** | Built into training | Lasso, tree importance |

---

## Python Implementation

```python
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import Lasso

# Filter: ANOVA F-test
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Wrapper: Recursive Feature Elimination
from sklearn.ensemble import RandomForestClassifier
rfe = RFE(RandomForestClassifier(), n_features_to_select=10)
X_rfe = rfe.fit_transform(X, y)

# Embedded: Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X, y)
important = np.abs(lasso.coef_) > 0
```

---

## R Implementation

```r
library(caret)

# RFE with random forest
control <- rfeControl(functions = rfFuncs, method = "cv", number = 5)
results <- rfe(X, y, sizes = c(5, 10, 15), rfeControl = control)
predictors(results)
```

---

## Related Concepts

- [[stats/03_Regression_Analysis/Lasso Regression\|Lasso Regression]] - L1 regularization selects features
- [[Principal Component Analysis (PCA)\|Principal Component Analysis (PCA)]] - Feature extraction (not selection)
- [[stats/01_Foundations/Data Leakage\|Data Leakage]] - Feature selection must be inside CV

---

## References

- **Article:** Guyon, I., & Elisseeff, A. (2003). An introduction to variable and feature selection. *Journal of Machine Learning Research*, 3, 1157-1182. [JMLR Link](https://jmlr.org/papers/v3/guyon03a.html)
- **Book:** Kuhn, M., & Johnson, K. (2013). *Applied Predictive Modeling*. Springer. [Springer Link](https://link.springer.com/book/10.1007/978-1-4614-6849-3)
