---
{"dg-publish":true,"permalink":"/30-knowledge/stats/04-supervised-learning/feature-importance/","tags":["machine-learning","supervised"]}
---


## Definition

> [!abstract] Core Statement
> **Feature Importance** measures ==how much each feature contributes== to model predictions. It helps identify which variables matter most and can guide feature selection and model interpretation.

---

## Types of Feature Importance

| Method | Model-Specific? | Pros | Cons |
|--------|-----------------|------|------|
| **Impurity-based (Gini)** | Trees only | Fast, built-in | Biased toward high-cardinality |
| **Permutation** | Any model | Unbiased, considers interactions | Slow, affected by correlation |
| **SHAP** | Any model | Theoretically grounded | Complex, slow |
| **Coefficients** | Linear models | Fast, interpretable | Only for linear models |
| **Drop-column** | Any model | True importance | Very slow |

---

## Python Implementation

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# ========== FIT MODEL ==========
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# ========== 1. IMPURITY-BASED (MDI) ==========
# Built into tree-based models
mdi_importance = pd.Series(
    rf.feature_importances_, 
    index=feature_names
).sort_values(ascending=False)

print("Top 10 Features (MDI):")
print(mdi_importance.head(10))

# ========== 2. PERMUTATION IMPORTANCE ==========
perm_importance = permutation_importance(
    rf, X_test, y_test, 
    n_repeats=10, 
    random_state=42,
    n_jobs=-1
)

perm_sorted_idx = perm_importance.importances_mean.argsort()[::-1]
print("\nTop 10 Features (Permutation):")
for idx in perm_sorted_idx[:10]:
    print(f"{feature_names[idx]}: {perm_importance.importances_mean[idx]:.4f} "
          f"± {perm_importance.importances_std[idx]:.4f}")

# ========== VISUALIZATION ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# MDI Plot
mdi_importance.head(15).plot(kind='barh', ax=axes[0])
axes[0].set_xlabel('Importance (Gini)')
axes[0].set_title('Impurity-based (MDI)')
axes[0].invert_yaxis()

# Permutation Plot
top_idx = perm_sorted_idx[:15]
axes[1].barh(range(15), perm_importance.importances_mean[top_idx])
axes[1].set_yticks(range(15))
axes[1].set_yticklabels([feature_names[i] for i in top_idx])
axes[1].set_xlabel('Decrease in Accuracy')
axes[1].set_title('Permutation Importance')
axes[1].invert_yaxis()

plt.tight_layout()
plt.show()

# ========== 3. DROP-COLUMN IMPORTANCE ==========
from sklearn.model_selection import cross_val_score

baseline_score = cross_val_score(rf, X, y, cv=5).mean()
drop_importance = {}

for col in feature_names:
    X_dropped = X.drop(columns=[col])
    score = cross_val_score(rf, X_dropped, y, cv=5).mean()
    drop_importance[col] = baseline_score - score

drop_importance = pd.Series(drop_importance).sort_values(ascending=False)
```

---

## R Implementation

```r
library(randomForest)
library(vip)

# ========== FIT MODEL ==========
rf <- randomForest(target ~ ., data = train_data, importance = TRUE)

# ========== BUILT-IN IMPORTANCE ==========
importance(rf)
varImpPlot(rf, main = "Random Forest Feature Importance")

# ========== USING VIP PACKAGE ==========
library(vip)

# Permutation importance
vip(rf, method = "permute", train = train_data, target = "target",
    metric = "accuracy", pred_wrapper = predict)

# Model-specific importance
vip(rf, num_features = 15)
```

---

## Comparison of Methods

> [!example] Same Data, Different Importance
>
> | Feature | MDI (Gini) | Permutation | SHAP |
> |---------|------------|-------------|------|
> | feature_A (high cardinality) | **0.25** | 0.08 | 0.10 |
> | feature_B (correlated with C) | 0.15 | 0.05 | 0.12 |
> | feature_C (truly important) | 0.10 | **0.20** | **0.22** |
>
> **Lesson:** MDI is biased toward high-cardinality features. Permutation and SHAP often agree better on true importance.

---

## When to Use Which Method

| Situation | Recommended Method |
|-----------|-------------------|
| Quick exploration | MDI (built-in) |
| Publication/Stakeholders | Permutation + SHAP |
| Correlated features | SHAP (handles better) |
| Feature selection | Permutation |
| Individual predictions | SHAP |

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. High-Cardinality Bias (MDI)**
> - *Problem:* ID columns, zipcodes artificially high importance
> - *Solution:* Use permutation importance
>
> **2. Correlated Features**
> - *Problem:* Importance split between correlated features
> - *Solution:* Use grouped permutation or SHAP
>
> **3. Overfitting to Training Data**
> - *Problem:* Importance measured on training data
> - *Solution:* Always evaluate on held-out test set

---

## Related Concepts

- [[30_Knowledge/Stats/04_Supervised_Learning/SHAP Values\|SHAP Values]] — Most theoretically grounded method
- [[30_Knowledge/Stats/04_Supervised_Learning/Feature Engineering\|Feature Engineering]] — Creating important features
- [[30_Knowledge/Stats/01_Foundations/Feature Scaling\|Feature Scaling]] — May affect some importance measures
- [[30_Knowledge/Stats/04_Supervised_Learning/Random Forest\|Random Forest]] — Common model with built-in importance

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Dataset is too small for training
> - Interpretability is more important than accuracy

---

## References

- **Paper:** Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
- **Paper:** Altmann, A., et al. (2010). Permutation importance: a corrected feature importance measure. *Bioinformatics*, 26(10), 1340-1347.
- **Package:** [vip R Package](https://koalaverse.github.io/vip/)
