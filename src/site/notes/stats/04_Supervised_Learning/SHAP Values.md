---
{"dg-publish":true,"permalink":"/stats/04-supervised-learning/shap-values/","tags":["probability","interpretability","machine-learning","explainability"]}
---


## Definition

> [!abstract] Core Statement
> **SHAP (SHapley Additive exPlanations)** values explain individual predictions by assigning ==each feature a contribution== to the difference between the prediction and the average prediction. Based on game-theoretic Shapley values.

---

> [!tip] Intuition (ELI5): The Fair Split
> Imagine your model is a team project that got 90 points (when the average is 70). SHAP asks: "How much did each teammate (feature) contribute to those extra 20 points?" It fairly distributes credit, considering all possible team combinations.

---

## Why SHAP?

| Property | Meaning |
|----------|---------|
| **Local Accuracy** | SHAP values sum to the difference between prediction and mean |
| **Consistency** | If a feature's contribution increases, so does its SHAP value |
| **Missingness** | Features not in model have SHAP value = 0 |

---

## Types of SHAP Explainers

| Explainer | Best For | Speed |
|-----------|----------|-------|
| **TreeExplainer** | Tree models (XGBoost, LightGBM, RF) | Fast |
| **DeepExplainer** | Neural networks | Moderate |
| **KernelExplainer** | Any model (model-agnostic) | Slow |
| **LinearExplainer** | Linear models | Fast |

---

## Python Implementation

```python
import shap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ========== TRAIN MODEL ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ========== CREATE EXPLAINER ==========
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# For binary classification, shap_values is a list [class_0, class_1]
# Use shap_values[1] for positive class

# ========== SUMMARY PLOT (GLOBAL) ==========
shap.summary_plot(shap_values[1], X_test, feature_names=feature_names)

# ========== BAR PLOT (FEATURE IMPORTANCE) ==========
shap.summary_plot(shap_values[1], X_test, plot_type="bar")

# ========== FORCE PLOT (SINGLE PREDICTION) ==========
# Explain first prediction
shap.initjs()
shap.force_plot(
    explainer.expected_value[1],
    shap_values[1][0],
    X_test.iloc[0],
    feature_names=feature_names
)

# ========== WATERFALL PLOT (SINGLE PREDICTION) ==========
shap.plots.waterfall(shap.Explanation(
    values=shap_values[1][0],
    base_values=explainer.expected_value[1],
    data=X_test.iloc[0],
    feature_names=feature_names
))

# ========== DEPENDENCE PLOT ==========
shap.dependence_plot("feature_name", shap_values[1], X_test)

# ========== FOR LIGHTGBM/XGBOOST ==========
import lightgbm as lgb
model_lgb = lgb.LGBMClassifier().fit(X_train, y_train)

explainer_lgb = shap.TreeExplainer(model_lgb)
shap_values_lgb = explainer_lgb.shap_values(X_test)
shap.summary_plot(shap_values_lgb[1], X_test)
```

---

## Key Visualizations

### 1. Summary Plot
Shows feature impact across all predictions:
- X-axis: SHAP value (impact on prediction)
- Y-axis: Features (sorted by importance)
- Color: Feature value (red=high, blue=low)

### 2. Force Plot
Single prediction explained:
- Red bars push prediction higher
- Blue bars push prediction lower
- Base value = average prediction

### 3. Waterfall Plot
Step-by-step breakdown of one prediction from base value to final output.

### 4. Dependence Plot
SHAP value vs feature value, reveals:
- Non-linear relationships
- Interactions (via coloring by another feature)

---

## Interpretation Example

> [!example] Loan Default Prediction
>
> **Model:** Predicts 75% probability of default (average is 30%)
>
> | Feature | Value | SHAP Value |
> |---------|-------|------------|
> | Income | $25,000 | +0.20 |
> | Credit Score | 580 | +0.15 |
> | Debt-to-Income | 45% | +0.08 |
> | Employment Years | 1 | +0.02 |
>
> **Interpretation:**
> - Low income (+20% probability) is the biggest risk factor
> - Low credit score (+15%) is second
> - All factors combined push prediction from 30% to 75%

---

## R Implementation

```r
library(shapviz)
library(xgboost)

# Train XGBoost model
dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
model <- xgboost(data = dtrain, nrounds = 100, objective = "binary:logistic")

# Create SHAP explainer
shp <- shapviz(model, X = as.matrix(X_test))

# ========== PLOTS ==========
sv_importance(shp, kind = "beeswarm")  # Summary plot
sv_importance(shp, kind = "bar")       # Bar plot
sv_waterfall(shp, row_id = 1)          # Single prediction
sv_dependence(shp, v = "feature_name") # Dependence plot
```

---

## SHAP vs LIME vs Permutation Importance

| Method | Scope | Consistency | Speed |
|--------|-------|-------------|-------|
| **SHAP** | Local + Global | Theoretically grounded | Moderate |
| **LIME** | Local only | Approximation | Fast |
| **Permutation** | Global only | Empirical | Slow |

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Correlated Features**
> - *Problem:* SHAP splits credit among correlated features
> - *Interpretation:* "Feature A has low importance" might mean "shared with B"
>
> **2. KernelSHAP for Tree Models**
> - *Problem:* Slow and approximate
> - *Solution:* Always use TreeExplainer for tree-based models
>
> **3. Confusing SHAP with Causality**
> - *Problem:* High SHAP ≠ causal importance
> - *Reality:* SHAP shows association, not causation

---

## Related Concepts

- [[stats/04_Supervised_Learning/Feature Importance\|Feature Importance]] — Alternative approaches
- [[stats/04_Supervised_Learning/LightGBM\|LightGBM]] — Common model to explain
- [[stats/04_Supervised_Learning/XGBoost\|XGBoost]] — Common model to explain
- [[stats/04_Supervised_Learning/Interpretability\|Interpretability]] — Broader concept

---

## References

- **Paper:** Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*. [Paper](https://papers.nips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)
- **Documentation:** [SHAP Docs](https://shap.readthedocs.io/)
- **Visual Guide:** [SHAP Tutorial](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html)
