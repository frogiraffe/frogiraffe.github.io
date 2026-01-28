---
{"dg-publish":true,"permalink":"/stats/04-supervised-learning/interpretability/","tags":["Machine-Learning","Explainability","Ethics"]}
---


## Definition

> [!abstract] Core Statement
> **Interpretability** in machine learning refers to the degree to which humans can ==understand and explain model predictions==. It's crucial for trust, debugging, regulatory compliance, and ensuring fairness.

---

> [!tip] Intuition (ELI5): The Glass Box
> A black-box model is like a magician's locked box — you see input and output but not how. Interpretability is like having a glass box — you can see the gears turning.

---

## Types of Interpretability

| Type | Description | Example |
|------|-------------|---------|
| **Inherent** | Model is naturally interpretable | Linear regression, Decision trees |
| **Post-hoc** | Explaining black-box models | SHAP, LIME, Permutation Importance |
| **Global** | Understand overall model behavior | Feature importance, GAMs |
| **Local** | Explain individual predictions | SHAP waterfall, LIME |

---

## Interpretability vs Accuracy Trade-off

```
High Accuracy         ←──────────────────→         High Interpretability

Deep Learning    XGBoost    Random Forest    GAM    Logistic Regression    Linear Regression
     ●              ●             ●           ●              ●                     ●
```

> [!tip] Not Always a Trade-off!
> With proper feature engineering, interpretable models can match complex ones.

---

## Methods Overview

### Inherently Interpretable

| Model | Interpretation |
|-------|----------------|
| **Linear Regression** | Coefficients = marginal effects |
| **Logistic Regression** | Coefficients → odds ratios |
| **Decision Tree** | If-then rules |
| **GAM** | Smooth effect plots per feature |

### Post-hoc Methods

| Method | Scope | Approach |
|--------|-------|----------|
| **[[stats/04_Supervised_Learning/SHAP Values\|SHAP Values]]** | Local + Global | Game-theoretic attribution |
| **LIME** | Local | Local linear approximation |
| **[[stats/04_Supervised_Learning/Feature Importance\|Feature Importance]]** | Global | Permutation or impurity |
| **Partial Dependence** | Global | Marginal effect of features |
| **ICE Plots** | Local + Global | Individual conditional expectation |

---

## Python Examples

```python
# ========== PARTIAL DEPENDENCE PLOT ==========
from sklearn.inspection import PartialDependenceDisplay

fig, ax = plt.subplots(figsize=(10, 6))
PartialDependenceDisplay.from_estimator(
    model, X_train, features=[0, 1, (0, 1)],  # Individual + interaction
    ax=ax
)
plt.show()

# ========== ICE PLOTS ==========
from sklearn.inspection import PartialDependenceDisplay

fig, ax = plt.subplots(figsize=(10, 6))
PartialDependenceDisplay.from_estimator(
    model, X_train, features=[0],
    kind='both',  # PDP + ICE
    ax=ax
)
plt.show()

# ========== LIME ==========
import lime
import lime.lime_tabular

explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=feature_names,
    class_names=['No', 'Yes'],
    mode='classification'
)

# Explain one instance
exp = explainer.explain_instance(X_test.iloc[0].values, model.predict_proba)
exp.show_in_notebook()
```

---

## When Interpretability Matters

| Domain | Requirement | Reason |
|--------|-------------|--------|
| **Healthcare** | Critical | Life-or-death decisions |
| **Finance** | Regulatory | GDPR right to explanation |
| **Criminal Justice** | Ethical | Fairness and bias detection |
| **Autonomous Vehicles** | Safety | Liability and debugging |

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Over-Trusting Simple Models**
> - *Problem:* Interpretable ≠ correct
> - *Solution:* Still validate with held-out data
>
> **2. Cherry-Picking Explanations**
> - *Problem:* Showing only favorable examples
> - *Solution:* Systematic evaluation across data
>
> **3. Confusing Correlation with Causation**
> - *Problem:* Feature importance shows association, not causation
> - *Solution:* Use causal inference methods

---

## Related Concepts

- [[stats/04_Supervised_Learning/SHAP Values\|SHAP Values]] — Gold standard for attribution
- [[stats/04_Supervised_Learning/Feature Importance\|Feature Importance]] — Global importance methods
- [[stats/03_Regression_Analysis/Generalized Additive Models\|Generalized Additive Models]]
 — Inherently interpretable non-linear models

---

## References

- **Book:** Molnar, C. (2022). *Interpretable Machine Learning* (2nd ed.). [Free Online](https://christophm.github.io/interpretable-ml-book/)
- **Paper:** Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. *KDD*.
