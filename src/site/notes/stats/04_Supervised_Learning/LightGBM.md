---
{"dg-publish":true,"permalink":"/stats/04-supervised-learning/light-gbm/","tags":["Machine-Learning","Gradient-Boosting","Ensemble"]}
---


## Definition

> [!abstract] Core Statement
> **LightGBM** is a high-performance ==gradient boosting framework== that uses tree-based learning algorithms. It's designed for speed and efficiency through histogram-based splitting and leaf-wise (best-first) tree growth.

---

> [!tip] Intuition (ELI5): The Speed Demon
> XGBoost is like a meticulous worker checking every possibility. LightGBM is the smart worker who groups similar things together (histograms) and focuses on the most promising branches first. Same quality, much faster.

---

## Key Innovations

| Innovation | Description | Benefit |
|------------|-------------|---------|
| **Histogram-based** | Bins continuous features into discrete buckets | 10x faster than exact splits |
| **Leaf-wise Growth** | Grows tree by splitting the leaf with max gain | Better accuracy |
| **GOSS** | Gradient-based One-Side Sampling | Reduces data size |
| **EFB** | Exclusive Feature Bundling | Reduces feature dimensionality |

---

## When to Use

> [!success] Use LightGBM When...
> - **Large datasets** (millions of rows)
> - You need **fast training** with good accuracy
> - **Tabular data** with mixed feature types
> - Kaggle competitions!

> [!failure] Consider Alternatives When...
> - Very small datasets (overfitting risk)
> - Need model interpretability (use simpler models)
> - Highly imbalanced data without proper handling

---

## Python Implementation

```python
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# ========== PREPARE DATA ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# LightGBM Dataset (more efficient)
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# ========== PARAMETERS ==========
params = {
    'objective': 'binary',         # 'multiclass', 'regression'
    'metric': 'binary_logloss',    # 'auc', 'rmse', 'multi_logloss'
    'boosting_type': 'gbdt',       # 'dart', 'goss', 'rf'
    'num_leaves': 31,              # Main complexity param (2^max_depth)
    'max_depth': -1,               # -1 = no limit
    'learning_rate': 0.05,
    'n_estimators': 1000,
    'min_child_samples': 20,       # Min data in leaf
    'subsample': 0.8,              # Row sampling
    'colsample_bytree': 0.8,       # Column sampling
    'reg_alpha': 0.1,              # L1 regularization
    'reg_lambda': 0.1,             # L2 regularization
    'random_state': 42,
    'verbose': -1
}

# ========== TRAIN WITH EARLY STOPPING ==========
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, test_data],
    valid_names=['train', 'valid'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
)

# ========== PREDICT ==========
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

# ========== FEATURE IMPORTANCE ==========
lgb.plot_importance(model, max_num_features=20, importance_type='gain')

# ========== SKLEARN API (ALTERNATIVE) ==========
from lightgbm import LGBMClassifier

clf = LGBMClassifier(**params)
clf.fit(X_train, y_train, 
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50)])
```

---

## R Implementation

```r
library(lightgbm)

# ========== PREPARE DATA ==========
dtrain <- lgb.Dataset(data = as.matrix(X_train), label = y_train)
dtest <- lgb.Dataset(data = as.matrix(X_test), label = y_test, reference = dtrain)

# ========== PARAMETERS ==========
params <- list(
  objective = "binary",
  metric = "binary_logloss",
  num_leaves = 31,
  learning_rate = 0.05,
  feature_fraction = 0.8,
  bagging_fraction = 0.8,
  bagging_freq = 5,
  verbose = -1
)

# ========== TRAIN ==========
model <- lgb.train(
  params = params,
  data = dtrain,
  nrounds = 1000,
  valids = list(train = dtrain, valid = dtest),
  early_stopping_rounds = 50
)

# ========== PREDICT ==========
pred <- predict(model, as.matrix(X_test))

# ========== FEATURE IMPORTANCE ==========
importance <- lgb.importance(model)
lgb.plot.importance(importance, top_n = 20)
```

---

## Hyperparameter Tuning Guide

| Parameter | Typical Range | Effect |
|-----------|---------------|--------|
| `num_leaves` | 20-150 | Higher = more complex, risk overfitting |
| `max_depth` | 3-12 | Limit tree depth |
| `learning_rate` | 0.01-0.3 | Lower = more trees needed, better generalization |
| `min_child_samples` | 10-100 | Higher = more regularization |
| `subsample` | 0.5-1.0 | Row sampling per iteration |
| `colsample_bytree` | 0.5-1.0 | Feature sampling per tree |
| `reg_alpha/lambda` | 0-10 | L1/L2 regularization |

---

## LightGBM vs XGBoost vs CatBoost

| Feature | LightGBM | XGBoost | CatBoost |
|---------|----------|---------|----------|
| **Speed** | ⚡ Fastest | Fast | Moderate |
| **Memory** | Low | High | Moderate |
| **Categorical** | Basic | Manual encoding | ⭐ Native handling |
| **Accuracy** | High | High | High |
| **Overfitting** | Risk on small data | Moderate | Most robust |

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Overfitting with Leaf-wise Growth**
> - *Problem:* `num_leaves` too high on small data
> - *Solution:* Use `max_depth` to limit, increase `min_child_samples`
>
> **2. Ignoring Categorical Features**
> - *Problem:* One-hot encoding creates many sparse features
> - *Solution:* Use `categorical_feature` parameter for native handling
>
> **3. Learning Rate vs Iterations Tradeoff**
> - *Rule:* Lower `learning_rate` + more `n_estimators` = better but slower

---

## Related Concepts

- [[stats/04_Supervised_Learning/XGBoost\|XGBoost]] — Main competitor
- [[stats/04_Supervised_Learning/CatBoost\|CatBoost]] — Best for categorical data
- [[stats/04_Supervised_Learning/Gradient Boosting\|Gradient Boosting]] — Theoretical foundation
- [[stats/04_Supervised_Learning/Hyperparameter Tuning\|Hyperparameter Tuning]] — Optimizing parameters
- [[stats/04_Supervised_Learning/Feature Importance\|Feature Importance]] — Understanding model

---

## References

- **Paper:** Ke, G., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS*. [Paper](https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)
- **Documentation:** [LightGBM Docs](https://lightgbm.readthedocs.io/)
- **Tutorial:** [LightGBM Parameter Tuning](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html)
