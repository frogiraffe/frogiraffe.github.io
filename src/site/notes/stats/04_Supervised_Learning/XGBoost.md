---
{"dg-publish":true,"permalink":"/stats/04-supervised-learning/xg-boost/","tags":["Machine-Learning","Gradient-Boosting","Ensemble"]}
---


## Definition

> [!abstract] Core Statement
> **XGBoost** (eXtreme Gradient Boosting) is an optimized implementation of gradient boosting with ==regularization, parallel processing, and built-in handling of missing values==. It dominated Kaggle competitions for years.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Regularization** | L1/L2 on leaf weights prevents overfitting |
| **Parallel Processing** | Fast tree construction |
| **Missing Values** | Learned optimal direction |
| **Sparsity-Aware** | Efficient for sparse data |
| **Tree Pruning** | Depth-first, prunes with gamma |

---

## Python Implementation

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# ========== DMATRIX (NATIVE API) ==========
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'seed': 42
}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=[(dtrain, 'train'), (dtest, 'valid')],
    early_stopping_rounds=50,
    verbose_eval=100
)

# ========== SKLEARN API ==========
from xgboost import XGBClassifier

clf = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], 
        early_stopping_rounds=50, verbose=False)

# ========== FEATURE IMPORTANCE ==========
xgb.plot_importance(model, max_num_features=15)
```

---

## R Implementation

```r
library(xgboost)

dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
dtest <- xgb.DMatrix(data = as.matrix(X_test), label = y_test)

params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  max_depth = 6,
  eta = 0.1
)

model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 1000,
  watchlist = list(train = dtrain, valid = dtest),
  early_stopping_rounds = 50
)
```

---

## Key Hyperparameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `max_depth` | Tree depth | 3-10 |
| `learning_rate` (eta) | Step size | 0.01-0.3 |
| `n_estimators` | Number of trees | 100-1000+ |
| `subsample` | Row sampling | 0.5-1.0 |
| `colsample_bytree` | Feature sampling | 0.5-1.0 |
| `reg_alpha` | L1 regularization | 0-10 |
| `reg_lambda` | L2 regularization | 0-10 |
| `gamma` | Min split loss | 0-5 |

---

## XGBoost vs LightGBM vs CatBoost

| Feature | XGBoost | LightGBM | CatBoost |
|---------|---------|----------|----------|
| **Speed** | Fast | Fastest | Moderate |
| **Categorical** | Manual | Basic | Native |
| **Customization** | Most flexible | Good | Good |
| **GPU Support** | Yes | Yes | Yes |

---

## Related Concepts

- [[stats/04_Supervised_Learning/Gradient Boosting\|Gradient Boosting]] — Theoretical foundation
- [[stats/04_Supervised_Learning/LightGBM\|LightGBM]] — Faster alternative
- [[stats/04_Supervised_Learning/CatBoost\|CatBoost]] — Best for categoricals
- [[stats/04_Supervised_Learning/Hyperparameter Tuning\|Hyperparameter Tuning]] — Essential for XGBoost

---

## References

- **Paper:** Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD*.
- **Documentation:** [XGBoost Docs](https://xgboost.readthedocs.io/)
