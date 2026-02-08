---
{"dg-publish":true,"permalink":"/30-knowledge/stats/04-supervised-learning/cat-boost/","tags":["machine-learning","supervised"]}
---


## Definition

> [!abstract] Core Statement
> **CatBoost** is a gradient boosting library with ==native handling of categorical features==. It uses ordered boosting and symmetric trees to prevent prediction shift and overfitting.

---

> [!tip] Intuition (ELI5): The Smart Category Handler
> When you have categories like "red/blue/green", other algorithms need you to convert them to numbers first. CatBoost says "give me the raw categories, I'll figure out the smartest way to use them."

---

## Key Innovations

| Innovation | Description | Benefit |
|------------|-------------|---------|
| **Ordered Boosting** | Uses random permutations to compute gradients | Prevents target leakage |
| **Symmetric Trees** | Same split at each level | Faster inference |
| **Native Categoricals** | Handles cats without encoding | Better accuracy |
| **Ordered Target Encoding** | Smart categorical encoding | No leakage |

---

## When to Use CatBoost

> [!success] Use CatBoost When...
> - Data has **many categorical features**
> - You want **minimal preprocessing**
> - Need **good default performance**
> - Want **robust to overfitting**

> [!failure] Consider Alternatives When...
> - Need fastest training → LightGBM
> - Very large dataset → LightGBM
> - Need maximum customization → XGBoost

---

## Python Implementation

```python
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
import numpy as np

# ========== IDENTIFY CATEGORICAL COLUMNS ==========
cat_features = ['category_1', 'category_2', 'city', 'product_type']
cat_indices = [X.columns.get_loc(c) for c in cat_features]

# ========== CREATE POOL (CATBOOST DATA FORMAT) ==========
train_pool = Pool(
    data=X_train, 
    label=y_train, 
    cat_features=cat_indices
)
test_pool = Pool(
    data=X_test, 
    label=y_test, 
    cat_features=cat_indices
)

# ========== FIT MODEL ==========
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,                    # Tree depth
    l2_leaf_reg=3,              # L2 regularization
    loss_function='Logloss',    # 'MultiClass', 'RMSE' for regression
    cat_features=cat_indices,
    early_stopping_rounds=50,
    verbose=100,
    random_seed=42
)

model.fit(
    train_pool,
    eval_set=test_pool,
    plot=True  # Jupyter visualization
)

# ========== PREDICTIONS ==========
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# ========== FEATURE IMPORTANCE ==========
feature_importance = model.get_feature_importance(train_pool)
feature_names = X.columns

import pandas as pd
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(importance_df.head(15))

# ========== SHAP VALUES ==========
shap_values = model.get_feature_importance(
    train_pool, 
    type='ShapValues'
)

# ========== SAVE AND LOAD ==========
model.save_model('catboost_model.cbm')
loaded_model = CatBoostClassifier()
loaded_model.load_model('catboost_model.cbm')
```

---

## R Implementation

```r
library(catboost)

# ========== PREPARE DATA ==========
train_pool <- catboost.load_pool(
  data = X_train,
  label = y_train,
  cat_features = c("category_1", "category_2")
)

test_pool <- catboost.load_pool(
  data = X_test,
  label = y_test,
  cat_features = c("category_1", "category_2")
)

# ========== PARAMETERS ==========
params <- list(
  iterations = 1000,
  learning_rate = 0.05,
  depth = 6,
  loss_function = "Logloss",
  eval_metric = "AUC",
  random_seed = 42
)

# ========== TRAIN ==========
model <- catboost.train(
  learn_pool = train_pool,
  test_pool = test_pool,
  params = params
)

# ========== PREDICT ==========
predictions <- catboost.predict(model, test_pool, prediction_type = "Probability")
```

---

## Key Hyperparameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `iterations` | Number of boosting rounds | 500-3000 |
| `learning_rate` | Step size shrinkage | 0.01-0.3 |
| `depth` | Tree depth | 4-10 |
| `l2_leaf_reg` | L2 regularization | 1-10 |
| `border_count` | Splits for numerical features | 32-255 |
| `bagging_temperature` | Bootstrap variance control | 0-1 |

---

## CatBoost vs XGBoost vs LightGBM

| Feature | CatBoost | XGBoost | LightGBM |
|---------|----------|---------|----------|
| **Categorical Handling** | ⭐ Native | Manual encoding | Basic |
| **Default Performance** | ⭐ Best | Good | Good |
| **Training Speed** | Moderate | Moderate | ⭐ Fastest |
| **Overfitting Resistance** | ⭐ Best | Good | Risk on small data |
| **GPU Support** | Yes | Yes | Yes |

---

## Categorical Encoding in CatBoost

CatBoost uses **Ordered Target Statistics**:

1. Randomly order training data
2. For each sample, calculate target mean of **preceding** samples with same category
3. Add regularization to prevent overfitting

This prevents target leakage that occurs with standard target encoding!

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Forgetting to Specify Cat Features**
> - *Problem:* Treats categories as numbers → nonsense splits
> - *Solution:* Always pass `cat_features` parameter
>
> **2. Too Many Iterations Without Early Stopping**
> - *Problem:* Overfitting
> - *Solution:* Use `early_stopping_rounds=50-100`
>
> **3. Mixed Types in Categorical Columns**
> - *Problem:* CatBoost errors with NaN + strings
> - *Solution:* Convert NaN to string "missing" first

---

## Related Concepts

- [[30_Knowledge/Stats/04_Supervised_Learning/LightGBM\|LightGBM]] — Speed-focused alternative
- [[30_Knowledge/Stats/04_Supervised_Learning/XGBoost\|XGBoost]] — Most customizable
- [[30_Knowledge/Stats/04_Supervised_Learning/Gradient Boosting\|Gradient Boosting]] — Theoretical foundation
- [[30_Knowledge/Stats/04_Supervised_Learning/Encoding Categorical Variables\|Encoding Categorical Variables]] — Manual alternative

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Dataset is too small for training
> - Interpretability is more important than accuracy

---

## References

- **Paper:** Prokhorenkova, L., et al. (2018). CatBoost: unbiased boosting with categorical features. *NeurIPS*. [Paper](https://papers.nips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html)
- **Documentation:** [CatBoost Docs](https://catboost.ai/docs/)
- **Tutorial:** [CatBoost Parameter Tuning](https://catboost.ai/docs/concepts/parameter-tuning.html)
