---
{"dg-publish":true,"permalink":"/stats/04-supervised-learning/hyperparameter-tuning/","tags":["probability","machine-learning","model-selection","optimization"]}
---


## Definition

> [!abstract] Core Statement
> **Hyperparameter Tuning** is the process of ==finding optimal model settings== that aren't learned during training. Unlike model parameters (weights), hyperparameters must be set before training begins.

---

> [!tip] Intuition (ELI5): The Volume Knob
> Training finds the best song to play (model parameters). Hyperparameter tuning finds the best volume, bass, and treble settings (hyperparameters). Both affect the final experience, but they're adjusted differently.

---

## Parameters vs Hyperparameters

| | Parameters | Hyperparameters |
|---|------------|-----------------|
| **Set by** | Training algorithm | Human/Search |
| **Examples** | Weights, biases | Learning rate, tree depth |
| **When set** | During training | Before training |
| **How many** | Millions (deep learning) | Dozens |

---

## Search Strategies

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **Grid Search** | Exhaustive, reproducible | Slow, curse of dimensionality | Few hyperparameters |
| **Random Search** | Faster, covers more space | No guarantees | Initial exploration |
| **Bayesian Optimization** | Smart, efficient | Complex setup | Production tuning |
| **Optuna** | Pruning, visualization | Learning curve | All modern use cases |

---

## Python Implementation

### 1. Grid Search

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid Search with Cross-Validation
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_:.4f}")
```

### 2. Random Search

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Define distributions (not discrete values)
param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20),
    'learning_rate': uniform(0.01, 0.3)  # For gradient boosting
}

random_search = RandomizedSearchCV(
    estimator,
    param_distributions=param_dist,
    n_iter=100,  # Number of random combinations
    cv=5,
    scoring='f1',
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)
```

### 3. Optuna (Bayesian)

```python
import optuna
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier

def objective(trial):
    # Define search space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    
    model = LGBMClassifier(**params, random_state=42, verbose=-1)
    
    # Cross-validation score
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    
    return scores.mean()

# Create study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, show_progress_bar=True)

print(f"Best Parameters: {study.best_params}")
print(f"Best F1 Score: {study.best_value:.4f}")

# ========== VISUALIZATION ==========
optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_param_importances(study)
optuna.visualization.plot_slice(study)
```

---

## R Implementation

```r
library(caret)

# ========== GRID SEARCH ==========
tune_grid <- expand.grid(
  mtry = c(2, 4, 6, 8),
  splitrule = c("gini", "extratrees"),
  min.node.size = c(1, 5, 10)
)

ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

model <- train(
  target ~ .,
  data = train_data,
  method = "ranger",
  tuneGrid = tune_grid,
  trControl = ctrl,
  metric = "ROC"
)

print(model$bestTune)
plot(model)
```

---

## Best Practices

> [!tip] Hyperparameter Tuning Tips
>
> 1. **Start with Random Search** — 60 random trials often beat exhaustive grid search
> 2. **Use Log Scale** — For learning rates, regularization (0.001 to 1.0)
> 3. **Tune in Stages** — First broad search, then narrow around best region
> 4. **Use Early Stopping** — Saves time, often better than fixed iterations
> 5. **Cross-Validate** — Never tune on test set!

---

## Common Hyperparameters by Model

| Model | Key Hyperparameters |
|-------|---------------------|
| **Random Forest** | n_estimators, max_depth, min_samples_split |
| **XGBoost/LightGBM** | learning_rate, max_depth, num_leaves, reg_alpha/lambda |
| **SVM** | C, kernel, gamma |
| **Neural Networks** | learning_rate, batch_size, hidden_layers, dropout |
| **Logistic Regression** | C (regularization), penalty type |

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Tuning on Test Set**
> - *Problem:* Information leakage, overly optimistic results
> - *Solution:* Use cross-validation or separate validation set
>
> **2. Too Many Hyperparameters**
> - *Problem:* Curse of dimensionality, takes forever
> - *Solution:* Focus on most impactful parameters first
>
> **3. Ignoring Computational Budget**
> - *Problem:* Running for days
> - *Solution:* Use Optuna with pruning, set time limits

---

## Related Concepts

- [[stats/04_Supervised_Learning/Cross-Validation\|Cross-Validation]] — Evaluation during tuning
- [[stats/04_Supervised_Learning/Overfitting\|Overfitting]] — What poor tuning causes
- [[stats/04_Supervised_Learning/Learning Curves\|Learning Curves]] — Diagnose over/underfitting
- [[stats/04_Supervised_Learning/LightGBM\|LightGBM]] — Popular model to tune
- [[stats/04_Supervised_Learning/XGBoost\|XGBoost]] — Popular model to tune

---

## References

- **Paper:** Bergstra, J., & Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization. *JMLR*, 13, 281-305.
- **Documentation:** [Optuna](https://optuna.org/)
- **Tutorial:** [Scikit-learn Tuning](https://scikit-learn.org/stable/modules/grid_search.html)
