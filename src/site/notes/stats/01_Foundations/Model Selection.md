---
{"dg-publish":true,"permalink":"/stats/01-foundations/model-selection/","tags":["Machine-Learning","Methodology","Evaluation"]}
---


## Definition

> [!abstract] Core Statement
> **Model Selection** is the process of choosing the best model from a set of candidates based on performance metrics, complexity, and interpretability. The goal is to select a model that **generalizes well** to unseen data.

**Intuition (ELI5):** You're hiring for a job. You don't just pick the person with the best resume (training performance) — you also interview them (validation) to see how they handle new challenges. Model selection works the same way.

**Key Tradeoffs:**
- **Bias vs Variance:** Simple models underfit, complex models overfit
- **Accuracy vs Interpretability:** Black-box vs white-box
- **Performance vs Speed:** Deep learning vs linear models

---

## When to Use Different Approaches

> [!success] Model Selection Scenarios
> - **Few models:** Manual comparison with CV
> - **Many hyperparameters:** Grid Search or Random Search
> - **Large search space:** Bayesian Optimization
> - **Limited compute:** Random Search > Grid Search

---

## Selection Criteria

### Information Criteria (Penalize Complexity)

| Criterion | Formula | When to Use |
|-----------|---------|-------------|
| **AIC** | $-2\ln(L) + 2k$ | General comparison |
| **BIC** | $-2\ln(L) + k\ln(n)$ | Larger penalty; favors simpler models |
| **Adjusted R²** | $1 - \frac{(1-R^2)(n-1)}{n-k-1}$ | Regression comparison |

Where $L$ = likelihood, $k$ = parameters, $n$ = sample size

**Rule:** Lower AIC/BIC = better model

### Cross-Validation Metrics

| Task | Metric | Goal |
|------|--------|------|
| **Regression** | RMSE, MAE, R² | Lower error, higher R² |
| **Classification** | Accuracy, F1, AUC | Higher is better |
| **Imbalanced** | Precision, Recall, PR-AUC | Depends on cost |

---

## Implementation

### Python

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score
import warnings
warnings.filterwarnings('ignore')

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ========== COMPARE MULTIPLE MODELS ==========
models = {
    'Logistic': LogisticRegression(max_iter=1000),
    'RandomForest': RandomForestClassifier(),
    'SVM': SVC()
}

print("=== 5-Fold Cross-Validation ===")
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"{name}: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")

# ========== GRID SEARCH FOR BEST HYPERPARAMETERS ==========
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV F1: {grid_search.best_score_:.3f}")
print(f"Test F1: {f1_score(y_test, grid_search.predict(X_test)):.3f}")

# ========== AIC/BIC COMPARISON (Statsmodels) ==========
import statsmodels.api as sm

X_const = sm.add_constant(X_train[:, :5])  # First 5 features
model1 = sm.Logit(y_train, X_const[:, :3]).fit(disp=0)  # Simple
model2 = sm.Logit(y_train, X_const).fit(disp=0)         # Complex

print(f"\n=== Information Criteria ===")
print(f"Model 1 (2 features): AIC={model1.aic:.1f}, BIC={model1.bic:.1f}")
print(f"Model 2 (5 features): AIC={model2.aic:.1f}, BIC={model2.bic:.1f}")
```

### R

```r
library(caret)

# Generate data
set.seed(42)
data <- data.frame(matrix(rnorm(1000*20), ncol=20))
data$y <- factor(rbinom(1000, 1, 0.5))

# Train-test split
train_idx <- sample(1:1000, 800)
train <- data[train_idx, ]
test <- data[-train_idx, ]

# ========== COMPARE MODELS WITH CV ==========
ctrl <- trainControl(method = "cv", number = 5)

models <- list(
  logistic = train(y ~ ., data = train, method = "glm", family = "binomial", trControl = ctrl),
  rf = train(y ~ ., data = train, method = "rf", trControl = ctrl),
  svm = train(y ~ ., data = train, method = "svmRadial", trControl = ctrl)
)

# Compare results
results <- resamples(models)
summary(results)
bwplot(results)

# ========== AIC/BIC COMPARISON ==========
model1 <- glm(y ~ V1 + V2, data = train, family = binomial)
model2 <- glm(y ~ V1 + V2 + V3 + V4 + V5, data = train, family = binomial)

cat("\n=== Information Criteria ===\n")
cat("Model 1: AIC =", AIC(model1), ", BIC =", BIC(model1), "\n")
cat("Model 2: AIC =", AIC(model2), ", BIC =", BIC(model2), "\n")

# Likelihood Ratio Test
anova(model1, model2, test = "LRT")
```

---

## Decision Framework

```
Start with problem type:
│
├─ Small data (<1k), need interpretability
│  → Linear/Logistic Regression, Decision Tree
│
├─ Medium data, tabular
│  → XGBoost, Random Forest
│
├─ Large data, unstructured (image/text)
│  → Neural Networks
│
└─ For any choice, use:
   1. Cross-validation to estimate generalization
   2. Compare multiple models
   3. Select simplest model with acceptable performance
```

---

## Common Pitfalls

> [!warning] Traps to Avoid
>
> **1. Selecting Based on Training Performance**
> - Training accuracy is always optimistic
> - Solution: Use CV or holdout validation
>
> **2. Data Leakage in Feature Selection**
> - Selecting features on full data, then CV
> - Solution: Feature selection inside CV loop
>
> **3. Comparing Models on Different Metrics**
> - Model A: optimized for accuracy; Model B: for AUC
> - Solution: Decide on ONE primary metric upfront
>
> **4. Ignoring Model Complexity**
> - 99% accuracy with 1M parameters vs 98% with 100 parameters
> - Solution: Consider Occam's Razor; use AIC/BIC

---

## Model Selection Workflow

> [!example] Step-by-Step Process
> 1. **Define the problem:** Classification? Regression? Ranking?
> 2. **Choose evaluation metric:** Based on business needs
> 3. **Split data:** Train/Validation/Test
> 4. **Try multiple model families:** Start simple
> 5. **Tune hyperparameters:** Grid/Random search with CV
> 6. **Compare with AIC/BIC/CV scores:** Lower is better
> 7. **Final evaluation on test set:** Only once!
> 8. **Consider interpretability vs performance tradeoff**

---

## Related Concepts

- [[stats/04_Supervised_Learning/Cross-Validation\|Cross-Validation]] — Estimating generalization
- [[stats/04_Supervised_Learning/Overfitting\|Overfitting]] — Why we need validation
- [[stats/01_Foundations/Bias-Variance Trade-off\|Bias-Variance Trade-off]] — Model complexity
- [[Hyperparameter Tuning\|Hyperparameter Tuning]] — Optimizing within model

---

## References

- **Book:** Burnham, K. P., & Anderson, D. R. (2002). *Model Selection and Multimodel Inference* (2nd ed.). Springer. [Springer Link](https://link.springer.com/book/10.1007/b97639)
- **Book:** James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.). Springer. [ISLR Site](https://www.statlearning.com/)
- **Historical:** Akaike, H. (1974). A new look at the statistical model identification. *IEEE Transactions on Automatic Control*, 19(6), 716-723. [IEEE Xplore](https://ieeexplore.ieee.org/document/1100705)
