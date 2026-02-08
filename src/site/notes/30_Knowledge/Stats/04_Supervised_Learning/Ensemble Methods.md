---
{"dg-publish":true,"permalink":"/30-knowledge/stats/04-supervised-learning/ensemble-methods/","tags":["machine-learning","supervised"]}
---


## Definition

> [!abstract] Core Statement
> **Ensemble Methods** combine multiple "weak" models into a single "strong" model that achieves better predictive performance than any individual model alone. The key insight: diverse models make different errors, and combining them cancels out individual mistakes.

**Intuition (ELI5):** Imagine asking 100 people to guess the number of jellybeans in a jar. Most individuals will be wrong, but the *average* of all guesses is usually very close to the true number. Ensemble methods apply this "wisdom of the crowd" to machine learning models.

**Core Principle:**
$$
\text{Ensemble Error} < \text{Average Individual Error}
$$
...as long as models are **diverse** and **better than random**.

---

## When to Use

> [!success] Use Ensemble Methods When...
> - You need **maximum predictive accuracy** (Kaggle competitions, production systems).
> - Individual models are **unstable** (high variance, like Decision Trees).
> - You have **sufficient computational resources** (ensembles are expensive).
> - You want to **reduce overfitting** without sacrificing complexity (Bagging).
> - Your base model **underfits** and needs boosting (Boosting).

> [!failure] Do NOT Use Ensemble Methods When...
> - **Interpretability** is critical — ensembles are black boxes.
> - You have **limited computational resources** — training 100+ models is slow.
> - A **simple model** already achieves required performance.
> - You need **real-time predictions** with strict latency constraints.
> - Training data is **very small** — risk of overfitting the ensemble itself.

---

## Theoretical Background

### The Three Main Strategies

| Strategy | Goal | Training | Combination | Key Algorithm |
|----------|------|----------|-------------|---------------|
| **Bagging** | Reduce Variance | Parallel (independent) | Voting / Averaging | [[30_Knowledge/Stats/04_Supervised_Learning/Random Forest\|Random Forest]] |
| **Boosting** | Reduce Bias | Sequential (dependent) | Weighted sum | [[30_Knowledge/Stats/04_Supervised_Learning/Gradient Boosting\|Gradient Boosting]] |
| **Stacking** | Optimize Both | Parallel, then meta-model | Learned weights | Stacked Generalization |

---

### 1. Bagging (Bootstrap Aggregating)

**Mechanism:**
1. Create $B$ bootstrap samples (random sampling with replacement)
2. Train one model on each bootstrap sample (in parallel)
3. Combine predictions: **majority vote** (classification) or **mean** (regression)

$$
\hat{f}_{bag}(x) = \frac{1}{B} \sum_{b=1}^{B} \hat{f}_b(x)
$$

**Why It Works:**
- If individual trees have variance $\sigma^2$, averaging $B$ independent trees reduces variance to $\sigma^2 / B$
- Works best with **high-variance, low-bias** models (e.g., deep Decision Trees)

**Random Forest Enhancement:**
- At each split, consider only a **random subset of features** ($\sqrt{p}$ for classification)
- Further decorrelates trees → more variance reduction

---

### 2. Boosting

**Mechanism:**
1. Train model $M_1$ on full data
2. Identify errors → Give more weight to misclassified samples
3. Train model $M_2$ focusing on errors
4. Repeat → Each model corrects previous mistakes

$$
\hat{f}_{boost}(x) = \sum_{m=1}^{M} \alpha_m \cdot h_m(x)
$$

Where $\alpha_m$ = weight of model $m$ (higher for better models)

**Key Variants:**

| Algorithm | Error Focus | Regularization |
|-----------|-------------|----------------|
| **AdaBoost** | Misclassified samples get more weight | Sample reweighting |
| **Gradient Boosting** | Fit residuals (gradients) | Learning rate, tree constraints |
| **XGBoost** | Gradient + Hessian | L1/L2 regularization, pruning |
| **LightGBM** | Leaf-wise growth | Histogram binning, faster |
| **CatBoost** | Ordered boosting | Handles categoricals natively |

---

### 3. Stacking (Stacked Generalization)

**Mechanism:**
1. Train diverse base models (KNN, SVM, Tree, etc.)
2. Use their **predictions as features** for a meta-model
3. Meta-model learns optimal combination weights

```
Level 0: [KNN, SVM, RF, XGB] → predictions P1, P2, P3, P4
Level 1: Meta-model(P1, P2, P3, P4) → Final prediction
```

**Critical Rule:** Use **out-of-fold predictions** to train meta-model (prevent data leakage)

---

## Assumptions & Diagnostics

- [ ] **Model Diversity:** Ensemble members should make different errors.
- [ ] **Better Than Random:** Each model should be at least 50% accurate (classification).
- [ ] **Independence (Bagging):** Bootstrap samples should create diverse models.
- [ ] **No Overfitting (Boosting):** Monitor validation error — boosting can overfit!

### Diagnostics

| Diagnostic | Purpose | Warning Sign |
|------------|---------|--------------|
| **OOB Score (RF)** | Validation without holdout | OOB << CV score (data leakage) |
| **Learning Curve (Boosting)** | Iterations vs. error | Val error rises while train drops |
| **Feature Importance** | Variable contribution | One feature dominates (potential leak) |
| **Model Correlation** | Ensemble diversity | High correlation = low diversity |

---

## Implementation

### Python

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import (RandomForestClassifier, 
                               GradientBoostingClassifier,
                               StackingClassifier,
                               VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

# Sample data
X, y = make_classification(n_samples=1000, n_features=20, 
                           n_informative=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ========== 1. BAGGING: RANDOM FOREST ==========
rf = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_features='sqrt',   # Features per split
    oob_score=True,        # Out-of-bag validation
    random_state=42
)
rf.fit(X_train, y_train)
print(f"Random Forest OOB Score: {rf.oob_score_:.3f}")
print(f"Random Forest Test Score: {rf.score(X_test, y_test):.3f}")

# ========== 2. BOOSTING: XGBOOST ==========
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,     # Shrinkage (lower = slower but better)
    max_depth=5,           # Tree depth (lower = less overfit)
    subsample=0.8,         # Row sampling
    colsample_bytree=0.8,  # Column sampling
    random_state=42
)
xgb_model.fit(X_train, y_train, 
              eval_set=[(X_test, y_test)],
              verbose=False)
print(f"XGBoost Test Score: {xgb_model.score(X_test, y_test):.3f}")

# ========== 3. VOTING ENSEMBLE ==========
voting = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=50)),
        ('svm', SVC(probability=True)),
        ('knn', KNeighborsClassifier())
    ],
    voting='soft'  # Use probabilities (better than hard voting)
)
voting.fit(X_train, y_train)
print(f"Voting Ensemble Score: {voting.score(X_test, y_test):.3f}")

# ========== 4. STACKING ENSEMBLE ==========
stacking = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=50)),
        ('svm', SVC(probability=True)),
        ('knn', KNeighborsClassifier())
    ],
    final_estimator=LogisticRegression(),  # Meta-model
    cv=5  # Use 5-fold CV predictions for meta-model training
)
stacking.fit(X_train, y_train)
print(f"Stacking Ensemble Score: {stacking.score(X_test, y_test):.3f}")

# ========== 5. FEATURE IMPORTANCE (RF) ==========
import matplotlib.pyplot as plt
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1][:10]

plt.figure(figsize=(10, 5))
plt.bar(range(10), importances[indices])
plt.xticks(range(10), [f'X{i}' for i in indices])
plt.title('Top 10 Feature Importances (Random Forest)')
plt.show()
```

### R

```r
library(randomForest)
library(xgboost)
library(caret)

# Sample data
set.seed(42)
data <- data.frame(matrix(rnorm(1000 * 20), ncol = 20))
data$y <- factor(rbinom(1000, 1, 0.5))

# Train-test split
train_idx <- sample(1:1000, 800)
train <- data[train_idx, ]
test <- data[-train_idx, ]

# ========== 1. RANDOM FOREST ==========
rf <- randomForest(y ~ ., data = train, 
                   ntree = 100,
                   mtry = sqrt(20))  # Features per split

print(rf)
cat("RF Test Accuracy:", mean(predict(rf, test) == test$y), "\n")

# OOB error plot
plot(rf, main = "RF: Error vs. Number of Trees")

# Variable importance
varImpPlot(rf, n.var = 10, main = "Top 10 Variable Importance")

# ========== 2. XGBOOST ==========
train_matrix <- xgb.DMatrix(data = as.matrix(train[, -21]), 
                             label = as.numeric(train$y) - 1)
test_matrix <- xgb.DMatrix(data = as.matrix(test[, -21]),
                            label = as.numeric(test$y) - 1)

xgb_model <- xgb.train(
  params = list(
    objective = "binary:logistic",
    eta = 0.1,           # Learning rate
    max_depth = 5,
    subsample = 0.8,
    colsample_bytree = 0.8
  ),
  data = train_matrix,
  nrounds = 100,
  watchlist = list(train = train_matrix, test = test_matrix),
  verbose = 0
)

# Predictions
xgb_pred <- predict(xgb_model, test_matrix)
xgb_class <- ifelse(xgb_pred > 0.5, 1, 0)
cat("XGBoost Test Accuracy:", mean(xgb_class == (as.numeric(test$y) - 1)), "\n")

# ========== 3. CARET: STACKING ==========
# Define models
ctrl <- trainControl(method = "cv", number = 5, 
                     savePredictions = "final",
                     classProbs = TRUE)

models <- caretList(
  y ~ ., data = train,
  trControl = ctrl,
  methodList = c("rf", "svmRadial", "knn")
)

# Stack with logistic regression
stack <- caretStack(models, method = "glm")
stack_pred <- predict(stack, test)
cat("Stacking Accuracy:", mean(stack_pred == test$y), "\n")
```

---

## Interpretation Guide

| Output | Example Value | Interpretation | Edge Case/Warning |
|--------|---------------|----------------|-------------------|
| **OOB Score (RF)** | 0.85 | 85% accuracy on out-of-bag samples (free validation). | If OOB >> test score, possible data leakage or distribution shift. |
| **n_estimators** | 100 trees | More trees = less variance, more compute. Diminishing returns after ~100. | Very few trees (10) → high variance. 1000+ rarely helps. |
| **learning_rate (Boosting)** | 0.1 | Lower = more trees needed, but better generalization. | Too high (1.0) → overfitting. Too low (0.001) → slow. |
| **Training vs Val curve** | Train:0.99, Val:0.85 | Gap indicates overfitting. | If gap widens with iterations, apply early stopping. |
| **Feature importance** | X1=0.4, X2=0.3, others<0.1 | X1 and X2 dominate predictions. | If one feature >> others, may indicate target leakage. |
| **Stacking score** | 0.88 vs base avg 0.85 | Stacking improved by 3%. | If no improvement, base models may be too similar. |

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Overfitting with Boosting**
> - *Problem:* Adding 1000 boosting iterations without validation.
> - *Result:* Training accuracy 100%, test accuracy 75%.
> - *Solution:* Always use early stopping with validation set.
>
> **2. Low Diversity in Ensemble**
> - *Problem:* Combining 5 Decision Trees with same hyperparameters.
> - *Result:* All models make similar errors → no improvement.
> - *Solution:* Use different algorithms, hyperparameters, or feature subsets.
>
> **3. Data Leakage in Stacking**
> - *Problem:* Training meta-model on same data used for base models.
> - *Result:* Meta-model learns to trust overfitted base predictions.
> - *Solution:* Use out-of-fold (cross-validated) predictions for stacking.
>
> **4. Ignoring Computational Cost**
> - *Problem:* Deploying 500-tree Random Forest for real-time predictions.
> - *Result:* Latency too high, users frustrated.
> - *Solution:* Consider model distillation or single-model surrogates for production.

---

## Worked Numerical Example

> [!example] Bagging Reduces Variance
> **Scenario:** 3 Decision Trees each predict house price.
>
> **Individual Predictions:**
> ```
> Tree 1: $300,000
> Tree 2: $280,000
> Tree 3: $320,000
> 
> True value: $290,000
> ```
>
> **Step 1: Individual Errors**
> ```
> Tree 1 Error: |300K - 290K| = $10,000
> Tree 2 Error: |280K - 290K| = $10,000
> Tree 3 Error: |320K - 290K| = $30,000
> 
> Average individual error: (10K + 10K + 30K) / 3 = $16,667
> ```
>
> **Step 2: Ensemble Prediction (Average)**
> ```
> Ensemble = (300K + 280K + 320K) / 3 = $300,000
> Ensemble Error = |300K - 290K| = $10,000
> ```
>
> **Result:** Ensemble error ($10K) < Average individual error ($16.7K)
>
> **Why?** Tree 3's overestimate (+30K) is partially cancelled by Tree 2's underestimate (-10K).
>
> **Key Insight:** Ensembles work because **individual errors cancel out** — but only if models are **diverse** (make different mistakes).

---

## Comparison: Bagging vs Boosting

| Aspect | Bagging (Random Forest) | Boosting (XGBoost) |
|--------|-------------------------|---------------------|
| **Training** | Parallel (fast) | Sequential (slower) |
| **Goal** | Reduce variance | Reduce bias |
| **Base Model** | High variance (deep trees) | Low variance (shallow trees) |
| **Combining** | Equal average/vote | Weighted by performance |
| **Outliers** | Robust (random samples) | Sensitive (focuses on hard cases) |
| **Overfitting Risk** | Low (hard to overfit) | High (without regularization) |
| **When to Use** | Noisy data, need stability | Clean data, need accuracy |

---

## Related Concepts

**Core Algorithms:**
- [[30_Knowledge/Stats/04_Supervised_Learning/Random Forest\|Random Forest]] — Bagging with Decision Trees
- [[30_Knowledge/Stats/04_Supervised_Learning/Gradient Boosting\|Gradient Boosting]] — Sequential boosting
- [[30_Knowledge/Stats/04_Supervised_Learning/AdaBoost\|AdaBoost]] — Original boosting algorithm

**Prerequisites:**
- [[30_Knowledge/Stats/04_Supervised_Learning/Decision Tree\|Decision Tree]] — Base learner for most ensembles
- [[30_Knowledge/Stats/01_Foundations/Bias-Variance Trade-off\|Bias-Variance Trade-off]] — Theoretical foundation
- [[30_Knowledge/Stats/04_Supervised_Learning/Cross-Validation\|Cross-Validation]] — For stacking and model selection

**Advanced:**
- [[30_Knowledge/Stats/04_Supervised_Learning/Overfitting & Underfitting\|Overfitting & Underfitting]] — What ensembles solve
- [[30_Knowledge/Stats/04_Supervised_Learning/Feature Importance\|Feature Importance]] — Interpreting ensemble contributions

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Dataset is too small for training
> - Interpretability is more important than accuracy

---

## References

- **Historical:** Breiman, L. (1996). Bagging predictors. *Machine Learning*, 24(2), 123-140. [Springer Link](https://link.springer.com/article/10.1007/BF00058655)
- **Historical:** Freund, Y., & Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning and an application to boosting. *Journal of Computer and System Sciences*, 55(1), 119-139. [ScienceDirect](https://doi.org/10.1006/jcss.1997.1504)
- **Book:** Zhou, Z. H. (2012). *Ensemble Methods: Foundations and Algorithms*. CRC Press. [Link](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/ensemble-book.htm)
