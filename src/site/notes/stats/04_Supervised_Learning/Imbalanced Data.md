---
{"dg-publish":true,"permalink":"/stats/04-supervised-learning/imbalanced-data/","tags":["Machine-Learning","Classification","Data-Quality"]}
---


## Definition

> [!abstract] Core Statement
> **Imbalanced Data** occurs when class distributions are ==significantly skewed==, with one or more classes heavily outnumbered. Standard algorithms tend to favor the majority class, leading to poor performance on minority classes.

---

> [!tip] Intuition (ELI5): The Rare Disease
> Imagine predicting a disease that affects 1 in 1000 people. A model that always says "no disease" is 99.9% accurate — but completely useless! Imbalanced data requires special handling to catch those rare but important cases.

---

## Why It Matters

| Metric | Problem with Imbalanced Data |
|--------|------------------------------|
| **Accuracy** | Misleadingly high (99% by predicting majority always) |
| **Model learning** | Algorithm ignores minority class |
| **Business impact** | Missing rare events (fraud, defects, diseases) |

---

## Strategies

### 1. Resampling Techniques

| Method | Type | Description |
|--------|------|-------------|
| **Random Oversampling** | Oversample | Duplicate minority samples |
| **Random Undersampling** | Undersample | Remove majority samples |
| **SMOTE** | Synthetic | Create synthetic minority samples |
| **SMOTE + Tomek Links** | Hybrid | Synthetic + clean boundaries |
| **ADASYN** | Adaptive | Focus on hard-to-learn samples |

### 2. Algorithm-Level Solutions

| Method | Description |
|--------|-------------|
| **Class Weights** | Higher weight for minority class loss |
| **Threshold Moving** | Adjust decision threshold from 0.5 |
| **Cost-Sensitive Learning** | Explicit misclassification costs |

### 3. Ensemble Methods

- **BalancedRandomForest**
- **EasyEnsemble**
- **RUSBoost**

---

## Python Implementation

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import numpy as np

# ========== CREATE IMBALANCED DATA ==========
X, y = make_classification(n_samples=10000, n_features=20, 
                           weights=[0.95, 0.05], random_state=42)
print(f"Class distribution: {np.bincount(y)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                     stratify=y, random_state=42)

# ========== BASELINE (NO TREATMENT) ==========
clf_base = LogisticRegression(max_iter=1000)
clf_base.fit(X_train, y_train)
y_pred_base = clf_base.predict(X_test)
print(f"\nBaseline F1 (minority): {f1_score(y_test, y_pred_base):.3f}")

# ========== CLASS WEIGHT ==========
clf_weighted = LogisticRegression(class_weight='balanced', max_iter=1000)
clf_weighted.fit(X_train, y_train)
y_pred_weighted = clf_weighted.predict(X_test)
print(f"Class Weight F1: {f1_score(y_test, y_pred_weighted):.3f}")

# ========== SMOTE ==========
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {np.bincount(y_train_sm)}")

clf_smote = LogisticRegression(max_iter=1000)
clf_smote.fit(X_train_sm, y_train_sm)
y_pred_smote = clf_smote.predict(X_test)
print(f"SMOTE F1: {f1_score(y_test, y_pred_smote):.3f}")

# ========== COMBINED PIPELINE ==========
pipeline = ImbPipeline([
    ('under', RandomUnderSampler(sampling_strategy=0.5)),
    ('over', SMOTE(sampling_strategy=1.0)),
    ('clf', LogisticRegression(max_iter=1000))
])
pipeline.fit(X_train, y_train)
y_pred_pipe = pipeline.predict(X_test)
print(f"Combined Pipeline F1: {f1_score(y_test, y_pred_pipe):.3f}")

# ========== THRESHOLD OPTIMIZATION ==========
y_proba = clf_weighted.predict_proba(X_test)[:, 1]
for thresh in [0.3, 0.4, 0.5, 0.6]:
    y_pred_thresh = (y_proba >= thresh).astype(int)
    print(f"Threshold {thresh}: F1 = {f1_score(y_test, y_pred_thresh):.3f}")
```

---

## R Implementation

```r
library(caret)
library(ROSE)
library(themis)

# ========== CREATE IMBALANCED DATA ==========
set.seed(42)
n <- 10000
X <- data.frame(
  x1 = rnorm(n),
  x2 = rnorm(n),
  class = factor(c(rep(0, 9500), rep(1, 500)))
)

# Train/test split
idx <- createDataPartition(X$class, p = 0.8, list = FALSE)
train <- X[idx, ]
test <- X[-idx, ]

cat("Class distribution:\n")
table(train$class)

# ========== BASELINE ==========
model_base <- train(class ~ ., data = train, method = "glm")
pred_base <- predict(model_base, test)
confusionMatrix(pred_base, test$class, positive = "1")

# ========== SMOTE WITH CARET ==========
library(themis)
recipe_smote <- recipe(class ~ ., data = train) %>%
  step_smote(class)

# ========== USING ROSE (RANDOM OVERSAMPLING) ==========
train_rose <- ROSE(class ~ ., data = train)$data
table(train_rose$class)
```

---

## Evaluation for Imbalanced Data

| Metric | Use When |
|--------|----------|
| **F1 Score** | General balance between precision/recall |
| **Precision-Recall AUC** | Better than ROC-AUC for imbalance |
| **Matthews Correlation Coefficient (MCC)** | All four confusion matrix cells |
| **Balanced Accuracy** | Average of class-wise recall |

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Applying SMOTE Before Train/Test Split**
> - *Problem:* Synthetic samples leak into test set
> - *Solution:* Always split first, then resample only training data
>
> **2. Over-Relying on Accuracy**
> - *Problem:* 99% accuracy means nothing for 1% minority
> - *Solution:* Use F1, Precision-Recall AUC, or MCC
>
> **3. SMOTE with High-Dimensional Data**
> - *Problem:* Creates unrealistic samples in sparse space
> - *Solution:* Use SMOTE-NC for mixed types, or reduce dimensions first

---

## Related Concepts

- [[stats/04_Supervised_Learning/F1 Score\|F1 Score]] — Appropriate metric
- [[stats/04_Supervised_Learning/ROC-AUC\|ROC-AUC]] — Threshold-independent metric
- [[stats/04_Supervised_Learning/Cross-Validation\|Cross-Validation]] — Use stratified splits
- [[stats/04_Supervised_Learning/Threshold Optimization\|Threshold Optimization]] — Adjust decision boundary

---

## References

- **Article:** Chawla, N. V., et al. (2002). SMOTE: Synthetic minority over-sampling technique. *JAIR*, 16, 321-357. [PDF](https://jair.org/index.php/jair/article/view/10302)
- **Article:** He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263-1284.
- **Package:** [imbalanced-learn](https://imbalanced-learn.org/)
