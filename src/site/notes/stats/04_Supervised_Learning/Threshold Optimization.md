---
{"dg-publish":true,"permalink":"/stats/04-supervised-learning/threshold-optimization/","tags":["probability","machine-learning","classification","model-evaluation"]}
---


## Definition

> [!abstract] Core Statement
> **Threshold Optimization** involves choosing the ==optimal probability cutoff== for converting predicted probabilities into class labels. The default 0.5 threshold is often suboptimal.

---

> [!tip] Intuition (ELI5): The Alarm System
> Security alarms can be sensitive (catches burglars but also cats) or conservative (misses some real threats). Threshold optimization finds the right sensitivity for your specific needs.

---

## Why Not 0.5?

| Scenario | Optimal Threshold |
|----------|-------------------|
| **Balanced classes, equal costs** | 0.5 |
| **Imbalanced classes** | Usually < 0.5 |
| **High FN cost** (disease) | Lower threshold |
| **High FP cost** (spam to inbox) | Higher threshold |

---

## Optimization Strategies

### 1. Maximize F1 Score

```python
from sklearn.metrics import f1_score
import numpy as np

def find_best_threshold_f1(y_true, y_pred_proba):
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_f1 = 0
    best_threshold = 0.5
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    return best_threshold, best_f1

threshold, f1 = find_best_threshold_f1(y_test, y_pred_proba)
print(f"Best threshold: {threshold:.2f}, F1: {f1:.4f}")
```

### 2. Youden's J Statistic (ROC-based)

$$
J = \text{Sensitivity} + \text{Specificity} - 1 = \text{TPR} - \text{FPR}
$$

```python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_threshold = thresholds[best_idx]
print(f"Youden's J optimal threshold: {best_threshold:.3f}")
```

### 3. Cost-Sensitive Threshold

```python
def find_cost_optimal_threshold(y_true, y_pred_proba, 
                                 cost_fp=1, cost_fn=10):  # FN is 10x worse
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_cost = float('inf')
    best_threshold = 0.5
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        total_cost = cost_fp * fp + cost_fn * fn
        
        if total_cost < best_cost:
            best_cost = total_cost
            best_threshold = thresh
    
    return best_threshold, best_cost
```

### 4. Precision-Recall Curve Based

```python
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

# Find threshold for minimum recall
min_recall = 0.95
idx = np.argmax(recall >= min_recall)
threshold_for_recall = thresholds[idx] if idx < len(thresholds) else 0.0
```

---

## Complete Example

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# ========== VISUALIZE THRESHOLDS ==========
thresholds = np.arange(0.1, 0.9, 0.01)
f1_scores = []
precisions = []
recalls = []

for thresh in thresholds:
    y_pred = (y_pred_proba >= thresh).astype(int)
    f1_scores.append(f1_score(y_test, y_pred))
    precisions.append(precision_score(y_test, y_pred))
    recalls.append(recall_score(y_test, y_pred))

plt.figure(figsize=(10, 6))
plt.plot(thresholds, f1_scores, label='F1')
plt.plot(thresholds, precisions, label='Precision')
plt.plot(thresholds, recalls, label='Recall')
plt.axvline(0.5, color='gray', linestyle='--', label='Default 0.5')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Metrics vs Classification Threshold')
plt.legend()
plt.show()

# ========== APPLY OPTIMAL THRESHOLD ==========
optimal_threshold = 0.35  # From analysis
y_pred_optimized = (y_pred_proba >= optimal_threshold).astype(int)
print(classification_report(y_test, y_pred_optimized))
```

---

## R Implementation

```r
library(pROC)

# ROC curve
roc_obj <- roc(y_test, predicted_proba)

# Find optimal threshold (Youden's J)
coords <- coords(roc_obj, "best", ret = c("threshold", "sensitivity", "specificity"))
print(coords)

# Apply threshold
y_pred <- ifelse(predicted_proba >= coords$threshold, 1, 0)
```

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Optimizing on Test Set**
> - *Problem:* Overfitting to test set
> - *Solution:* Use cross-validation or separate validation set
>
> **2. Ignoring Business Costs**
> - *Problem:* F1 maximization may not match business goals
> - *Solution:* Define explicit cost matrix
>
> **3. Class Imbalance**
> - *Problem:* 0.5 always predicts majority class
> - *Solution:* Use precision-recall curve, not ROC

---

## Related Concepts

- [[stats/04_Supervised_Learning/ROC-AUC\|ROC-AUC]] — Threshold-independent evaluation
- [[stats/04_Supervised_Learning/Precision\|Precision]] — Affected by threshold
- [[stats/04_Supervised_Learning/Recall\|Recall]] — Trade-off with precision
- [[stats/04_Supervised_Learning/Imbalanced Data\|Imbalanced Data]] — Often requires threshold tuning

---

## References

- **Article:** Lipton, Z. C., Elkan, C., & Naryanaswamy, B. (2014). Optimal thresholding of classifiers to maximize F1 measure. *ECML-PKDD*.
