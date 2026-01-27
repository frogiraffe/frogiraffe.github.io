---
{"dg-publish":true,"permalink":"/stats/04-supervised-learning/f1-score/","tags":["ML-Evaluation","Classification","Metrics"]}
---


## Definition

> [!abstract] Core Statement
> The **F1 Score** is the ==harmonic mean of Precision and Recall==, providing a single metric that balances both. It ranges from 0 (worst) to 1 (best).

$$
F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}
$$

---

> [!tip] Intuition (ELI5): The Balanced Report Card
> Imagine a spam filter. **Precision** asks: "Of the emails marked spam, how many were actually spam?" **Recall** asks: "Of all actual spam, how much did we catch?" F1 is like the overall grade — if you do badly on either, your score suffers.

---

## When to Use

> [!success] Use F1 Score When...
> - Classes are **imbalanced** (accuracy is misleading)
> - You care about **both precision and recall**
> - **False positives and false negatives** are equally costly

> [!failure] F1 May Be Inappropriate When...
> - One error type is much worse (use Fβ or focus on Precision/Recall)
> - Classes are balanced (Accuracy works fine)
> - You need **probability calibration** (use Brier Score)

---

## The Fβ Family

$$
F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}
$$

| Score | β | Emphasizes |
|-------|---|------------|
| **F0.5** | 0.5 | Precision (spam filter: avoid false alarms) |
| **F1** | 1 | Equal balance |
| **F2** | 2 | Recall (disease screening: don't miss cases) |

---

## Python Implementation

```python
from sklearn.metrics import f1_score, precision_score, recall_score, fbeta_score
from sklearn.metrics import classification_report
import numpy as np

# Example predictions
y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0])
y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 0, 0])

# ========== INDIVIDUAL METRICS ==========
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")

# ========== F-BETA ==========
f05 = fbeta_score(y_true, y_pred, beta=0.5)  # Precision-weighted
f2 = fbeta_score(y_true, y_pred, beta=2)    # Recall-weighted
print(f"F0.5 (Precision focus): {f05:.3f}")
print(f"F2 (Recall focus): {f2:.3f}")

# ========== FULL REPORT ==========
print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# ========== MULTI-CLASS ==========
y_true_mc = [0, 1, 2, 0, 1, 2, 0, 1, 2]
y_pred_mc = [0, 2, 1, 0, 0, 1, 0, 1, 2]

# macro = unweighted mean, weighted = by class support
f1_macro = f1_score(y_true_mc, y_pred_mc, average='macro')
f1_weighted = f1_score(y_true_mc, y_pred_mc, average='weighted')
print(f"Macro F1: {f1_macro:.3f}")
print(f"Weighted F1: {f1_weighted:.3f}")
```

---

## R Implementation

```r
library(caret)

y_true <- factor(c(1, 1, 1, 0, 0, 0, 1, 0, 1, 0))
y_pred <- factor(c(1, 0, 1, 0, 0, 1, 1, 0, 0, 0))

# ========== CONFUSION MATRIX ==========
cm <- confusionMatrix(y_pred, y_true, positive = "1")
print(cm)

# ========== EXTRACT METRICS ==========
precision <- cm$byClass["Precision"]
recall <- cm$byClass["Sensitivity"]  # Same as Recall
f1 <- cm$byClass["F1"]

cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1, "\n")
```

---

## Interpretation Guide

| F1 Score | Interpretation |
|----------|----------------|
| **0.9+** | Excellent |
| **0.7-0.9** | Good |
| **0.5-0.7** | Moderate |
| **< 0.5** | Poor (often worse than random) |

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Ignoring Class Imbalance Averaging**
> - *Problem:* Default F1 is for positive class only
> - *Solution:* Use `macro` (equal class weight) or `weighted` (by support)
>
> **2. Threshold Dependency**
> - *Problem:* F1 depends on classification threshold (default 0.5)
> - *Solution:* Consider F1-optimal threshold or use AUC-ROC
>
> **3. Not Considering Domain Costs**
> - *Problem:* Assuming FP and FN are equally bad
> - *Solution:* Use Fβ with appropriate β, or custom cost function

---

## Related Concepts

- [[stats/04_Supervised_Learning/Precision\|Precision]] — TP / (TP + FP)
- [[stats/04_Supervised_Learning/Recall\|Recall]] — TP / (TP + FN)
- [[stats/07_Causal_Inference/Sensitivity and Specificity\|Sensitivity and Specificity]] — Alternative framing
- [[stats/04_Supervised_Learning/ROC-AUC\|ROC-AUC]] — Threshold-independent metric
- [[stats/04_Supervised_Learning/Imbalanced Data\|Imbalanced Data]] — When F1 matters most

---

## References

- **Article:** Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance measures for classification tasks. *Information Processing & Management*, 45(4), 427-437.
- **Book:** Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. [Free PDF](https://web.stanford.edu/~hastie/ElemStatLearn/)
