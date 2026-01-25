---
{"dg-publish":true,"permalink":"/stats/04-machine-learning/precision-recall-curve/","tags":["Machine-Learning","Evaluation","Diagnostics"]}
---


## Definition

> [!abstract] Overview
> The **Precision-Recall (PR) Curve** plots Precision (Y-axis) against Recall (X-axis) for different probability thresholds. It is the preferred metric for **Imbalanced Class** problems (e.g., Fraud Detection, Cancer Diagnosis).

- **Precision:** Of all predicted positives, how many are actually positive? ($\frac{TP}{TP + FP}$)
- **Recall (Sensitivity):** Of all actual positives, how many did we find? ($\frac{TP}{TP + FN}$)

---

## 1. When to Use?

| Metric | Condition | Example |
|--------|-----------|---------|
| **ROC / AUC** | Balanced Classes | 50% Dogs, 50% Cats. |
| **PR Curve** | Imbalanced Classes | 99% Normal, 1% Fraud. |

**Why?**
ROC includes True Negatives (TN) in the calculation. Since TNs are huge in imbalanced data, ROC always looks "optimistically good" ($0.98$ AUC). PR Curve ignores TNs and focuses on the minority class.

---

## 2. Threshold Moving

Standard predict (`model.predict()`) uses a threshold of 0.5.
For cancer detection, we might care more about Recall (finding all cases) even if Precision drops. We lower the threshold (e.g., to 0.1).

---

## 3. Python Implementation

```python
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

# Get probabilities (Positive Class)
y_scores = model.predict_proba(X_test)[:, 1]

# Calculate Precision and Recall for all thresholds
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

# Calculate Area Under Curve (AUC-PR)
pr_auc = auc(recall, precision)

# Plot
plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()
```

---

## Related Concepts

- [[stats/04_Machine_Learning/ROC & AUC\|ROC & AUC]]
- [[stats/04_Machine_Learning/Confusion Matrix\|Confusion Matrix]]
- [[F1 Score\|F1 Score]] (Harmonic mean of Precision and Recall)
