---
{"dg-publish":true,"permalink":"/stats/04-machine-learning/roc-and-auc/","tags":["Classification","Machine-Learning","Model-Evaluation","Metrics"]}
---

## Definition

> [!abstract] Core Statement
> **ROC (Receiver Operating Characteristic) Curve** plots the **True Positive Rate (Recall)** against the **False Positive Rate** at all possible classification thresholds. **AUC (Area Under the Curve)** summarizes the ROC into a single number representing the model's ability to ==discriminate between classes==.

---

## Purpose

1.  Evaluate classifier performance **across all thresholds**.
2.  Compare multiple models using a single metric (AUC).
3.  Diagnose trade-offs between sensitivity and specificity.

---

## When to Use

> [!success] Use ROC/AUC When...
> - Comparing binary classifiers.
> - You need a **threshold-independent** metric.
> - Classes are **reasonably balanced**.

> [!failure] Alternatives
> - **Imbalanced Classes:** Use **Precision-Recall AUC** instead. ROC can be overly optimistic.

---

## Theoretical Background

### Axes of the ROC Curve

- **Y-axis: True Positive Rate (TPR, Recall):** $\frac{TP}{TP+FN}$
- **X-axis: False Positive Rate (FPR):** $\frac{FP}{FP+TN}$

### Interpreting the Curve

| AUC Value | Interpretation |
|-----------|----------------|
| **1.0** | Perfect classifier. |
| **0.9 - 1.0** | Excellent discrimination. |
| **0.8 - 0.9** | Good discrimination. |
| **0.7 - 0.8** | Fair discrimination. |
| **0.5** | Random guessing (diagonal line). |
| **< 0.5** | Worse than random (model is making inverse predictions). |

### Geometric Interpretation

AUC is the probability that a randomly chosen positive instance is ranked higher than a randomly chosen negative instance.

---

## Limitations

> [!warning] Pitfalls
> 1.  **Imbalanced Data:** With rare positives, a model can have high AUC but low precision. Use **Precision-Recall AUC**.
> 2.  **Ignores Calibration:** AUC measures ranking, not probability correctness. Use **Brier Score** for calibration.
> 3.  **Does not pick threshold:** AUC tells you *how good* the model is overall, but you still need to choose a threshold for deployment.

---

## Python Implementation

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Get predicted probabilities (not class labels)
probs = model.predict_proba(X_test)[:, 1]

# Calculate AUC
auc = roc_auc_score(y_test, probs)
print(f"AUC: {auc:.3f}")

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

---

## R Implementation

```r
library(pROC)

# predicted_probs: probabilities from model
roc_obj <- roc(actual_labels, predicted_probs)

# Get AUC
auc(roc_obj)

# Plot
plot(roc_obj, main = "ROC Curve", print.auc = TRUE)
```

---

## Worked Numerical Example

> [!example] Spam Filter Comparison
> **Scenario:** Comparing two models for detecting spam emails.
> - **Model A:** AUC = 0.92
> - **Model B:** AUC = 0.85
> 
> **Interpretation:** 
> - If you pick a random Spam email and a random Non-Spam email:
> - Model A has a 92% chance of assigning a higher "Spam Score" to the actual spam email.
> - Model A separates the classes better than Model B.
> 
> **BUT:** Model A might be worse at *low False Positive Rates* (e.g., it blocks too many real emails). You must check the curve shape, not just the AUC number.

---

## Interpretation Guide

| Scenario | Interpretation | Edge Case Notes |
|----------|----------------|-----------------|
| AUC = 0.95 | Excellent discrimination. | Check for "Target Leakage" (AUC=1.0 is suspicious). |
| AUC = 0.50 | Random guessing. | Model provides no value. |
| AUC < 0.50 | Worse than random. | **Inverted labels?** Maybe 0=True and 1=False in code? |
| Curves Cross | Models trade off performance. | Model A better for high precision, B better for high recall. Pick based on business goal. |

---

## Related Concepts

- [[stats/04_Machine_Learning/Confusion Matrix\|Confusion Matrix]] - Metrics at a single threshold.
- [[stats/04_Machine_Learning/Precision-Recall Curve\|Precision-Recall Curve]] - Better for imbalanced data.
- [[stats/03_Regression_Analysis/Binary Logistic Regression\|Binary Logistic Regression]]

---

## References

- **Historical:** Hanley, J. A., & McNeil, B. J. (1982). The meaning and use of the area under a receiver operating characteristic (ROC) curve. *Radiology*. [RSNA](https://doi.org/10.1148/radiology.143.1.7063747)
- **Article:** Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters*. [ScienceDirect](https://doi.org/10.1016/j.patrec.2005.10.010)
- **Book:** James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.). Springer. [Book Website](https://www.statlearning.com/)