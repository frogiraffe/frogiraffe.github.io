---
{"dg-publish":true,"permalink":"/stats/04-supervised-learning/confusion-matrix/","tags":["Classification","Machine-Learning","Model-Evaluation","Metrics"]}
---

## Definition

> [!abstract] Core Statement
> A **Confusion Matrix** is a table that summarizes the performance of a classification model by comparing predicted labels to actual labels. It forms the basis for calculating metrics like Accuracy, Precision, Recall, and F1-Score.

---

> [!tip] Intuition (ELI5): The "Truth Detector"
> Imagine you are a teacher grading a "Yes/No" test. A Confusion Matrix is a report card that doesn't just say "you got 8 out of 10." It tells you *exactly* which "Yes" answers you missed and which "No" answers you accidentally called "Yes." 

---

## Purpose

1.  Visualize classification performance.
2.  Calculate key metrics for model evaluation.
3.  Identify types of errors (False Positives vs False Negatives).

---

## When to Use

> [!success] Use Confusion Matrix When...
> - Evaluating any **classification model** (Binary or Multi-class).
> - You need to understand *what kind* of errors the model makes.
> - Accuracy alone is misleading (imbalanced classes).

---

## Theoretical Background

### The Matrix (Binary Classification)

|  | **Predicted Negative (0)** | **Predicted Positive (1)** |
|--|---------------------------|---------------------------|
| **Actual Negative (0)** | **True Negative (TN)** | False Positive (FP) *Type I Error* |
| **Actual Positive (1)** | False Negative (FN) *Type II Error* | **True Positive (TP)** |

### Key Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Accuracy** | $\frac{TP + TN}{Total}$ | Overall correctness. (Misleading if imbalanced). |
| **Precision** | $\frac{TP}{TP + FP}$ | Of all predicted positives, how many are correct? (Avoid false alarms). |
| **Recall (Sensitivity, TPR)** | $\frac{TP}{TP + FN}$ | Of all actual positives, how many were found? (Avoid missing cases). |
| **Specificity (TNR)** | $\frac{TN}{TN + FP}$ | Of all actual negatives, how many were correctly identified? |
| **F1-Score** | $\frac{2 \cdot Precision \cdot Recall}{Precision + Recall}$ | Harmonic mean of Precision and Recall. Balances both. |

---

## Precision vs Recall Trade-off

> [!important] Context Matters
> - **High Precision Priority:** Spam detection. (False Positives = Real emails in spam).
> - **High Recall Priority:** Disease screening. (False Negatives = Missing sick patients).

---

## Limitations

> [!warning] Pitfalls
> 1.  **Accuracy Paradox:** In imbalanced data (e.g., 99% negative), a model predicting all negative gets 99% accuracy but is useless.
> 2.  **Threshold Dependent:** Metrics change with the classification threshold. Use [[stats/04_Supervised_Learning/ROC-AUC\|ROC-AUC]] for threshold-independent evaluation.

---

## Python Implementation

```python
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# y_true: actual labels, y_pred: predicted labels
cm = confusion_matrix(y_true, y_pred)

# Heatmap Visualization
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred 0', 'Pred 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.title("Confusion Matrix")
plt.show()

# Full Report
print(classification_report(y_true, y_pred))
```

---

## R Implementation

```r
library(caret)

# Create factors
pred <- factor(predicted_labels)
actual <- factor(actual_labels)

# Confusion Matrix
confusionMatrix(data = pred, reference = actual)

# Output: Accuracy, Sensitivity, Specificity, Precision, etc.
```

---

## Worked Numerical Example

> [!example] Rare Disease Detection (Imbalanced)
> **Data:** 100 Patients (95 Healthy, 5 Sick).
> **Model Prediction:** Predicts everyone is "Healthy" (All Negative).
> 
> **Confusion Matrix:**
> - TP = 0, FN = 5 (Missed all sick people!)
> - TN = 95, FP = 0
> 
> **Metrics:**
> - **Accuracy:** $(0+95)/100 = 95\%$ (Looks amazing!)
> - **Recall:** $0 / (0+5) = 0\%$ (Missed everyone!)
> - **Precision:** Undefined (0/0) or 0.
> 
> **Conclusion:** The model is useless despite 95% accuracy.

---

## Interpretation Guide

| Output | Interpretation | Edge Case Notes |
|--------|----------------|-----------------|
| High Accuracy, Low Recall | **Accuracy Paradox**. | Common in imbalanced data (Fraud, Disease). Ignore accuracy. |
| High Recall, Low Precision | "Nets too wide". Many false alarms. | Okay for screening tests (cheap filters). |
| High Precision, Low Recall | "Very picky". Misses many cases, but trusts positive predictions. | Good for spam filters (don't delete real email). |
| F1 = 0.9 | Strong balance of P and R. | Excellent model. |

---

## Common Pitfall Example

> [!warning] Threshold Setting Blindness
> **Scenario:** Logistic Regression outputs probabilities (0.1, 0.6, 0.9, ...).
> **Default:** Classify as 1 if $p > 0.5$.
> 
> **Problem:** 
> - If "Positive" is Cancer, $p > 0.5$ might be too strict.
> - You miss patients with $p=0.4$ who might have cancer.
> 
> **Solution:** 
> - **Tune the Threshold.**
> - Lower threshold to 0.2 $\to$ Increase Recall (Find more cancer), but decrease Precision (More false alarms).
> - Use the **ROC Curve** to make this decision intentionally.

---

## Related Concepts

- [[stats/04_Supervised_Learning/ROC-AUC\|ROC-AUC]] - Threshold-independent performance.
- [[stats/03_Regression_Analysis/Binary Logistic Regression\|Binary Logistic Regression]] - Typical model producing these outputs.
- [[stats/04_Supervised_Learning/Precision-Recall Curve\|Precision-Recall Curve]]

---

## References

- **Book:** James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.). Springer. [Book Website](https://www.statlearning.com/)
- **Article:** Powers, D. M. W. (2011). Evaluation: From precision, recall and F-measure to ROC, informedness, markedness and correlation. *Journal of Machine Learning Technologies*, 2(1), 37-63. [Link](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.214.2210)
- **Book:** Provost, F., & Fawcett, T. (2013). *Data Science for Business*. O'Reilly Media. [O'Reilly](https://www.oreilly.com/library/view/data-science-for/9781449374273/)
