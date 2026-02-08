---
{"dg-publish":true,"permalink":"/30-knowledge/stats/04-supervised-learning/recall/","tags":["machine-learning","supervised"]}
---


## Definition

> [!abstract] Core Statement
> **Recall** (Sensitivity, TPR) measures the proportion of ==actual positives that are correctly identified==. It answers: "Of all the true positive cases, how many did I catch?"

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

---

> [!tip] Intuition (ELI5): The Disease Screening Test
> Recall asks: "Of all the people who have the disease, how many did my test detect?" High recall means we catch most cases, even if some healthy people get false positives.

---

## When Recall Matters

> [!success] Prioritize Recall When...
> - **False negatives are costly**
> - Examples: Cancer screening (don't miss cancer), fraud detection (don't miss fraud)

| High Recall | Low Recall |
|-------------|------------|
| Few missed positives | Many missed positives |
| May have more false alarms | Conservative predictions |
| Catches almost everything | Misses many positives |

---

## Python Implementation

```python
from sklearn.metrics import recall_score, classification_report

y_true = [1, 1, 0, 1, 0, 0, 1, 0]
y_pred = [1, 0, 0, 1, 0, 1, 1, 0]

# ========== BINARY ==========
recall = recall_score(y_true, y_pred)
print(f"Recall: {recall:.4f}")

# ========== AT DIFFERENT THRESHOLDS ==========
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)

# Find threshold for desired recall
target_recall = 0.95
threshold_for_target = thresholds[np.argmax(recall >= target_recall) - 1]
```

---

## Precision-Recall Trade-off

As you lower the threshold:
- More predictions become positive
- Recall increases (catch more positives)
- Precision decreases (more false positives)

---

## Related Concepts

- [[30_Knowledge/Stats/04_Supervised_Learning/Precision\|Precision]] — Trade-off partner
- [[30_Knowledge/Stats/04_Supervised_Learning/F1 Score\|F1 Score]] — Balances both
- [[30_Knowledge/Stats/07_Causal_Inference/Sensitivity and Specificity\|Sensitivity and Specificity]] — Clinical terminology
- [[30_Knowledge/Stats/04_Supervised_Learning/ROC-AUC\|ROC-AUC]] — Recall on Y-axis

---

## When to Use

> [!success] Use Recall When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Dataset is too small for training
> - Interpretability is more important than accuracy

---

## R Implementation

```r
# Recall in R
set.seed(42)

# Example implementation
data <- rnorm(100)
summary(data)
```

---

## References

- **Article:** Davis, J., & Goadrich, M. (2006). The relationship between Precision-Recall and ROC curves.
