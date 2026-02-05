---
{"dg-publish":true,"permalink":"/stats/04-supervised-learning/precision/","tags":["probability","ml-evaluation","classification","metrics"]}
---


## Definition

> [!abstract] Core Statement
> **Precision** measures the proportion of ==positive predictions that are actually correct==. It answers: "Of all the items I predicted as positive, how many were truly positive?"

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

---

> [!tip] Intuition (ELI5): The Spam Filter Test
> Precision asks: "Of the emails my filter marked as spam, how many were really spam?" High precision means few false alarms.

---

## When Precision Matters

> [!success] Prioritize Precision When...
> - **False positives are costly**
> - Examples: Spam filter (don't lose important emails), fraud detection (don't freeze good accounts)

| High Precision | Low Precision |
|----------------|---------------|
| Few false positives | Many false positives |
| Conservative predictions | Liberal predictions |
| May miss some positives | Catches more positives |

---

## Python Implementation

```python
from sklearn.metrics import precision_score, classification_report

y_true = [1, 1, 0, 1, 0, 0, 1, 0]
y_pred = [1, 0, 0, 1, 0, 1, 1, 0]

# ========== BINARY ==========
precision = precision_score(y_true, y_pred)
print(f"Precision: {precision:.4f}")

# ========== MULTI-CLASS ==========
# 'macro' = unweighted mean, 'weighted' = by support
precision_macro = precision_score(y_true, y_pred, average='macro')
precision_weighted = precision_score(y_true, y_pred, average='weighted')

# ========== FULL REPORT ==========
print(classification_report(y_true, y_pred))
```

---

## Precision-Recall Trade-off

| Threshold ↑ | Precision ↑ | Recall ↓ |
|-------------|-------------|----------|
| Threshold ↓ | Precision ↓ | Recall ↑ |

> [!important] You Can't Have Both
> Improving precision usually hurts recall, and vice versa. Choose based on business costs.

---

## Related Concepts

- [[stats/04_Supervised_Learning/Recall\|Recall]] — Trade-off partner
- [[stats/04_Supervised_Learning/F1 Score\|F1 Score]] — Harmonic mean of both
- [[stats/04_Supervised_Learning/ROC-AUC\|ROC-AUC]] — Threshold-independent metric
- [[stats/07_Causal_Inference/Sensitivity and Specificity\|Sensitivity and Specificity]] — Alternative framing

---

## References

- **Article:** Davis, J., & Goadrich, M. (2006). The relationship between Precision-Recall and ROC curves. *ICML*.
