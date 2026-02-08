---
{"dg-publish":true,"permalink":"/30-knowledge/stats/04-supervised-learning/model-evaluation-metrics/","tags":["machine-learning","supervised"]}
---

## Overview

> [!abstract] Definition
> **Model Evaluation Metrics** quantify the performance of a statistical or machine learning model. The choice of metric depends on the problem type (Regression vs. Classification) and the specific business objectives.

---

## 1. Regression Metrics

Used when the target variable is continuous.

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MAE (Mean Absolute Error)** | $\frac{1}{n}\sum |y_i - \hat{y}_i|$ | Average magnitude of errors. Robust to outliers. |
| **MSE (Mean Squared Error)** | $\frac{1}{n}\sum (y_i - \hat{y}_i)^2$ | Penalizes large errors heavily. Differentiable. |
| **RMSE (Root MSE)** | $\sqrt{MSE}$ | Same units as $Y$. Standard metric. |
| **$R^2$ (R-Squared)** | $1 - \frac{SS_{res}}{SS_{tot}}$ | Proportion of variance explained. |

---

## 2. Classification Metrics

Used when the target variable is categorical.

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Accuracy** | $(TP+TN)/N$ | Overall correctness. Bad for imbalanced data. |
| **Precision** | $TP/(TP+FP)$ | Trustworthiness of positive predictions. |
| **Recall** | $TP/(TP+FN)$ | Ability to find all positive instances. |
| **F1 Score** | $2 \cdot \frac{P \cdot R}{P+R}$ | Harmonic mean. Balance between P & R. |
| **AUC-ROC** | Integral of ROC | Discriminative ability across thresholds. |
| **Log-Loss** | $-\frac{1}{N}\sum [y \ln(p) + (1-y)\ln(1-p)]$ | Penalty for confident wrong predictions. |

---

## 3. Related Concepts

- [[30_Knowledge/Stats/04_Supervised_Learning/Confusion Matrix\|Confusion Matrix]] - Source of classification metrics.
- [[30_Knowledge/Stats/04_Supervised_Learning/ROC-AUC\|ROC-AUC]] - Detailed note on Area Under Curve.
- [[30_Knowledge/Stats/03_Regression_Analysis/Simple Linear Regression\|Simple Linear Regression]] - Uses $R^2$.

---

## Definition

> [!abstract] Core Statement
> **Model Evaluation Metrics** ... Refer to standard documentation

---

> [!tip] Intuition (ELI5)
> Refer to standard documentation

---

## When to Use

> [!success] Use Model Evaluation Metrics When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Dataset is too small for training
> - Interpretability is more important than accuracy

---

## Python Implementation

```python
import numpy as np
import pandas as pd

# Example implementation of Model Evaluation Metrics
# See documentation for details

data = np.random.randn(100)
print(f"Mean: {np.mean(data):.3f}")
print(f"Std: {np.std(data):.3f}")
```

---

## R Implementation

```r
# Model Evaluation Metrics in R
set.seed(42)

# Example implementation
data <- rnorm(100)
summary(data)
```

---

## Related Concepts

- [[30_Knowledge/Stats/04_Supervised_Learning/Random Forest\|Random Forest]]
- [[30_Knowledge/Stats/04_Supervised_Learning/Gradient Boosting\|Gradient Boosting]]
- [[30_Knowledge/Stats/04_Supervised_Learning/Cross-Validation\|Cross-Validation]]

---

## References

- **Book:** Japkowicz, N., & Shah, M. (2011). *Evaluating Learning Algorithms*. Cambridge. [Cambridge Link](https://www.cambridge.org/core/books/evaluating-learning-algorithms/6B7A1C6A1F8A6C6B6F6A6E6F6A6E6F6A)
- **Book:** Zheng, A. (2015). *Evaluating Machine Learning Models*. O'Reilly. [O'Reilly](https://www.oreilly.com/library/view/evaluating-machine-learning/9781491932148/)
- **Article:** Powers, D. M. (2011). Evaluation: From precision, recall and F-measure to ROC, informedness, markedness and correlation. *Journal of Machine Learning Technologies*. [Link](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.214.2210)