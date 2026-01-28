---
{"dg-publish":true,"permalink":"/stats/04-supervised-learning/precision-recall-curve/","tags":["Machine-Learning","Evaluation","Classification","Imbalanced-Data"]}
---


## Definition

> [!abstract] Core Statement
> The **Precision-Recall Curve** plots **Precision** (y-axis) against **Recall** (x-axis) at various classification thresholds. It is the **gold standard** for evaluating classifiers on **imbalanced datasets** where the positive class is rare.

![Precision-Recall Curve Visualization](https://upload.wikimedia.org/wikipedia/commons/2/26/Precisionrecall.svg)

$$
\text{Precision} = \frac{TP}{TP + FP} \quad \text{Recall} = \frac{TP}{TP + FN}
$$

**Intuition (ELI5):** You're a doctor screening for a rare disease.
- **Precision:** "Of all the patients I *flagged* as sick, how many are *actually* sick?" (Avoiding false alarms)
- **Recall:** "Of all the *actually* sick patients, how many did I *find*?" (Catching everyone)

You can't maximize both. High recall = catch everyone, but flag many healthy people (low precision). High precision = only flag when certain, but miss some sick people (low recall).

---

## When to Use

> [!success] Use PR Curve When...
> - **Classes are heavily imbalanced** (e.g., 99% negative, 1% positive).
> - You care more about the **positive class** (fraud, disease, churn).
> - True Negatives are **not meaningful** — you don't care about predicting "not fraud" correctly.
> - Comparing models where [[stats/04_Supervised_Learning/ROC-AUC\|ROC-AUC]] gives **misleadingly optimistic** results.

> [!failure] Use ROC-AUC Instead When...
> - Classes are **balanced** (50/50 or close).
> - You care equally about **both classes**.
> - True Negatives are important (e.g., correctly classifying "healthy" matters).

---

## Theoretical Background

### Why ROC Fails on Imbalanced Data

ROC uses **False Positive Rate (FPR):**
$$
\text{FPR} = \frac{FP}{FP + TN}
$$

With 99% negatives, TN is huge. Even adding 100 false positives barely changes FPR. ROC looks great while model is useless.

PR Curve uses **Precision** instead, which directly penalizes false positives without TN dilution.

### The Trade-off

| Low Threshold (0.1) | High Threshold (0.8) |
|---------------------|----------------------|
| Classify many as positive | Classify few as positive |
| High Recall (catch everyone) | Low Recall (miss positives) |
| Low Precision (many false alarms) | High Precision (confident predictions) |

### Area Under PR Curve (AUC-PR)

$$
\text{AUC-PR} = \int_0^1 P(r) \, dr
$$

| AUC-PR | Interpretation |
|--------|----------------|
| 1.0 | Perfect classifier |
| = Baseline | Random classifier. Baseline = proportion of positive class. |
| < Baseline | Worse than random |

> [!important] PR Curve Baseline is NOT 0.5!
> For a dataset with 1% positive class, a random classifier has AUC-PR ≈ 0.01, NOT 0.5.

---

## Assumptions & Diagnostics

- [ ] **Binary Classification:** PR curves are for two-class problems (can extend to multi-class with one-vs-rest).
- [ ] **Probability Outputs:** Model must output probabilities (not just class labels).
- [ ] **Threshold Independence:** PR curve shows performance across *all* thresholds.

### Key Diagnostics

| Metric | Purpose | Good Value |
|--------|---------|------------|
| **AUC-PR** | Overall ranking quality | Higher than class proportion |
| **Average Precision (AP)** | Weighted mean precision | Equivalent to AUC-PR |
| **F1 Score** | Single threshold balance | > 0.7 for most applications |
| **Precision @ Recall=0.9** | "If I must catch 90% of positives, what's my precision?" | Depends on use case |

---

## Implementation

### Python

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (precision_recall_curve, average_precision_score,
                             PrecisionRecallDisplay, f1_score)

# Create imbalanced dataset
X, y = make_classification(n_samples=10000, n_features=20, n_informative=10,
                           n_classes=2, weights=[0.95, 0.05],  # 95% negative
                           random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                     stratify=y, random_state=42)

# Fit model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ========== GET PROBABILITIES (NOT LABELS) ==========
y_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class

# ========== PRECISION-RECALL CURVE ==========
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

# ========== AVERAGE PRECISION (AUC-PR) ==========
ap = average_precision_score(y_test, y_proba)
print(f"Average Precision (AUC-PR): {ap:.3f}")
print(f"Baseline (positive class proportion): {y_test.mean():.3f}")

# ========== PLOT PR CURVE ==========
plt.figure(figsize=(10, 6))

# Method 1: Manual plot
plt.subplot(1, 2, 1)
plt.plot(recall, precision, 'b-', label=f'PR Curve (AP = {ap:.2f})')
plt.axhline(y=y_test.mean(), color='r', linestyle='--', label='Random Baseline')
plt.xlabel('Recall (Sensitivity)')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True, alpha=0.3)

# Method 2: sklearn display
plt.subplot(1, 2, 2)
PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=plt.gca())
plt.title('PR Curve (sklearn)')

plt.tight_layout()
plt.show()

# ========== FIND OPTIMAL THRESHOLD ==========
# Optimize for F1 score
f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"\nOptimal Threshold: {optimal_threshold:.3f}")
print(f"  Precision: {precision[optimal_idx]:.3f}")
print(f"  Recall: {recall[optimal_idx]:.3f}")
print(f"  F1 Score: {f1_scores[optimal_idx]:.3f}")

# ========== APPLY CUSTOM THRESHOLD ==========
y_pred_custom = (y_proba >= optimal_threshold).astype(int)
print(f"\nF1 with custom threshold: {f1_score(y_test, y_pred_custom):.3f}")
print(f"F1 with default (0.5): {f1_score(y_test, model.predict(X_test)):.3f}")
```

### R

```r
library(PRROC)
library(ggplot2)
library(caret)

# Create imbalanced dataset
set.seed(42)
n <- 10000
X <- matrix(rnorm(n * 20), ncol = 20)
prob <- plogis(1.5 * X[,1] - 2 * X[,2] + 0.5 * X[,3] - 3)  # Low base rate
y <- rbinom(n, 1, prob)
table(y)  # Check imbalance

# Split data
train_idx <- sample(1:n, 0.7 * n)
X_train <- X[train_idx, ]
X_test <- X[-train_idx, ]
y_train <- y[train_idx]
y_test <- y[-train_idx]

# Fit model
model <- glm(y_train ~ X_train, family = binomial)
y_proba <- predict(model, newdata = data.frame(X_train = X_test), type = "response")

# ========== PR CURVE WITH PRROC ==========
pr <- pr.curve(scores.class0 = y_proba[y_test == 1],  # Positive class scores
               scores.class1 = y_proba[y_test == 0],  # Negative class scores
               curve = TRUE)

print(pr)
plot(pr, main = "Precision-Recall Curve")

# ========== MANUAL CALCULATION ==========
# Sort by probability
order_idx <- order(y_proba, decreasing = TRUE)
y_sorted <- y_test[order_idx]

# Calculate precision and recall at each threshold
n_pos <- sum(y_test)
tp <- cumsum(y_sorted)
fp <- cumsum(1 - y_sorted)
precision <- tp / (tp + fp)
recall <- tp / n_pos

# Plot
df <- data.frame(precision = precision, recall = recall)
ggplot(df, aes(x = recall, y = precision)) +
  geom_line(color = "blue", size = 1) +
  geom_hline(yintercept = mean(y_test), linetype = "dashed", color = "red") +
  labs(title = "Precision-Recall Curve",
       x = "Recall", y = "Precision") +
  annotate("text", x = 0.5, y = mean(y_test) + 0.05, 
           label = "Random Baseline", color = "red") +
  theme_minimal()
```

---

## Interpretation Guide

| Output | Example Value | Interpretation | Edge Case/Warning |
|--------|---------------|----------------|-------------------|
| **AUC-PR** | 0.45 | Model ranks positives well above random (baseline=0.05). | Compare to baseline! AUC-PR=0.45 vs baseline=0.05 is excellent. |
| **AUC-PR** | 0.06 | Barely above baseline (0.05). Model is nearly random. | Even small AUC-PR can be useful if baseline is very low. |
| **AUC-PR** | 0.98 | Near-perfect ranking of positive cases. | Suspiciously high — check for data leakage or test set contamination. |
| **Precision @ Recall=0.9** | 0.30 | To catch 90% of positives, 70% of flags are false alarms. | Acceptable if cost of missing positives >> cost of false alarms. |
| **Optimal threshold** | 0.15 | Threshold that maximizes F1 (balance of Precision/Recall). | Lower than 0.5 is expected for imbalanced data! |
| **PR curve shape** | Drops sharply | Precision degrades quickly as you try to catch more positives. | Model struggles to identify positives without many false alarms. |

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Comparing AUC-PR Across Datasets**
> - *Problem:* Dataset A (1% positive) has AUC-PR=0.40, Dataset B (10% positive) has AUC-PR=0.60.
> - *Conclusion:* Model B is better? **No!**
> - *Reality:* Baseline differs (0.01 vs 0.10). Model A is relatively much better.
> - *Solution:* Compare to baseline or use "lift" over baseline.
>
> **2. Using Default Threshold (0.5) for Imbalanced Data**
> - *Problem:* With 1% positive class, threshold 0.5 predicts almost everything as negative.
> - *Result:* High accuracy (99%) but zero recall for positives.
> - *Solution:* Use PR curve to find optimal threshold based on business needs.
>
> **3. Ignoring Class Weights in Training**
> - *Problem:* Model trained without class weights learns to always predict majority class.
> - *Result:* PR curve hugs the baseline.
> - *Solution:* Use `class_weight='balanced'` in sklearn or SMOTE oversampling.
>
> **4. Reporting Only AUC-PR Without Curve**
> - *Problem:* Two models with same AUC-PR can have very different curves.
> - *Example:* Model A is flat at 50% precision. Model B has 80% precision for top 10% recall, then drops.
> - *Solution:* Always visualize the curve. Report Precision @ specific Recall levels.

---

## Worked Numerical Example

> [!example] Fraud Detection System Evaluation
> **Scenario:** 10,000 transactions, 100 are fraud (1% positive rate).
>
> **Step 1: Model Predictions (Sorted by Probability)**
> ```
> Rank | Probability | Actual | TP (cumulative) | FP (cumulative)
> 1    | 0.95        | Fraud  | 1               | 0
> 2    | 0.92        | Fraud  | 2               | 0
> 3    | 0.88        | Fraud  | 3               | 0
> 4    | 0.85        | Legit  | 3               | 1
> 5    | 0.82        | Fraud  | 4               | 1
> ...
> 50   | 0.60        | Fraud  | 40              | 10
> ...
> 100  | 0.45        | Fraud  | 70              | 30
> ...
> ```
>
> **Step 2: Calculate Precision and Recall at Key Points**
> ```
> After Top 5 predictions:
>   TP=4, FP=1, FN=96
>   Precision = 4/(4+1) = 0.80
>   Recall = 4/100 = 0.04
>   
> After Top 50 predictions:
>   TP=40, FP=10, FN=60
>   Precision = 40/(40+10) = 0.80
>   Recall = 40/100 = 0.40
>   
> After Top 100 predictions:
>   TP=70, FP=30, FN=30
>   Precision = 70/(70+30) = 0.70
>   Recall = 70/100 = 0.70
>   
> After Top 200 predictions:
>   TP=90, FP=110, FN=10
>   Precision = 90/(90+110) = 0.45
>   Recall = 90/100 = 0.90
> ```
>
> **Step 3: Plot PR Curve**
> ```
> Precision
>   1.0|*
>   0.8| ****
>   0.6|     ****
>   0.4|         ****
>   0.2|             ****
>   0.0|_________________
>      0    0.4   0.8  1.0  Recall
> ```
>
> **Step 4: Business Decision**
> - If cost of missing fraud (\$10,000 per fraud) >> cost of investigation (\$100 per flag):
>   - Choose high recall (0.90): Catch 90 of 100 frauds, investigate 200 total.
>   - Cost: 10 missed × \$10,000 + 200 × \$100 = \$120,000
>   
> - If investigation is expensive:
>   - Choose high precision (0.80): Catch 40 frauds, investigate only 50.
>   - Cost: 60 missed × \$10,000 + 50 × \$100 = \$605,000
>
> **Conclusion:** High recall is better for this fraud scenario.

---

## PR Curve vs ROC Curve

| Aspect | PR Curve | ROC Curve |
|--------|----------|-----------|
| **Axes** | Precision vs Recall | TPR vs FPR |
| **Uses True Negatives?** | ❌ No | ✅ Yes |
| **Baseline** | Positive class proportion | 0.5 diagonal |
| **Imbalanced data** | ✅ Preferred | ⚠️ Misleadingly optimistic |
| **Balanced data** | Works fine | ✅ Preferred |
| **Interpretation** | Focus on positive class | Equal focus on both classes |

---

## Related Concepts

**Prerequisites:**
- [[stats/04_Supervised_Learning/Confusion Matrix\|Confusion Matrix]] — TP, FP, FN definitions
- [[stats/03_Regression_Analysis/Binary Logistic Regression\|Binary Logistic Regression]] — Common classifier to evaluate

**Companions:**
- [[stats/04_Supervised_Learning/ROC-AUC\|ROC-AUC]] — Alternative for balanced data
- [[stats/04_Supervised_Learning/F1 Score\|F1 Score]] — Harmonic mean of Precision and Recall

**Applications:**
- [[stats/04_Supervised_Learning/Imbalanced Data\|Imbalanced Data]] — When PR curve is essential
- [[stats/04_Supervised_Learning/Threshold Optimization\|Threshold Optimization]] — Choosing operating point on curve

---

## References

- **Historical:** Davis, J., & Goadrich, M. (2006). The relationship between Precision-Recall and ROC curves. *ICML*. [ACM Digital Library](https://dl.acm.org/doi/10.1145/1143844.1143874)
- **Article:** Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE*. [DOI: 10.1371/journal.pone.0118432](https://doi.org/10.1371/journal.pone.0118432)
- **Book:** James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.). Springer. [Book Website](https://www.statlearning.com/)
