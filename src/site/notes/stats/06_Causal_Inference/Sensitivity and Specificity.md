---
{"dg-publish":true,"permalink":"/stats/06-causal-inference/sensitivity-and-specificity/","tags":["Classification","Model-Evaluation","Diagnostic-Testing","Medical-Statistics"]}
---


## Definition

> [!abstract] Core Statement
> **Sensitivity** and **Specificity** are fundamental metrics for evaluating ==binary classification tests==, especially in medical diagnostics. 
> - **Sensitivity (True Positive Rate):** Proportion of actual positives correctly identified.
> - **Specificity (True Negative Rate):** Proportion of actual negatives correctly identified.

**Intuition (ELI5):** 
- **Sensitivity:** "If you're sick, will the test catch it?" (Don't miss the sick people!)
- **Specificity:** "If you're healthy, will the test say you're healthy?" (Don't scare healthy people!)

---

## Purpose

1.  **Evaluate Diagnostic Tests:** How well does a test detect disease?
2.  **Compare Classifiers:** Which model has better detection rates?
3.  **Set Thresholds:** Trade off between missing cases and false alarms.
4.  **Clinical Decision Making:** Understand test limitations.

---

## When to Use

> [!success] Use Sensitivity/Specificity When...
> - Evaluating **binary classification** (disease/healthy, fraud/legitimate).
> - **Prevalence varies** across populations (unlike accuracy).
> - Comparing **diagnostic tests** or screening tools.
> - Building **clinical decision rules**.

> [!failure] Limitations...
> - They describe **one threshold**. For overall performance, use [[stats/04_Machine_Learning/ROC & AUC\|ROC & AUC]].
> - They don't tell you **what a positive result means** (use Predictive Values).
> - Misleading with **imbalanced classes** without context.

---

## Theoretical Background

### Confusion Matrix Foundation

|  | Actual Positive (+) | Actual Negative (−) |
|--|---------------------|---------------------|
| **Predicted +** | True Positive (TP) | False Positive (FP) |
| **Predicted −** | False Negative (FN) | True Negative (TN) |

### Formulas

$$
\text{Sensitivity} = \frac{TP}{TP + FN} = P(\text{Test}+ | \text{Disease}+)
$$

$$
\text{Specificity} = \frac{TN}{TN + FP} = P(\text{Test}- | \text{Disease}-)
$$

### Related Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Sensitivity (TPR)** | $\frac{TP}{TP+FN}$ | Probability of positive test given disease |
| **Specificity (TNR)** | $\frac{TN}{TN+FP}$ | Probability of negative test given healthy |
| **False Positive Rate (FPR)** | $\frac{FP}{TN+FP} = 1 - \text{Specificity}$ | Type I error rate |
| **False Negative Rate (FNR)** | $\frac{FN}{TP+FN} = 1 - \text{Sensitivity}$ | Type II error rate |
| **Positive Predictive Value (PPV)** | $\frac{TP}{TP+FP}$ | $P(\text{Disease}+ | \text{Test}+)$ |
| **Negative Predictive Value (NPV)** | $\frac{TN}{TN+FN}$ | $P(\text{Disease}- | \text{Test}-)$ |

### The Sensitivity-Specificity Trade-off

Moving the classification threshold affects both metrics:
- **Lower threshold** → Higher sensitivity, lower specificity (more positives)
- **Higher threshold** → Higher specificity, lower sensitivity (fewer positives)

The [[stats/04_Machine_Learning/ROC & AUC\|ROC & AUC]] curve visualizes this trade-off across all thresholds.

### Relationship to Bayes' Theorem

Using [[stats/01_Foundations/Bayes' Theorem\|Bayes' Theorem]], we can convert sensitivity/specificity to predictive values:

$$
\text{PPV} = \frac{\text{Sensitivity} \times \text{Prevalence}}{\text{Sensitivity} \times \text{Prevalence} + (1-\text{Specificity}) \times (1-\text{Prevalence})}
$$

**Key Insight:** Even with high sensitivity and specificity, PPV can be low if disease is rare!

---

## Assumptions

- [ ] **Binary Outcome:** Classification is dichotomous.
- [ ] **Gold Standard:** True disease status is known (reference test is perfect).
- [ ] **Representative Sample:** Sample reflects the target population.
- [ ] **Independent Observations:** Each test result is independent.

---

## Limitations

> [!warning] Pitfalls
> 1. **Threshold Dependency:** Sensitivity/specificity are for one threshold only.
> 2. **Prevalence Ignored:** A 95% sensitive test is useless if PPV is 1% (rare disease).
> 3. **Not Actionable Alone:** Clinicians need predictive values, not just test characteristics.
> 4. **Spectrum Bias:** Performance varies across disease severity.

---

## Python Implementation

```python
import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, classification_report, 
                             roc_curve, roc_auc_score)
import matplotlib.pyplot as plt

# ========== EXAMPLE DATA ==========
# Binary classification: Disease screening test
np.random.seed(42)
n = 1000

# True disease status (10% prevalence)
y_true = np.random.binomial(1, 0.10, n)

# Test results (sensitivity=0.90, specificity=0.85)
y_pred = np.where(y_true == 1, 
                  np.random.binomial(1, 0.90, n),  # 90% sensitivity
                  np.random.binomial(1, 0.15, n))  # 15% FPR = 85% specificity

# ========== CONFUSION MATRIX ==========
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

print("=== Confusion Matrix ===")
print(f"TN: {tn}, FP: {fp}")
print(f"FN: {fn}, TP: {tp}")

# ========== SENSITIVITY & SPECIFICITY ==========
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0
prevalence = y_true.mean()

print("\n=== Diagnostic Metrics ===")
print(f"Sensitivity (TPR): {sensitivity:.3f}")
print(f"Specificity (TNR): {specificity:.3f}")
print(f"PPV (Precision): {ppv:.3f}")
print(f"NPV: {npv:.3f}")
print(f"Prevalence: {prevalence:.3f}")

# ========== EFFECT OF PREVALENCE ON PPV ==========
def calculate_ppv(sens, spec, prev):
    return (sens * prev) / (sens * prev + (1 - spec) * (1 - prev))

print("\n=== PPV at Different Prevalences ===")
for prev in [0.01, 0.05, 0.10, 0.20, 0.50]:
    ppv_calc = calculate_ppv(0.90, 0.85, prev)
    print(f"Prevalence {prev*100:5.1f}%: PPV = {ppv_calc:.3f}")

# ========== ROC CURVE ==========
# For probabilistic predictions
y_prob = np.random.beta(2 + 3*y_true, 5 - 3*y_true)  # Simulated probabilities

fpr, tpr, thresholds = roc_curve(y_true, y_prob)
auc = roc_auc_score(y_true, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, 'b-', label=f'ROC Curve (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('1 - Specificity (False Positive Rate)')
plt.ylabel('Sensitivity (True Positive Rate)')
plt.title('ROC Curve: Sensitivity vs (1-Specificity)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ========== OPTIMAL THRESHOLD (YOUDEN'S J) ==========
j_stat = tpr - fpr  # Youden's J statistic
optimal_idx = np.argmax(j_stat)
optimal_threshold = thresholds[optimal_idx]

print(f"\n=== Optimal Threshold (Youden's J) ===")
print(f"Threshold: {optimal_threshold:.3f}")
print(f"Sensitivity: {tpr[optimal_idx]:.3f}")
print(f"Specificity: {1 - fpr[optimal_idx]:.3f}")

# ========== THRESHOLD ANALYSIS ==========
print("\n=== Threshold Analysis ===")
print("Threshold | Sensitivity | Specificity | PPV")
for thresh in [0.1, 0.3, 0.5, 0.7, 0.9]:
    y_pred_t = (y_prob >= thresh).astype(int)
    cm_t = confusion_matrix(y_true, y_pred_t)
    tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
    sens_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
    spec_t = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0
    ppv_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
    print(f"   {thresh:.1f}    |    {sens_t:.3f}    |    {spec_t:.3f}    | {ppv_t:.3f}")
```

---

## R Implementation

```r
library(caret)
library(pROC)
library(ggplot2)

# ========== EXAMPLE DATA ==========
set.seed(42)
n <- 1000

# True disease status (10% prevalence)
y_true <- rbinom(n, 1, 0.10)

# Test results
y_pred <- ifelse(y_true == 1,
                 rbinom(n, 1, 0.90),   # 90% sensitivity
                 rbinom(n, 1, 0.15))   # 15% FPR

y_true <- factor(y_true, levels = c(0, 1), labels = c("Negative", "Positive"))
y_pred <- factor(y_pred, levels = c(0, 1), labels = c("Negative", "Positive"))

# ========== CONFUSION MATRIX ==========
cm <- confusionMatrix(y_pred, y_true, positive = "Positive")
print(cm)

# ========== EXTRACT METRICS ==========
sensitivity <- cm$byClass["Sensitivity"]
specificity <- cm$byClass["Specificity"]
ppv <- cm$byClass["Pos Pred Value"]
npv <- cm$byClass["Neg Pred Value"]

cat("\n=== Key Metrics ===\n")
cat("Sensitivity:", round(sensitivity, 3), "\n")
cat("Specificity:", round(specificity, 3), "\n")
cat("PPV:", round(ppv, 3), "\n")
cat("NPV:", round(npv, 3), "\n")

# ========== EFFECT OF PREVALENCE ON PPV ==========
calculate_ppv <- function(sens, spec, prev) {
  (sens * prev) / (sens * prev + (1 - spec) * (1 - prev))
}

cat("\n=== PPV at Different Prevalences ===\n")
for (prev in c(0.01, 0.05, 0.10, 0.20, 0.50)) {
  ppv_calc <- calculate_ppv(0.90, 0.85, prev)
  cat(sprintf("Prevalence %5.1f%%: PPV = %.3f\n", prev * 100, ppv_calc))
}

# ========== ROC CURVE ==========
# Simulate probability scores
y_prob <- rbeta(n, 2 + 3 * as.numeric(y_true == "Positive"), 
                   5 - 3 * as.numeric(y_true == "Positive"))

roc_obj <- roc(y_true, y_prob)
print(roc_obj)

# Plot ROC
plot(roc_obj, 
     main = "ROC Curve",
     print.auc = TRUE,
     legacy.axes = TRUE,  # 1-Specificity on x-axis
     xlab = "1 - Specificity (False Positive Rate)",
     ylab = "Sensitivity (True Positive Rate)")

# ========== OPTIMAL THRESHOLD (YOUDEN'S J) ==========
coords_best <- coords(roc_obj, "best", ret = c("threshold", "sensitivity", "specificity"))
cat("\n=== Optimal Threshold (Youden's J) ===\n")
print(coords_best)

# ========== THRESHOLD ANALYSIS ==========
cat("\n=== Threshold Analysis ===\n")
for (thresh in c(0.1, 0.3, 0.5, 0.7, 0.9)) {
  coords_t <- coords(roc_obj, thresh, ret = c("sensitivity", "specificity", "ppv"))
  cat(sprintf("Threshold %.1f: Sens=%.3f, Spec=%.3f, PPV=%.3f\n", 
              thresh, coords_t[1], coords_t[2], coords_t[3]))
}

# ========== VISUALIZATION ==========
# Sensitivity vs Specificity at different thresholds
thresholds <- seq(0, 1, 0.01)
sens_vec <- sapply(thresholds, function(t) coords(roc_obj, t, ret = "sensitivity"))
spec_vec <- sapply(thresholds, function(t) coords(roc_obj, t, ret = "specificity"))

df_plot <- data.frame(
  Threshold = thresholds,
  Sensitivity = sens_vec,
  Specificity = spec_vec
)

library(tidyr)
df_long <- pivot_longer(df_plot, cols = c(Sensitivity, Specificity),
                        names_to = "Metric", values_to = "Value")

ggplot(df_long, aes(x = Threshold, y = Value, color = Metric)) +
  geom_line(size = 1) +
  geom_vline(xintercept = coords_best$threshold, linetype = "dashed") +
  labs(title = "Sensitivity-Specificity Trade-off",
       subtitle = paste("Optimal threshold:", round(coords_best$threshold, 3))) +
  theme_minimal()
```

---

## Worked Numerical Example

> [!example] COVID-19 Rapid Test Evaluation
> **Study:** 1000 people tested; 100 have COVID (10% prevalence).
> **Results:**
> 
> |  | COVID+ | COVID− |
> |--|--------|--------|
> | Test+ | 90 | 90 |
> | Test− | 10 | 810 |
> 
> **Step 1: Calculate Sensitivity**
> $$\text{Sensitivity} = \frac{90}{90 + 10} = \frac{90}{100} = 0.90 \quad (90\%)$$
> 
> **Step 2: Calculate Specificity**
> $$\text{Specificity} = \frac{810}{810 + 90} = \frac{810}{900} = 0.90 \quad (90\%)$$
> 
> **Step 3: Calculate Predictive Values**
> $$\text{PPV} = \frac{90}{90 + 90} = \frac{90}{180} = 0.50 \quad (50\%)$$
> $$\text{NPV} = \frac{810}{810 + 10} = \frac{810}{820} = 0.988 \quad (98.8\%)$$
> 
> **Interpretation:**
> - The test correctly identifies 90% of COVID+ cases (sensitivity).
> - The test correctly identifies 90% of COVID− cases (specificity).
> - **Critical Insight:** If you test positive, there's only a 50% chance you actually have COVID!
> 
> **Why is PPV so low?**
> - Prevalence is only 10%.
> - With 900 healthy people and 10% FPR, you get 90 false positives.
> - These 90 false positives dilute the 90 true positives.
> 
> **At 50% Prevalence:**
> $$\text{PPV} = \frac{0.90 \times 0.50}{0.90 \times 0.50 + 0.10 \times 0.50} = \frac{0.45}{0.50} = 0.90$$
> PPV jumps to 90%!

---

## Interpretation Guide

| Metric | Value | Interpretation | Clinical Use |
|--------|-------|----------------|--------------|
| Sensitivity | 0.95 | Catches 95% of true cases | Good for **screening** (don't miss disease) |
| Sensitivity | 0.50 | Misses half of true cases | Poor screening test |
| Specificity | 0.99 | Only 1% false positives | Good for **confirmation** |
| Specificity | 0.70 | 30% false alarm rate | Too many unnecessary follow-ups |
| PPV | 0.10 | 90% of positive tests are wrong! | Low-prevalence setting |
| NPV | 0.99 | Negative test almost certainly rules out disease | High confidence for negatives |

### Optimal Trade-off Guidelines

| Clinical Scenario | Priority | Example |
|-------------------|----------|---------|
| **Screening (rule-out)** | High Sensitivity | Cancer screening, infections |
| **Confirmation (rule-in)** | High Specificity | Confirmatory biopsies |
| **Balanced** | Maximize Youden's J | General classification |

---

## Common Pitfall Example

> [!warning] The Base Rate Fallacy
> **Scenario:** A disease test has 99% sensitivity and 99% specificity. Amazing, right?
> 
> **Prevalence:** 1 in 10,000 (0.01%)
> 
> **Testing 1,000,000 people:**
> - True positives: $100 \times 0.99 = 99$
> - False positives: $999,900 \times 0.01 = 9,999$
> 
> **PPV:**
> $$\text{PPV} = \frac{99}{99 + 9999} = \frac{99}{10098} \approx 0.01 \quad (1\%)$$
> 
> **Reality:** 99% of positive results are FALSE POSITIVES!
> 
> **Lesson:** Always consider prevalence when interpreting test results.

---

## Related Concepts

**Metrics:**
- [[stats/04_Machine_Learning/ROC & AUC\|ROC & AUC]] - Aggregate performance across thresholds
- [[stats/04_Machine_Learning/Confusion Matrix\|Confusion Matrix]] - Full breakdown of predictions
- [[stats/04_Machine_Learning/Precision-Recall Curve\|Precision-Recall Curve]] - Alternative for imbalanced data

**Theory:**
- [[stats/01_Foundations/Bayes' Theorem\|Bayes' Theorem]] - Converting between conditional probabilities
- [[stats/02_Hypothesis_Testing/Type I & Type II Errors\|Type I & Type II Errors]] - Error framework

**Applications:**
- [[stats/03_Regression_Analysis/Binary Logistic Regression\|Binary Logistic Regression]] - Probability-based classification
- Medical Diagnostics - Clinical test evaluation

---

## References

- **Book:** Altman, D. G., & Bland, J. M. (1994). Diagnostic tests 1: Sensitivity and specificity. *BMJ*, 308(6943), 1552. [BMJ Link](https://doi.org/10.1136/bmj.308.6943.1552)
- **Book:** Zweig, M. H., & Campbell, G. (1993). Receiver-operating characteristic (ROC) plots: a fundamental evaluation tool in clinical medicine. *Clinical Chemistry*, 39(4), 561-577. [Oxford Link](https://doi.org/10.1093/clinchem/39.4.561)
- **Book:** Sox, H. C., Higgins, M. C., & Owens, D. K. (2013). *Medical Decision Making* (2nd ed.). Wiley-Blackwell. [Wiley Link](http://www.wiley.com/WileyCDA/WileyTitle/productCd-0470658665.html)
- **Article:** Lalkhen, A. G., & McCluskey, A. (2008). Clinical tests: sensitivity and specificity. *Continuing Education in Anaesthesia Critical Care & Pain*, 8(6), 221-223. [Oxford Link](https://doi.org/10.1093/bjaceaccp/mkn041)
- **Article:** Parikh, R., Mathai, A., Parikh, S., Sekhar, G. C., & Thomas, R. (2008). Understanding and using sensitivity, specificity and predictive values. *Indian Journal of Ophthalmology*, 56(1), 45. [IJO Link](https://doi.org/10.4103/0301-4738.37595)
