---
{"dg-publish":true,"permalink":"/stats/03-regression-analysis/hosmer-lemeshow-test/","tags":["Diagnostics","Regression","Calibration"]}
---


# Hosmer-Lemeshow Test

## Overview

> [!abstract] Definition
> The **Hosmer-Lemeshow Test** is a statistical test for goodness of fit for logistic regression models. It assesses whether the observed event rates match expected event rates in subgroups of the model population.

---

## 1. Procedure

1. **Predict Probabilities:** Calculate predicted probabilities for all observations.
2. **Group Data:** Sort observations by predicted probability and divide them into $g$ groups (typically deciles, $g=10$).
3. **Compare:** In each group, calculate the expected number of events versus observed events.
4. **Chi-Square Statistic:**
   $$ H = \sum_{j=1}^{g} \frac{(O_j - E_j)^2}{N_j \pi_j (1 - \pi_j)} $$
   Where $O_j$ is observed events, $E_j$ is expected events, and $\pi_j$ is the average predicted probability in group $j$.

---

## 2. Hypothesis

- $H_0$: The model fits the data well (No significant difference between observed and predicted).
- $H_1$: The model does not fit the data well.

**Interpretation:**
- **p > 0.05:** Evidence of good fit (Fail to reject $H_0$).
- **p < 0.05:** Evidence of poor fit (Reject $H_0$).

> [!warning] Limitation
> The test is sensitive to grouping method and sample size. It is often recommended to use it alongside calibration plots.

---

## 3. Python Implementation

*Note: Not available in standard sklearn. Custom implementation or libraries like `scikit-learn-extra` or statistical packages are needed.*

```python
# Conceptual implementation
# Group data by deciles of predicted probability
# Calculate Chi-square between observed and expected counts
```

---

## 4. Related Concepts

- [[stats/03_Regression_Analysis/Binary Logistic Regression\|Binary Logistic Regression]] - The model being tested.
- [[stats/04_Machine_Learning/ROC & AUC\|ROC & AUC]] - Measures discrimination (distinguishing classes) rather than calibration (accuracy of probability).
- [[stats/04_Machine_Learning/Confusion Matrix\|Confusion Matrix]] - Classification performance.