---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/data-leakage/","tags":["probability","foundations"]}
---


## Definition

> [!abstract] Core Statement
> **Data Leakage** occurs when information from ==outside the training dataset is used to create the model==, leading to overly optimistic performance estimates that don't generalize.

---

> [!tip] Intuition (ELI5)
> It's like taking a test where you accidentally saw the answers beforehand. You'll score 100% in practice, but fail when it counts because you never actually learned.

---

## Types

| Type | Example | Danger Level |
|------|---------|--------------|
| **Target Leakage** | Feature contains future information | 🔴 High |
| **Train-Test Contamination** | Test data used in preprocessing | 🔴 High |
| **Temporal Leakage** | Future data predicts past | 🔴 High |
| **Feature Engineering Leakage** | Aggregates computed on full data | 🟡 Medium |

---

## Real-World Examples

> [!warning] Famous Leakage Cases
>
> **1. Hospital Readmission**
> - Feature: "Appointment scheduled with specialist"
> - *Problem:* Only exists because patient was readmitted
>
> **2. Credit Default**
> - Feature: "Account closed date"
> - *Problem:* Only populated after default
>
> **3. Churn Prediction**
> - Feature: "Days since last login"
> - *Problem:* Calculated including future data

---

## Prevention: Use Pipelines

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# ========== WRONG: Leakage ==========
# Scaling BEFORE split leaks test statistics into training
X_scaled = StandardScaler().fit_transform(X)  # BAD!
X_train, X_test = train_test_split(X_scaled, y)

# ========== CORRECT: Pipeline ==========
# Pipeline ensures fit only on train data
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Cross-validation respects pipeline
cv_scores = cross_val_score(pipe, X_train, y_train, cv=5)
print(f"CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Final evaluation
pipe.fit(X_train, y_train)
test_score = pipe.score(X_test, y_test)
print(f"Test Accuracy: {test_score:.3f}")
```

---

## Time Series: Special Care

```python
from sklearn.model_selection import TimeSeriesSplit

# ========== WRONG FOR TIME SERIES ==========
# Random split mixes future and past
train_test_split(X, y, random_state=42)  # BAD for time series!

# ========== CORRECT ==========
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # Train is always BEFORE test in time
```

---

## Detection Checklist

| Red Flag | Check |
|----------|-------|
| **Too good to be true** | Test AUC > 0.95 on messy data? |
| **Feature timing** | Can you know this at prediction time? |
| **Preprocessing order** | Was test data touched before split? |
| **Production gap** | Model suddenly fails in deployment? |

```python
# Automatic leakage detection
def check_feature_timing(df, target_col, date_col):
    """Check if features were created after target"""
    for col in df.columns:
        if df[col].isna().sum() < len(df) * 0.1:
            continue
        # If missing in past, present in future → suspicious
        correlation = df[col].notna().corr(df[target_col])
        if abs(correlation) > 0.3:
            print(f"⚠️ Suspicious: {col} (corr with target: {correlation:.2f})")
```

---

## Common Pitfalls

> [!failure] Subtle Leakage Sources
>
> **1. Group-level features**
> - Computing mean by customer → leaks if customer in both train/test
>
> **2. Feature selection on full data**
> - Selecting top features before split
>
> **3. Imputation with global statistics**
> - Filling NA with dataset mean (includes test!)

---

## Related Concepts

- [[30_Knowledge/Stats/04_Supervised_Learning/Cross-Validation\|Cross-Validation]] — Proper validation prevents leakage
- [[30_Knowledge/Stats/01_Foundations/Feature Selection\|Feature Selection]] — Order matters
- [[30_Knowledge/Stats/01_Foundations/Feature Scaling\|Feature Scaling]] — Must be in pipeline
- [[30_Knowledge/Stats/04_Supervised_Learning/Learning Curves\|Learning Curves]] — Detect overfitting

---

## When to Use

> [!success] Use Data Leakage When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## Python Implementation

```python
import numpy as np
import pandas as pd

# Example implementation of Data Leakage
# See documentation for details

data = np.random.randn(100)
print(f"Mean: {np.mean(data):.3f}")
print(f"Std: {np.std(data):.3f}")
```

---

## R Implementation

```r
# Data Leakage in R
set.seed(42)

# Example implementation
data <- rnorm(100)
summary(data)
```

---

## References

- **Article:** Kaufman, S., et al. (2012). Leakage in data mining. *ACM TKDD*, 6(4), 1-21. [ACM Link](https://doi.org/10.1145/2382577.2382579)
- **Blog:** Kaggle. (2019). Data Leakage. [Kaggle Guide](https://www.kaggle.com/code/alexisbcook/data-leakage)
- **Paper:** Rosset, S., & Inger, A. (2000). KDD-cup 99: Knowledge discovery in a charitable organization's donor database.

