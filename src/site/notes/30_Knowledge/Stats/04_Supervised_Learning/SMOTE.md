---
{"dg-publish":true,"permalink":"/30-knowledge/stats/04-supervised-learning/smote/","tags":["machine-learning","supervised","imbalanced-data","preprocessing","oversampling"]}
---

## Definition

> [!abstract] Core Statement
> **SMOTE (Synthetic Minority Over-sampling Technique)** addresses class imbalance by generating ==synthetic samples for the minority class==. Instead of duplicating existing examples, it creates new ones by interpolating between existing minority instances and their k-nearest neighbors.

---

> [!tip] Intuition (ELI5): Making New Friends
> Imagine you have 100 photos of cats and only 5 photos of dogs. Instead of copying dog photos (boring), SMOTE creates *new* dog photos by blending features of existing dogs. It picks two similar dogs and makes a "mix" of them.

---

## Purpose

1. **Balance class distribution** without simple duplication
2. **Improve classifier performance** on minority class
3. **Reduce overfitting** compared to random oversampling

---

## When to Use

> [!success] Use SMOTE When...
> - Dataset is **imbalanced** (minority < 10-20%)
> - Minority class examples are **clustered** (similar to each other)
> - You want to **oversample** rather than undersample

---

## When NOT to Use

> [!danger] Do NOT Use SMOTE When...
> - Minority samples are **very noisy** or overlapping with majority
> - Features are **categorical** (use SMOTE-NC instead)
> - Dataset is extremely small (< 100 minority samples)
> - Minority class is already **well-separated**

---

## How It Works

### Algorithm

```
For each minority sample x:
    1. Find k nearest neighbors (in minority class)
    2. Randomly select one neighbor x_nn
    3. Generate synthetic sample:
       x_new = x + rand(0,1) * (x_nn - x)
```

### Visual

```
     x_nn ●
          |\
          | \  ← New synthetic point on the line
          |  ○ x_new
          | /
          |/
     x    ●
```

---

## Python Implementation

```python
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np

# Create imbalanced dataset
X, y = make_classification(n_samples=1000, n_features=20,
                          n_informative=10, n_redundant=5,
                          n_classes=2, weights=[0.95, 0.05],
                          random_state=42)

print(f"Original distribution: {np.bincount(y)}")
# Output: [950  50]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply SMOTE
smote = SMOTE(random_state=42, k_neighbors=5)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print(f"After SMOTE: {np.bincount(y_resampled)}")
# Output: [665 665]

# Train and evaluate
clf = RandomForestClassifier(random_state=42)

# Without SMOTE
clf.fit(X_train, y_train)
print("Without SMOTE:")
print(classification_report(y_test, clf.predict(X_test)))

# With SMOTE
clf.fit(X_resampled, y_resampled)
print("With SMOTE:")
print(classification_report(y_test, clf.predict(X_test)))
```

---

## R Implementation

```r
library(smotefamily)
library(caret)

# Create imbalanced data
set.seed(42)
n <- 1000
X <- matrix(rnorm(n * 10), ncol = 10)
y <- factor(c(rep(1, 950), rep(0, 50)))
df <- data.frame(X, class = y)

# Apply SMOTE
smote_result <- SMOTE(df[, 1:10], df$class, K = 5)
df_balanced <- smote_result$data

table(df_balanced$class)
# Should be balanced
```

---

## Variants

| Variant | Description |
|---------|-------------|
| **SMOTE-NC** | Handles categorical + numerical features |
| **Borderline-SMOTE** | Only oversamples near decision boundary |
| **ADASYN** | Focuses on hard-to-learn examples |
| **SMOTE-ENN** | SMOTE + Edited Nearest Neighbors (cleaning) |

---

## Limitations

> [!warning] Pitfalls
> 1. **Noise amplification:** If minority samples are noisy, SMOTE creates more noise
> 2. **Overfitting:** Can still overfit if minority cluster is very small
> 3. **Doesn't help with overlap:** If classes overlap, synthetic samples make it worse
> 4. **Apply ONLY to training data:** Never apply to test set!

---

## Best Practices

1. **Always apply SMOTE before splitting** or within cross-validation
2. **Combine with undersampling** (e.g., SMOTE-ENN)
3. **Tune k_neighbors** parameter
4. **Check cluster quality** of minority class first

---

## Related Concepts

- [[30_Knowledge/Stats/04_Supervised_Learning/Imbalanced Data\|Imbalanced Data]] - The underlying problem
- [[30_Knowledge/Stats/04_Supervised_Learning/K-Nearest Neighbors (KNN)\|K-Nearest Neighbors (KNN)]] - Used for neighbor selection
- Precision and Recall - Key metrics for imbalanced data
- [[30_Knowledge/Stats/04_Supervised_Learning/Cross-Validation\|Cross-Validation]] - Apply SMOTE inside CV folds

---

## References

1. Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*. [Paper](https://arxiv.org/abs/1106.1813)

2. He, H., & Garcia, E. A. (2009). Learning from Imbalanced Data. *IEEE TKDE*. [IEEE](https://ieeexplore.ieee.org/document/5128907)
