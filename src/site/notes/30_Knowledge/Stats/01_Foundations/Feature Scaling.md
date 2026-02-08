---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/feature-scaling/","tags":["probability","foundations"]}
---


## Definition

> [!abstract] Core Statement
> **Feature Scaling** transforms numerical features to a common scale without distorting differences in ranges. It is **critical** for algorithms that compute distances or use gradient-based optimization.

**Intuition (ELI5):** Imagine comparing houses by "number of bedrooms" (1-5) and "price" (100K-1M). Without scaling, price would completely dominate any distance calculation because its numbers are 100,000× larger. Scaling puts both features on equal footing.

**Why It Matters:**
- **Distance-based algorithms:** KNN, K-Means, SVM rely on distances — large-scale features dominate.
- **Gradient descent:** Differently-scaled features create elongated contours → slow convergence.
- **Regularization:** L1/L2 penalties are scale-dependent.

---

## When to Use

> [!success] Scale Features When Using...
> - **Distance-based:** KNN, K-Means, SVM, DBSCAN
> - **Gradient-based:** Linear/Logistic Regression, Neural Networks
> - **Regularization:** Ridge, Lasso, Elastic Net
> - **PCA:** Variance is scale-dependent
> - **Neural Networks:** Activation functions expect standardized input

> [!failure] Do NOT Scale When Using...
> - **Tree-based models:** Decision Trees, Random Forest, XGBoost — splits are based on thresholds, not distances.
> - **Naive Bayes:** Probability-based, not distance-based.
> - **Already normalized data:** Percentages, proportions.

---

## Theoretical Background

### The Three Main Techniques

| Technique | Formula | Range | Best For |
|-----------|---------|-------|----------|
| **Standardization (Z-score)** | $z = \frac{x - \mu}{\sigma}$ | Unbounded (~[-3, 3]) | Most ML algorithms |
| **Min-Max Scaling** | $x' = \frac{x - \min}{\max - \min}$ | [0, 1] | Neural networks, image data |
| **Robust Scaling** | $x' = \frac{x - Q_2}{Q_3 - Q_1}$ | Unbounded | Data with outliers |

---

### 1. Standardization (Z-Score Normalization)

$$
z = \frac{x - \mu}{\sigma}
$$

**Result:** Mean = 0, Standard Deviation = 1

**Properties:**
- No bounded range (outliers can be far from 0)
- Preserves outliers (doesn't squash them)
- Best for algorithms assuming Gaussian-like distributions

### 2. Min-Max Scaling (Normalization)

$$
x' = \frac{x - x_{min}}{x_{max} - x_{min}}
$$

**Result:** All values in [0, 1]

**Properties:**
- Bounded range
- Highly sensitive to outliers (one outlier compresses all data)
- Good for image pixels, neural network inputs

### 3. Robust Scaling

$$
x' = \frac{x - \text{median}}{IQR}
$$

Where IQR = Q3 - Q1 (interquartile range)

**Properties:**
- Uses median and IQR (robust statistics)
- Outliers don't affect scale
- Best for data with significant outliers

---

## Assumptions & Diagnostics

- [ ] **Numerical Features Only:** Scaling applies to continuous variables, not categorical.
- [ ] **Fit on Training Set Only:** Never fit scaler on test set (data leakage!).
- [ ] **Apply Same Transformation to Test Set:** Use `transform()`, not `fit_transform()`.

### When to Use Which?

```
Does your data have outliers?
├─ Many outliers → Robust Scaling
│
└─ Few or no outliers
    ├─ Is algorithm sensitive to bounds (NN, image)? → Min-Max
    │
    └─ General ML algorithm → Standardization
```

---

## Implementation

### Python

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

# Sample data
np.random.seed(42)
data = pd.DataFrame({
    'age': np.random.randint(18, 80, 100),
    'income': np.random.randint(20000, 200000, 100),
    'spending_score': np.random.randint(1, 100, 100)
})

# Add an outlier
data.loc[100] = [25, 10000000, 50]  # Income outlier

print("Original Data Statistics:")
print(data.describe())

# ========== TRAIN-TEST SPLIT FIRST ==========
X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)

# ========== 1. STANDARDIZATION ==========
std_scaler = StandardScaler()
X_train_std = std_scaler.fit_transform(X_train)  # Fit on TRAINING only
X_test_std = std_scaler.transform(X_test)        # Transform test

print("\nStandardization (Training):")
print(f"Mean: {X_train_std.mean(axis=0)}")  # Should be ~0
print(f"Std:  {X_train_std.std(axis=0)}")   # Should be ~1

# ========== 2. MIN-MAX SCALING ==========
minmax_scaler = MinMaxScaler()
X_train_mm = minmax_scaler.fit_transform(X_train)
X_test_mm = minmax_scaler.transform(X_test)

print("\nMin-Max Scaling (Training):")
print(f"Min: {X_train_mm.min(axis=0)}")  # Should be 0
print(f"Max: {X_train_mm.max(axis=0)}")  # Should be 1

# ========== 3. ROBUST SCALING ==========
robust_scaler = RobustScaler()
X_train_robust = robust_scaler.fit_transform(X_train)
X_test_robust = robust_scaler.transform(X_test)

print("\nRobust Scaling (Training):")
print(f"Median: {np.median(X_train_robust, axis=0)}")  # Should be ~0

# ========== COMPARISON WITH OUTLIER ==========
print("\n=== Effect of Outlier on Income Scaling ===")
print(f"Original Income Range: {data['income'].min()} to {data['income'].max()}")

# Find the outlier row in scaled data
print(f"\nOutlier Income (raw): 10,000,000")
if 100 in X_train.index:
    idx = list(X_train.index).index(100)
    print(f"  StandardScaler: {X_train_std[idx, 1]:.2f}")  # Very high
    print(f"  MinMaxScaler: {X_train_mm[idx, 1]:.2f}")    # ~1.0, others compressed
    print(f"  RobustScaler: {X_train_robust[idx, 1]:.2f}") # High but doesn't affect others

# ========== PIPELINE FOR PRODUCTION ==========
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

# Always put scaling inside pipeline!
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])

# pipeline.fit(X_train, y_train)  # Scaler fits on training only
# pipeline.predict(X_test)         # Scaler transforms test automatically
```

### R

```r
# Sample data
set.seed(42)
data <- data.frame(
  age = sample(18:80, 100, replace = TRUE),
  income = sample(20000:200000, 100, replace = TRUE),
  spending_score = sample(1:100, 100, replace = TRUE)
)

# Add outlier
data <- rbind(data, c(25, 10000000, 50))

print("Original Data Summary:")
summary(data)

# Train-test split
set.seed(42)
train_idx <- sample(1:nrow(data), 0.8 * nrow(data))
train <- data[train_idx, ]
test <- data[-train_idx, ]

# ========== 1. STANDARDIZATION (scale function) ==========
# Get parameters from training set
train_mean <- colMeans(train)
train_sd <- apply(train, 2, sd)

# Apply to both sets
train_std <- scale(train, center = train_mean, scale = train_sd)
test_std <- scale(test, center = train_mean, scale = train_sd)

cat("\nStandardization Check (Training):\n")
cat("Means:", colMeans(train_std), "\n")
cat("SDs:", apply(train_std, 2, sd), "\n")

# ========== 2. MIN-MAX SCALING ==========
minmax_scale <- function(x, min_val, max_val) {
  (x - min_val) / (max_val - min_val)
}

train_min <- apply(train, 2, min)
train_max <- apply(train, 2, max)

train_mm <- mapply(minmax_scale, train, train_min, train_max)
test_mm <- mapply(minmax_scale, test, train_min, train_max)

cat("\nMin-Max Check (Training):\n")
cat("Mins:", apply(train_mm, 2, min), "\n")
cat("Maxs:", apply(train_mm, 2, max), "\n")

# ========== 3. ROBUST SCALING ==========
robust_scale <- function(x, median_val, iqr_val) {
  (x - median_val) / iqr_val
}

train_median <- apply(train, 2, median)
train_iqr <- apply(train, 2, IQR)

train_robust <- mapply(robust_scale, train, train_median, train_iqr)

# ========== USING CARET ==========
library(caret)

# preProcess handles this elegantly
preproc <- preProcess(train, method = c("center", "scale"))
train_scaled <- predict(preproc, train)
test_scaled <- predict(preproc, test)  # Uses training parameters!
```

---

## Interpretation Guide

| Scenario | Recommendation | Why |
|----------|----------------|-----|
| General ML (SVM, KNN, PCA) | StandardScaler | Assumes Gaussian-ish distribution |
| Neural Networks | Either (MinMax or Standard) | NN weights adapt; just need consistency |
| Data with many outliers | RobustScaler | Median/IQR not affected by outliers |
| Image data (0-255 pixels) | MinMax to [0, 1] | Bounded input expected |
| Already on similar scales | Consider skipping | May not need if ranges are comparable |

### Comparison of Techniques

| Feature | StandardScaler | MinMaxScaler | RobustScaler |
|---------|----------------|--------------|--------------|
| **Outlier Sensitivity** | Low | Very High | None |
| **Range** | Unbounded | [0, 1] | Unbounded |
| **Center** | Mean = 0 | (depends) | Median = 0 |
| **Spread** | Std = 1 | (depends) | IQR = 1 |

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Fitting Scaler on Entire Dataset (Data Leakage)**
> - *Problem:* `scaler.fit_transform(X)` before train-test split.
> - *Result:* Test set statistics leak into training → optimistic results.
> - *Solution:* Split first, fit on train only: `fit(X_train)`, `transform(X_test)`.
>
> **2. Scaling Categorical Variables**
> - *Problem:* One-hot encoded features (0/1) get standardized.
> - *Result:* Meaningless scaled values; interpretation lost.
> - *Solution:* Scale only numeric features; leave binary/categorical as-is.
>
> **3. Using MinMax with Outliers**
> - *Problem:* One outlier at 1M compresses all normal data to tiny range.
> - *Result:* Most data points are nearly 0; information lost.
> - *Solution:* Use RobustScaler or remove outliers first.
>
> **4. Not Scaling Before Regularization**
> - *Problem:* Using Lasso/Ridge without scaling.
> - *Result:* Penalty hits large-scale features harder → biased selection.
> - *Solution:* Always standardize before regularized regression.

---

## Worked Numerical Example

> [!example] Scaling Customer Data for KNN
> **Raw Data:**
> | Customer | Age | Income |
> |----------|-----|--------|
> | A | 25 | 50,000 |
> | B | 45 | 100,000 |
> | C | 30 | 75,000 |
>
> **Problem:** Finding similar customers using KNN.
>
> **Step 1: Distance WITHOUT Scaling**
> Between A and B:
> $$d = \sqrt{(45-25)^2 + (100000-50000)^2} = \sqrt{400 + 2.5 \times 10^9} \approx 50,000$$
>
> Income completely dominates! Age difference is invisible.
>
> **Step 2: Standardize**
> ```
> Age:    μ=33.3, σ=10.4
> Income: μ=75K, σ=25K
>
> | Customer | Age (std) | Income (std) |
> |----------|-----------|--------------|
> | A        | -0.80     | -1.00        |
> | B        | +1.12     | +1.00        |
> | C        | -0.32     | 0.00         |
> ```
>
> **Step 3: Distance WITH Scaling**
> Between A and B:
> $$d = \sqrt{(1.12-(-0.80))^2 + (1.00-(-1.00))^2} = \sqrt{3.68 + 4.00} = 2.77$$
>
> Now both features contribute equally!
>
> **Conclusion:** Without scaling, KNN would match customers by income only. With scaling, age matters too.

---

## Related Concepts

**Algorithms Requiring Scaling:**
- [[30_Knowledge/Stats/04_Supervised_Learning/K-Nearest Neighbors (KNN)\|K-Nearest Neighbors (KNN)]] — Distance-based
- [[30_Knowledge/Stats/04_Supervised_Learning/Support Vector Machines (SVM)\|Support Vector Machines (SVM)]] — Kernel distance
- [[30_Knowledge/Stats/05_Unsupervised_Learning/PCA (Principal Component Analysis)\|PCA (Principal Component Analysis)]] — Variance-based

**Data Leakage:**
- [[30_Knowledge/Stats/01_Foundations/Data Leakage\|Data Leakage]] — Why fit only on training set
- [[30_Knowledge/Stats/04_Supervised_Learning/Cross-Validation\|Cross-Validation]] — Each fold needs separate scaling

**Related Preprocessing:**
- Log Transformations — For skewed distributions
- [[30_Knowledge/Stats/04_Supervised_Learning/Encoding Categorical Variables\|Encoding Categorical Variables]] — For non-numeric features

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## References

- **Book:** Han, J., Kamber, M., & Pei, J. (2011). *Data Mining: Concepts and Techniques* (3rd ed.). Morgan Kaufmann. [Elsevier](https://www.elsevier.com/books/data-mining-concepts-and-techniques/han/978-0-12-381479-1)
- **Book:** Zheng, A., & Casari, A. (2018). *Feature Engineering for Machine Learning*. O'Reilly Media. [O'Reilly](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)
- **Book:** Witten, I. H., Frank, E., & Hall, M. A. (2011). *Data Mining: Practical Machine Learning Tools and Techniques* (3rd ed.). Morgan Kaufmann. [Elsevier](https://www.elsevier.com/books/data-mining-practical-machine-learning-tools-and-techniques/witten/978-0-12-374856-0)
