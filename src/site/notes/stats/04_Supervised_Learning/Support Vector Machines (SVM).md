---
{"dg-publish":true,"permalink":"/stats/04-supervised-learning/support-vector-machines-svm/","tags":["Machine-Learning","Classification","Algorithms","Kernel-Methods"]}
---


## Definition

> [!abstract] Core Statement
> **Support Vector Machines (SVM)** find the **hyperplane** that maximizes the **margin** between two classes. Points closest to the hyperplane are called **support vectors** — they alone determine the decision boundary.

![SVM: Finding the Maximum Margin Hyperplane](https://commons.wikimedia.org/wiki/Special:FilePath/Svm_separating_hyperplanes_(SVG).svg)

**Intuition (ELI5):** Imagine separating red and blue marbles on a table with a stick. SVM finds the stick position that leaves the biggest gap on both sides. Only the marbles touching the gap matter — move the others, the stick stays the same.

**Key Concepts:**
- **Hyperplane:** Decision boundary (line in 2D, plane in 3D)
- **Margin:** Distance between hyperplane and nearest points
- **Support Vectors:** The critical points that define the margin

---

## When to Use

> [!success] Use SVM When...
> - You have **high-dimensional data** (text classification, genomics).
> - Dataset is **small to medium** (<100K samples).
> - Classes are **clearly separable** (possibly after kernel transformation).
> - You need **strong theoretical guarantees** (margin maximization).
> - Data has **complex, non-linear** decision boundaries (with kernels).

> [!failure] Do NOT Use SVM When...
> - Dataset is **very large** (>100K samples) — training is O(n²) to O(n³).
> - Data is **highly noisy** with overlapping classes.
> - You need **probability outputs** natively (SVM gives distances, not probabilities).
> - **Interpretability** is critical — kernelized SVM is a black box.
> - Features are **not scaled** — SVM is sensitive to scale.

---

## Theoretical Background

### Linear SVM: Maximum Margin Classifier

For linearly separable data, find hyperplane $w^Tx + b = 0$ that maximizes margin:

$$
\max_{w, b} \frac{2}{\|w\|}
$$

Subject to constraints:
$$
y_i(w^Tx_i + b) \geq 1 \quad \forall i
$$

### Soft Margin SVM (C Parameter)

Allow some misclassifications with slack variables $\xi_i$:

$$
\min_{w, b, \xi} \frac{1}{2}\|w\|^2 + C\sum_{i=1}^{n}\xi_i
$$

| C Value | Effect | Interpretation |
|---------|--------|----------------|
| **High C** | Penalty for misclassification | Narrow margin, less regularization |
| **Low C** | Tolerate misclassification | Wide margin, more regularization |

### The Kernel Trick

Transform data to higher dimensions where it becomes linearly separable, without explicitly computing the transformation:

$$
K(x_i, x_j) = \phi(x_i)^T \phi(x_j)
$$

| Kernel | Formula | Use Case |
|--------|---------|----------|
| **Linear** | $K(x,y) = x^Ty$ | Linearly separable data, text |
| **Polynomial** | $K(x,y) = (x^Ty + c)^d$ | Interaction effects |
| **RBF (Gaussian)** | $K(x,y) = e^{-\gamma\|x-y\|^2}$ | Most common, any shape |
| **Sigmoid** | $K(x,y) = \tanh(\alpha x^Ty + c)$ | Neural network-like |

### RBF Kernel Parameters

$$
K(x, y) = \exp\left(-\gamma \|x - y\|^2\right)
$$

| γ (gamma) | Effect | Risk |
|-----------|--------|------|
| **High γ** | Each point influences only nearby points | Overfitting (wiggly boundary) |
| **Low γ** | Each point influences distant points | Underfitting (smooth boundary) |

---

## Assumptions & Diagnostics

- [ ] **Feature Scaling:** CRITICAL — SVM uses distances. Standardize always.
- [ ] **Binary Classification:** Native SVM is binary; multi-class via OvO or OvR.
- [ ] **Kernel Selection:** RBF is default; try linear for high-dimensional sparse data.
- [ ] **No Heavy Class Imbalance:** Use `class_weight='balanced'` if imbalanced.

### Diagnostics

| Diagnostic | Purpose | Check |
|------------|---------|-------|
| **Support vector count** | Model complexity | Many SVs → overfitting or noisy data |
| **Decision boundary plot** | Visualize separation | Wiggly → high γ |
| **Cross-validation** | Tune C and γ | GridSearchCV recommended |

---

## Implementation

### Python

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import make_moons
from sklearn.metrics import classification_report

# Generate non-linear data
X, y = make_moons(n_samples=500, noise=0.2, random_state=42)

# ========== STEP 1: SCALE FEATURES (CRITICAL!) ==========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ========== STEP 2: TRAIN SVM WITH RBF KERNEL ==========
svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm.fit(X_train, y_train)

print(f"Accuracy: {svm.score(X_test, y_test):.3f}")
print(f"Number of support vectors: {len(svm.support_vectors_)}")

# ========== STEP 3: HYPERPARAMETER TUNING ==========
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1]
}

grid_search = GridSearchCV(
    SVC(kernel='rbf', random_state=42),
    param_grid, cv=5, scoring='accuracy', n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")

# ========== STEP 4: VISUALIZATION ==========
def plot_svm_boundary(model, X, y, title):
    plt.figure(figsize=(8, 6))
    
    # Create mesh grid
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict on mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='black')
    
    # Highlight support vectors
    plt.scatter(model.support_vectors_[:, 0], 
                model.support_vectors_[:, 1],
                s=100, facecolors='none', edgecolors='green', linewidth=2,
                label='Support Vectors')
    plt.legend()
    plt.title(title)
    plt.show()

plot_svm_boundary(grid_search.best_estimator_, X_train, y_train, 
                  f"SVM Decision Boundary (C={grid_search.best_params_['C']})")

# ========== STEP 5: GET PROBABILITIES ==========
svm_proba = SVC(kernel='rbf', C=1, gamma='scale', probability=True)
svm_proba.fit(X_train, y_train)
probs = svm_proba.predict_proba(X_test)[:5]
print("Probabilities (first 5):\n", probs)
```

### R

```r
library(e1071)
library(caret)

# Generate non-linear data
set.seed(42)
n <- 500
theta <- runif(n, 0, 2*pi)
r <- c(rep(1, n/2), rep(3, n/2)) + rnorm(n, 0, 0.3)
X <- cbind(r * cos(theta), r * sin(theta))
y <- factor(c(rep(0, n/2), rep(1, n/2)))
df <- data.frame(X1 = X[,1], X2 = X[,2], y = y)

# ========== STEP 1: SCALE FEATURES ==========
preproc <- preProcess(df[, 1:2], method = c("center", "scale"))
df_scaled <- predict(preproc, df)

# Train-test split
set.seed(42)
train_idx <- sample(1:n, 0.8*n)
train <- df_scaled[train_idx, ]
test <- df_scaled[-train_idx, ]

# ========== STEP 2: TRAIN SVM ==========
svm_model <- svm(y ~ ., data = train, 
                 kernel = "radial", # RBF
                 cost = 1,          # C parameter
                 gamma = 0.5)

print(svm_model)
pred <- predict(svm_model, test)
cat("Accuracy:", mean(pred == test$y), "\n")

# ========== STEP 3: HYPERPARAMETER TUNING ==========
tune_result <- tune(svm, y ~ ., data = train,
                    kernel = "radial",
                    ranges = list(
                      cost = c(0.1, 1, 10, 100),
                      gamma = c(0.01, 0.1, 1)
                    ))

summary(tune_result)
best_model <- tune_result$best.model

# ========== STEP 4: VISUALIZATION ==========
plot(best_model, train, X1 ~ X2,
     svSymbol = "x", dataSymbol = "o",
     main = "SVM Decision Boundary")
```

---

## Interpretation Guide

| Output | Example Value | Interpretation | Edge Case/Warning |
|--------|---------------|----------------|-------------------|
| **# Support Vectors** | 50 / 500 | 10% of data are SVs — compact model. | >50% SVs → noisy data or underfitting. |
| **# Support Vectors** | 450 / 500 | 90% are SVs — nearly memorizing data. | Model too complex or data not separable. |
| **C = 0.01** | | Wide margin, tolerant to violations. | May underfit; check accuracy. |
| **C = 100** | | Narrow margin, strict classification. | Risk of overfitting; try lower C. |
| **gamma = 'scale'** | | Auto-computed: 1/(n_features × var). | Good default; tune if needed. |
| **gamma = 10** | | Very high influence radius → wiggly. | Overfitting; try lower gamma. |
| **Linear kernel outperforms RBF** | | Data is linearly separable. | Simpler model preferred. |

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Forgetting to Scale Features**
> - *Problem:* Feature A ranges 0–1, Feature B ranges 0–10000.
> - *Result:* SVM dominated by Feature B; Feature A ignored.
> - *Solution:* Always use `StandardScaler` before SVM.
>
> **2. Using Default Parameters**
> - *Problem:* Using C=1, gamma='scale' without tuning.
> - *Result:* Suboptimal decision boundary.
> - *Solution:* Grid search over C and gamma — they interact!
>
> **3. Applying SVM to Huge Datasets**
> - *Problem:* Training SVM on 1 million samples.
> - *Result:* Kernel matrix is n×n = 10^12 entries. Memory explodes.
> - *Solution:* Use LinearSVC for linear kernels, or sample data.
>
> **4. Expecting Probabilities**
> - *Problem:* Calling `predict_proba` without `probability=True`.
> - *Reality:* SVM outputs distances to hyperplane, not probabilities.
> - *Solution:* Set `probability=True` — but this is computationally expensive.

---

## Worked Numerical Example

> [!example] Manually Calculating SVM Decision
> **Scenario:** 2D data with learned hyperplane: $w = [1, 2]$, $b = -3$
>
> **Decision function:**
> $$f(x) = w^T x + b = 1 \cdot x_1 + 2 \cdot x_2 - 3$$
>
> **Step 1: Classify Point A = (2, 1)**
> $$f(A) = 1(2) + 2(1) - 3 = 2 + 2 - 3 = +1$$
> Result: $f(A) > 0$ → **Class +1**
>
> **Step 2: Classify Point B = (1, 0)**
> $$f(B) = 1(1) + 2(0) - 3 = 1 + 0 - 3 = -2$$
> Result: $f(B) < 0$ → **Class -1**
>
> **Step 3: Calculate Distance to Hyperplane**
> Distance for point $x$:
> $$\text{distance} = \frac{|f(x)|}{\|w\|} = \frac{|f(x)|}{\sqrt{1^2 + 2^2}} = \frac{|f(x)|}{\sqrt{5}}$$
>
> - Point A: $\frac{|1|}{\sqrt{5}} = 0.447$ units from boundary
> - Point B: $\frac{|-2|}{\sqrt{5}} = 0.894$ units from boundary
>
> **Margin** = $\frac{2}{\|w\|} = \frac{2}{\sqrt{5}} = 0.894$

---

## SVM vs Other Classifiers

| Aspect | SVM | Logistic Regression | Random Forest |
|--------|-----|---------------------|---------------|
| **Interpretability** | Low (kernel) | High | Medium |
| **Scalability** | O(n²) | O(n) | O(n log n) |
| **Non-linearity** | Via kernels | Via features | Built-in |
| **Probabilities** | Indirect | Native | Native |
| **Best for** | Small-medium, non-linear | Large, linear | Any size |

---

## Related Concepts

**Prerequisites:**
- [[stats/01_Foundations/Feature Scaling\|Feature Scaling]] — Required preprocessing
- [[stats/01_Foundations/Euclidean Distance\|Euclidean Distance]] — Used in RBF kernel

**Alternatives:**
- [[stats/03_Regression_Analysis/Logistic Regression\|Logistic Regression]] — Linear, simpler
- [[stats/04_Supervised_Learning/Random Forest\|Random Forest]] — No scaling needed
- [[stats/04_Supervised_Learning/K-Nearest Neighbors (KNN)\|K-Nearest Neighbors (KNN)]] — Instance-based

**Extensions:**
- [[SVR (Support Vector Regression)\|SVR (Support Vector Regression)]] — Regression version
- [[stats/04_Supervised_Learning/Kernel Methods\|Kernel Methods]] — General framework

---

## References

- **Historical:** Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*. [Springer Link](https://link.springer.com/article/10.1007/BF00994018)
- **Book:** Schölkopf, B., & Smola, A. J. (2002). *Learning with Kernels*. MIT Press. [MIT Press](https://mitpress.mit.edu/9780262194754/learning-with-kernels/)
- **Book:** Cristianini, N., & Shawe-Taylor, J. (2000). *An Introduction to Support Vector Machines*. Cambridge. [Cambridge Link](https://www.cambridge.org/core/books/an-introduction-to-support-vector-machines/D5D3A3B0D3B0D3B0D3B0D3B0D3B0D3B0)
