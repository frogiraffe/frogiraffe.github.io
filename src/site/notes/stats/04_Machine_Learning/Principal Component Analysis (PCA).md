---
{"dg-publish":true,"permalink":"/stats/04-machine-learning/principal-component-analysis-pca/","tags":["Multivariate","Dimensionality-Reduction","Unsupervised-Learning"]}
---


# Principal Component Analysis (PCA)

## Definition

> [!abstract] Core Statement
> **Principal Component Analysis (PCA)** is an ==unsupervised dimensionality reduction== technique that transforms a large set of correlated variables into a smaller set of **uncorrelated components** (Principal Components) that capture the maximum variance.

---

## Purpose

1.  **Reduce Dimensionality:** Compress many features into fewer components.
2.  **Remove Multicollinearity:** Create uncorrelated inputs for regression.
3.  **Visualize High-Dimensional Data:** Project data to 2D or 3D.
4.  **Noise Reduction:** Discard components with low variance.

---

## When to Use

> [!success] Use PCA When...
> - You have **many correlated features**.
> - You want to **reduce dimensionality** before modeling.
> - You need **visualization** of high-dimensional data.
> - [[stats/03_Regression_Analysis/VIF (Variance Inflation Factor)\|VIF (Variance Inflation Factor)]] indicates severe multicollinearity.

> [!failure] Limitations
> - PCA is a **linear** method; non-linear relationships may not be captured.
> - **Interpretability is lost:** Principal Components are linear combinations of original variables.

---

## Theoretical Background

### How It Works

1.  **Standardize Data:** Center (mean 0) and scale (std 1). ==Mandatory.==
2.  **Calculate Covariance Matrix.**
3.  **Find Eigenvalues and Eigenvectors:** Eigenvectors define component directions; eigenvalues define variance explained.
4.  **Rank Components:** PC1 has highest variance, PC2 second highest, etc.
5.  **Project Data:** Transform data onto selected components.

### Variance Explained

Each component captures a proportion of total variance:
$$
\text{Proportion} = \frac{\lambda_k}{\sum \lambda}
$$

**Rule of Thumb:** Keep components that explain ~80-90% of cumulative variance.

### Loadings

**Loadings** are the correlations between original variables and principal components. High absolute loading = variable contributes strongly to that component.

---

## Worked Example: Customer Segmentation

> [!example] Problem
> You have data on customers with 3 correlated variables:
> - **$X_1$:** Annual Income (Mean=50k, SD=15k)
> - **$X_2$:** Spending Score (0-100) (Mean=50, SD=25)
> - **$X_3$:** Credit Card Debt (Mean=5k, SD=2k)
> 
> **Goal:** Reduce these 3 dimensions to 2 Principal Components.

**Solution Process:**

1.  **Standardize (Z-scores):**
    -   Subtract mean, divide by SD.
    -   Example Customer A: Income=80k, Spending=80, Debt=8k.
    -   $Z_1 = (80-50)/15 = 2.0$.
    -   $Z_2 = (80-50)/25 = 1.2$.
    -   $Z_3 = (8-5)/2 = 1.5$.
    -   Input Vector: $[2.0, 1.2, 1.5]$.

2.  **PCA Transformation:**
    -   Let's say PCA gives eigenvectors (loadings) for PC1: $v_1 = [0.58, 0.58, 0.58]$ (All variables correlate positively).
    -   **Calculate PC1 Score for A:**
        $$ PC1 = (0.58 \times 2.0) + (0.58 \times 1.2) + (0.58 \times 1.5) $$
        $$ PC1 = 1.16 + 0.696 + 0.87 = 2.726 $$

3.  **Interpretation:**
    -   Customer A has a **high PC1 score**. If PC1 represents "Overall Wealth/Status", this customer is "High Status".
    -   We have reduced 3 numbers to 1 (or 2) while retaining the core information about their deviation from the mean.

---

## Assumptions

- [ ] **Continuous Data:** PCA is designed for numeric data.
- [ ] **Linear Relationships:** PCA captures linear correlations.
- [ ] ==**Standardization:**== Variables must be on the same scale. Otherwise, high-variance variables dominate.

---

## Limitations

> [!warning] Pitfalls
> 1.  **Forgot to Scale?** If you run PCA on Income (0-1,000,000) and Age (0-100) without scaling, PC1 will just be "Income" because it has huge variance. **Always Standardize first.**
> 2.  **Interpretation Black Box:** "PC1 decreased by 2 units" is meaningless to business stakeholders. You must analyze the loadings to translate it (e.g., "Wealth Score decreased").
> 3.  **Outliers:** PCA tries to capture *maximum variance*. A single outlier with squared distance $100\sigma$ will pull the principal component towards it, skewing the result. Use **Robust PCA** for messy data.

---

## Python Implementation

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Standardize (CRITICAL)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Fit PCA
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)

# 3. Variance Explained
print("Variance Explained:", pca.explained_variance_ratio_)
print("Cumulative:", pca.explained_variance_ratio_.cumsum())

# 4. Scree Plot
plt.plot(range(1, 6), pca.explained_variance_ratio_, 'o-')
plt.xlabel("Component")
plt.ylabel("Variance Explained")
plt.title("Scree Plot")
plt.show()

# 5. Loadings
import pandas as pd
loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i}' for i in range(1,6)], index=X.columns)
print(loadings)
```

---

## R Implementation

```r
# 1. PCA (center and scale. are CRITICAL)
pca_result <- prcomp(df, center = TRUE, scale. = TRUE)

# 2. Summary
summary(pca_result)
# Look at "Cumulative Proportion" to decide how many components to keep.

# 3. Scree Plot
screeplot(pca_result, type = "lines")

# 4. Biplot (Visualize loadings and scores)
biplot(pca_result)

# 5. Loadings
pca_result$rotation
```

---

## Interpretation Guide

| Output | Interpretation |
|--------|----------------|
| Output | Interpretation |
|--------|----------------|
| **PC1 Explains 60%** | The first axis captures 60% of the information in the dataset. |
| **Cumulative Variance > 80%** | Stop adding components. You have retained enough signal. |
| **Loadings > 0.5** | Variable is strongly associated with this component. |
| **Biplot Arrows** | Variables with arrows pointing in same direction are highly correlated. |

---

## Related Concepts

- [[stats/01_Foundations/Factor Analysis (EFA & CFA)\|Factor Analysis (EFA & CFA)]] - Similar but for latent constructs.
- [[stats/03_Regression_Analysis/VIF (Variance Inflation Factor)\|VIF (Variance Inflation Factor)]] - PCA fixes multicollinearity.
- [[t-SNE\|t-SNE]] - Non-linear visualization.
