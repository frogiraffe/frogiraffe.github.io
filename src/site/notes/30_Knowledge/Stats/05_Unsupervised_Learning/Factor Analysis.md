---
{"dg-publish":true,"permalink":"/30-knowledge/stats/05-unsupervised-learning/factor-analysis/","tags":["machine-learning","unsupervised"]}
---


## Definition

> [!abstract] Core Statement
> **Factor Analysis** identifies ==latent (unobserved) variables== that explain correlations among observed variables. Unlike PCA, it assumes an underlying measurement model.

---

## PCA vs Factor Analysis

| Aspect | PCA | Factor Analysis |
|--------|-----|-----------------|
| **Goal** | Reduce dimensions | Find latent factors |
| **Model** | No error term | Includes unique variance |
| **Components** | Linear combos of variables | Latent causes of variables |
| **Rotation** | Not typically | Often rotated (Varimax) |

---

## Python Implementation

```python
from sklearn.decomposition import FactorAnalysis
import pandas as pd

# Fit factor analysis
fa = FactorAnalysis(n_components=3, rotation='varimax')
fa.fit(X)

# Factor loadings
loadings = pd.DataFrame(
    fa.components_.T,
    columns=['Factor 1', 'Factor 2', 'Factor 3'],
    index=feature_names
)
print(loadings)

# Factor scores (for new data)
scores = fa.transform(X)
```

---

## R Implementation

```r
library(psych)

# Factor analysis with varimax rotation
fa_result <- fa(data, nfactors = 3, rotate = "varimax")
print(fa_result$loadings)

# Scree plot to determine number of factors
fa.parallel(data)
```

---

## Rotation Types

| Rotation | Description |
|----------|-------------|
| **Varimax** | Orthogonal, maximizes variance of loadings |
| **Promax** | Oblique, allows correlated factors |
| **Quartimax** | Minimizes factors needed to explain variables |

---

## Interpretation

- **Loading > 0.7**: Strong association
- **Loading 0.4-0.7**: Moderate
- **Loading < 0.4**: Weak

---

## Related Concepts

- [[30_Knowledge/Stats/05_Unsupervised_Learning/PCA (Principal Component Analysis)\|PCA (Principal Component Analysis)]] — Different approach
- [[30_Knowledge/Stats/01_Foundations/Cronbach's Alpha\|Cronbach's Alpha]] — Check reliability after FA

---

## When to Use

> [!success] Use Factor Analysis When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Number of clusters/components is unknown and hard to estimate
> - Data is highly sparse

---

## References

- **Book:** Hair, J. F., et al. (2019). *Multivariate Data Analysis* (8th ed.). Cengage.
