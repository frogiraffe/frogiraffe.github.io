---
{"dg-publish":true,"permalink":"/stats/01-foundations/structural-equation-modeling-sem/","tags":["Multivariate","Causal-Inference","Latent-Variables"]}
---


## Definition

> [!abstract] Core Statement
> **Structural Equation Modeling (SEM)** is a multivariate technique combining ==factor analysis and path analysis== to test hypothesized relationships among observed and latent variables.

---

## Components

| Component | Description |
|-----------|-------------|
| **Measurement model** | Latent variable ↔ indicators (factor analysis) |
| **Structural model** | Relationships between latent variables (path analysis) |

---

## Key Terms

- **Latent variable:** Unobserved construct (e.g., "motivation")
- **Indicator:** Observed variable measuring latent construct
- **Path coefficient:** Regression weight between variables
- **Error variance:** Unexplained portion

---

## Fit Indices

| Index | Good Fit |
|-------|----------|
| **CFI** | > 0.95 |
| **TLI** | > 0.95 |
| **RMSEA** | < 0.06 |
| **SRMR** | < 0.08 |

---

## R Implementation

```r
library(lavaan)

model <- '
  # Measurement model
  motivation =~ item1 + item2 + item3
  performance =~ perf1 + perf2 + perf3
  
  # Structural model
  performance ~ motivation + age
'

fit <- sem(model, data = df)
summary(fit, fit.measures = TRUE, standardized = TRUE)
```

---

## Python Implementation

```python
import semopy

model = '''
motivation =~ item1 + item2 + item3
performance =~ perf1 + perf2 + perf3
performance ~ motivation + age
'''

sem_model = semopy.Model(model)
sem_model.fit(data)
print(sem_model.inspect())
```

---

## Related Concepts

- [[stats/04_Machine_Learning/Principal Component Analysis (PCA)\|Principal Component Analysis (PCA)]] - Simpler dimension reduction
- Path Analysis - SEM without latent variables
- Confirmatory Factor Analysis - Measurement model only

---

## References

- **Book:** Kline, R. B. (2016). *Principles and Practice of Structural Equation Modeling* (4th ed.). Guilford. [Guilford Press](https://www.guilford.com/books/Principles-and-Practice-of-Structural-Equation-Modeling/Rex-Kline/9781462551910)
