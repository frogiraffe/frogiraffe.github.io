---
{"dg-publish":true,"permalink":"/30-knowledge/stats/04-supervised-learning/privacy-preserving-ml/","tags":["machine-learning","supervised"]}
---


## Definition

> [!abstract] Core Statement
> **Privacy-Preserving Machine Learning** enables model training and inference ==without exposing raw data==, using cryptographic and statistical techniques.

---

## Techniques

| Technique                          | Privacy Level | Performance            |
| ---------------------------------- | ------------- | ---------------------- |
| **Differential Privacy**           | Strong        | Some utility loss      |
| **Federated Learning**             | Medium        | Communication overhead |
| **Homomorphic Encryption**         | Very Strong   | Very slow              |
| **Secure Multi-Party Computation** | Strong        | High computation       |

---

## Differential Privacy

Add calibrated noise to protect individuals:

```python
import numpy as np

def dp_mean(data, epsilon=1.0, sensitivity=1.0):
    """Differentially private mean"""
    true_mean = np.mean(data)
    noise_scale = sensitivity / epsilon
    noisy_mean = true_mean + np.random.laplace(0, noise_scale)
    return noisy_mean

# Smaller epsilon = more privacy, more noise
print(dp_mean(data, epsilon=0.1))  # High privacy
print(dp_mean(data, epsilon=10))   # Low privacy
```

---

## Related Concepts

- [[30_Knowledge/Stats/04_Supervised_Learning/Federated Learning\|Federated Learning]] — Distributed approach
- [[30_Knowledge/Stats/10_Ethics_and_Biases/Differential Privacy\|Differential Privacy]] — Formal privacy guarantee

---

## When to Use

> [!success] Use Privacy-Preserving ML When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Dataset is too small for training
> - Interpretability is more important than accuracy

---

## Python Implementation

```python
import numpy as np
import pandas as pd

# Example implementation of Privacy-Preserving ML
# See documentation for details

data = np.random.randn(100)
print(f"Mean: {np.mean(data):.3f}")
print(f"Std: {np.std(data):.3f}")
```

---

## R Implementation

```r
# Privacy-Preserving ML in R
set.seed(42)

# Example implementation
data <- rnorm(100)
summary(data)
```

---

## References

- **Book:** Dwork, C., & Roth, A. (2014). The Algorithmic Foundations of Differential Privacy.
