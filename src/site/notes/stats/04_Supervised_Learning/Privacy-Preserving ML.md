---
{"dg-publish":true,"permalink":"/stats/04-supervised-learning/privacy-preserving-ml/","tags":["Machine-Learning","Privacy","Security"]}
---


## Definition

> [!abstract] Core Statement
> **Privacy-Preserving Machine Learning** enables model training and inference ==without exposing raw data==, using cryptographic and statistical techniques.

---

## Techniques

| Technique | Privacy Level | Performance |
|-----------|---------------|-------------|
| **Differential Privacy** | Strong | Some utility loss |
| **Federated Learning** | Medium | Communication overhead |
| **Homomorphic Encryption** | Very Strong | Very slow |
| **Secure Multi-Party Computation** | Strong | High computation |

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

- [[stats/04_Supervised_Learning/Federated Learning\|Federated Learning]] — Distributed approach
- [[stats/10_Ethics_and_Biases/Differential Privacy\|Differential Privacy]] — Formal privacy guarantee

---

## References

- **Book:** Dwork, C., & Roth, A. (2014). The Algorithmic Foundations of Differential Privacy.
