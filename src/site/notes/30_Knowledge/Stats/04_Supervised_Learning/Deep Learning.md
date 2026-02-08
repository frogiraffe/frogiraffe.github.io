---
{"dg-publish":true,"permalink":"/30-knowledge/stats/04-supervised-learning/deep-learning/","tags":["machine-learning","supervised"]}
---


## Definition

> [!abstract] Core Statement
> **Deep Learning** uses ==multi-layered neural networks== to learn hierarchical representations of data, enabling automatic feature extraction and end-to-end learning.

---

## Key Architectures

| Architecture | Use Case |
|--------------|----------|
| **MLP** | Tabular data |
| **CNN** | Images |
| **RNN/LSTM** | Sequences |
| **[[30_Knowledge/Stats/04_Supervised_Learning/Transformers\|Transformers]]** | NLP, Vision |
| **[[30_Knowledge/Stats/05_Unsupervised_Learning/Autoencoders\|Autoencoders]]** | Unsupervised |

---

## Python (PyTorch)

```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleNet(10, 64, 1)
```

---

## Key Concepts

| Concept | Description |
|---------|-------------|
| **[[30_Knowledge/Stats/04_Supervised_Learning/Activation Functions\|Activation Functions]]** | Non-linearity (ReLU, sigmoid) |
| **[[30_Knowledge/Stats/04_Supervised_Learning/Batch Normalization\|Batch Normalization]]** | Stabilizes training |
| **[[30_Knowledge/Stats/04_Supervised_Learning/Dropout\|Dropout]]** | Regularization |
| **[[30_Knowledge/Stats/04_Supervised_Learning/Gradient Boosting\|Gradient Boosting]]** | Often better for tabular |

---

## Related Concepts

- [[30_Knowledge/Stats/04_Supervised_Learning/Transformers\|Transformers]] — State-of-the-art
- [[30_Knowledge/Stats/05_Unsupervised_Learning/Autoencoders\|Autoencoders]] — Unsupervised DL
- [[30_Knowledge/Stats/04_Supervised_Learning/Activation Functions\|Activation Functions]] — Enable non-linearity

---

## When to Use

> [!success] Use Deep Learning When...
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

# Example implementation of Deep Learning
# See documentation for details

data = np.random.randn(100)
print(f"Mean: {np.mean(data):.3f}")
print(f"Std: {np.std(data):.3f}")
```

---

## R Implementation

```r
# Deep Learning in R
set.seed(42)

# Example implementation
data <- rnorm(100)
summary(data)
```

---

## References

- **Book:** Goodfellow, I., et al. (2016). *Deep Learning*. MIT Press. [Free Online](https://www.deeplearningbook.org/)
