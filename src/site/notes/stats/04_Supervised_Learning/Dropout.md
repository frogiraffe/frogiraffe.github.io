---
{"dg-publish":true,"permalink":"/stats/04-supervised-learning/dropout/","tags":["probability","deep-learning","regularization","neural-networks"]}
---


## Definition

> [!abstract] Core Statement
> **Dropout** is a regularization technique that ==randomly ignores neurons== during training, preventing co-adaptation and reducing overfitting. At each training step, each neuron has probability $p$ of being "dropped out."

---

> [!tip] Intuition (ELI5): The Team Project
> Imagine a team where some members skip random meetings. No one can rely too heavily on any single person — everyone must learn to contribute independently. The team becomes more robust.

---

## How It Works

1. **During Training:** Randomly set activations to 0 with probability $p$
2. **Remaining neurons:** Scaled by $1/(1-p)$ to maintain expected values
3. **During Inference:** Use all neurons (no dropout)

---

## Mathematical View

Training: For hidden unit $h_i$ with activation $a_i$:
$$
\tilde{a}_i = \frac{r_i \cdot a_i}{1-p} \quad \text{where } r_i \sim \text{Bernoulli}(1-p)
$$

**Interpretation:** Dropout trains an ensemble of $2^n$ thinned networks (where $n$ = number of units).

---

## Python Implementation

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ========== KERAS MODEL ==========
model = Sequential([
    Dense(256, activation='relu', input_shape=(100,)),
    Dropout(0.5),  # 50% dropout rate
    Dense(128, activation='relu'),
    Dropout(0.3),  # 30% dropout rate
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ========== PYTORCH IMPLEMENTATION ==========
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(100, 256)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Note: model.train() enables dropout, model.eval() disables it
```

---

## Dropout Rates by Layer Type

| Layer Type | Typical Dropout Rate |
|------------|---------------------|
| Input layer | 0.1-0.2 (or none) |
| Hidden layers | 0.2-0.5 |
| Before output | 0.0-0.2 |
| CNN (Spatial) | 0.25 (or use SpatialDropout) |
| RNN | 0.2-0.5 (recurrent dropout) |

---

## Variants

| Variant | Description | Use Case |
|---------|-------------|----------|
| **Standard Dropout** | Drop individual neurons | Fully connected layers |
| **Spatial Dropout** | Drop entire feature maps | CNNs |
| **Recurrent Dropout** | Drop connections between time steps | RNNs/LSTMs |
| **DropConnect** | Drop weights instead of activations | Alternative approach |
| **Alpha Dropout** | For SELU activation | Self-normalizing networks |

---

## When to Use Dropout

> [!success] Use Dropout When...
> - Network is **overfitting** (large gap between train/val)
> - Building **deep networks** with many parameters
> - **Small dataset** relative to model capacity

> [!failure] Avoid/Reduce Dropout When...
> - Network is **underfitting**
> - Using **Batch Normalization** heavily (often redundant)
> - Very **small networks** (simplify instead)

---

## Dropout vs Other Regularization

| Method | Mechanism | When to Prefer |
|--------|-----------|----------------|
| **Dropout** | Random neuron masking | Deep networks, FC layers |
| **L2 Regularization** | Weight penalty | Any model, continuous |
| **Batch Normalization** | Normalize activations | Modern CNNs (often replaces dropout) |
| **Early Stopping** | Stop before overfit | Any model, simple |
| **Data Augmentation** | More training data | Images, NLP |

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Forgetting model.eval() in PyTorch**
> - *Problem:* Dropout active during inference → wrong predictions
> - *Solution:* Always call `model.eval()` for inference
>
> **2. Too Much Dropout**
> - *Problem:* Underfitting, slow convergence
> - *Solution:* Start with 0.2, increase gradually
>
> **3. Dropout + Batch Norm**
> - *Problem:* Can interfere with each other
> - *Solution:* Use one or the other, or apply dropout before BN

---

## Related Concepts

- [[stats/03_Regression_Analysis/Regularization\|Regularization]] — General concept
- [[stats/04_Supervised_Learning/Overfitting\|Overfitting]] — What dropout prevents
- [[stats/04_Supervised_Learning/Batch Normalization\|Batch Normalization]] — Alternative/complement
- [[stats/04_Supervised_Learning/Learning Curves\|Learning Curves]] — Diagnose if needed

---

## References

- **Paper:** Srivastava, N., et al. (2014). Dropout: A simple way to prevent neural networks from overfitting. *JMLR*, 15(1), 1929-1958. [PDF](https://jmlr.org/papers/v15/srivastava14a.html)
- **Paper:** Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation. *ICML*.
