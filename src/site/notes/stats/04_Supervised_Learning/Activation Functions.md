---
{"dg-publish":true,"permalink":"/stats/04-supervised-learning/activation-functions/","tags":["probability","deep-learning","neural-networks"]}
---


## Definition

> [!abstract] Core Statement
> **Activation Functions** introduce ==non-linearity== into neural networks, allowing them to learn complex patterns. Without activation functions, a deep network would be equivalent to a single linear transformation.

---

> [!tip] Intuition (ELI5): The Decision Maker
> Each neuron receives inputs and decides how much signal to pass on. The activation function is like a volume knob — ReLU says "pass positive signals, mute negative," while Sigmoid says "squash everything to 0-1."

---

## Common Activation Functions

| Function | Formula | Range | Use Case |
|----------|---------|-------|----------|
| **ReLU** | $\max(0, x)$ | $[0, \infty)$ | Hidden layers (default) |
| **Leaky ReLU** | $\max(0.01x, x)$ | $(-\infty, \infty)$ | Hidden layers |
| **Sigmoid** | $\frac{1}{1+e^{-x}}$ | $(0, 1)$ | Binary output |
| **Tanh** | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $(-1, 1)$ | Hidden layers (older) |
| **Softmax** | $\frac{e^{x_i}}{\sum e^{x_j}}$ | $(0, 1)$, sum=1 | Multi-class output |
| **GELU** | $x \cdot \Phi(x)$ | $(-\infty, \infty)$ | Transformers |
| **Swish** | $x \cdot \sigma(x)$ | $(-\infty, \infty)$ | Modern architectures |

---

## Visual Comparison

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 1000)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# ReLU
axes[0,0].plot(x, np.maximum(0, x), 'b-', linewidth=2)
axes[0,0].set_title('ReLU')
axes[0,0].axhline(0, color='gray', linestyle='--', alpha=0.5)
axes[0,0].axvline(0, color='gray', linestyle='--', alpha=0.5)

# Leaky ReLU
axes[0,1].plot(x, np.where(x > 0, x, 0.1 * x), 'b-', linewidth=2)
axes[0,1].set_title('Leaky ReLU (α=0.1)')

# Sigmoid
axes[0,2].plot(x, 1 / (1 + np.exp(-x)), 'b-', linewidth=2)
axes[0,2].set_title('Sigmoid')

# Tanh
axes[1,0].plot(x, np.tanh(x), 'b-', linewidth=2)
axes[1,0].set_title('Tanh')

# Softplus
axes[1,1].plot(x, np.log1p(np.exp(x)), 'b-', linewidth=2)
axes[1,1].set_title('Softplus')

# GELU (approximation)
from scipy.stats import norm
axes[1,2].plot(x, x * norm.cdf(x), 'b-', linewidth=2)
axes[1,2].set_title('GELU')

plt.tight_layout()
plt.show()
```

---

## Choosing Activation Functions

| Layer | Recommended | Reason |
|-------|-------------|--------|
| **Hidden (general)** | ReLU | Fast, works well |
| **Hidden (dying ReLU problem)** | Leaky ReLU, ELU | Non-zero gradient for negatives |
| **Hidden (transformers)** | GELU | State-of-the-art |
| **Output (binary)** | Sigmoid | Probability 0-1 |
| **Output (multi-class)** | Softmax | Probabilities sum to 1 |
| **Output (regression)** | Linear (none) | Unconstrained values |

---

## Python Implementation

```python
import torch
import torch.nn as nn
import tensorflow as tf

# ========== PYTORCH ==========
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.leaky = nn.LeakyReLU(0.1)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.gelu(self.fc2(x))
        x = self.fc3(x)  # No activation for output (use with CrossEntropyLoss)
        return x

# ========== TENSORFLOW/KERAS ==========
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(100,)),
    tf.keras.layers.Dense(32, activation='gelu'),
    tf.keras.layers.Dense(10, activation='softmax')  # Multi-class
])

# Available activations
# 'relu', 'leaky_relu', 'elu', 'selu', 'gelu', 'swish'
# 'sigmoid', 'tanh', 'softmax', 'linear'
```

---

## The Dying ReLU Problem

> [!warning] ReLU Danger
> If a ReLU neuron receives negative inputs, it outputs 0 and gets 0 gradient → never updates!
>
> **Symptoms:** Many neurons output exactly 0
>
> **Solutions:**
> - Use Leaky ReLU / ELU
> - Lower learning rate
> - Better weight initialization

---

## Vanishing Gradient Problem

| Function | Problem | Severity |
|----------|---------|----------|
| **Sigmoid** | Gradients → 0 for large $|x|$ | Severe |
| **Tanh** | Same issue, less severe | Moderate |
| **ReLU** | No vanishing for positive x | None (but dying ReLU) |

This is why **ReLU family dominates modern deep learning**.

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Sigmoid in Hidden Layers**
> - *Problem:* Vanishing gradients, slow training
> - *Solution:* Use ReLU or variants
>
> **2. ReLU for Output Layer**
> - *Problem:* Can't output negative values
> - *Solution:* Use linear (regression) or sigmoid/softmax (classification)
>
> **3. Wrong Output Activation**
> - Binary classification → Sigmoid (with BCELoss)
> - Multi-class → Softmax (with CrossEntropyLoss)

---

## Related Concepts

- [[stats/04_Supervised_Learning/Dropout\|Dropout]] — Regularization for neural networks
- [[stats/04_Supervised_Learning/Batch Normalization\|Batch Normalization]] — Often used with activations
- [[stats/04_Supervised_Learning/Gradient Descent\|Gradient Descent]] — Activation affects gradients

---

## References

- **Paper:** Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. *AISTATS*.
- **Paper:** Hendrycks, D., & Gimpel, K. (2016). Gaussian Error Linear Units (GELUs). *arXiv*.
- **Tutorial:** [PyTorch Activation Functions](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)
