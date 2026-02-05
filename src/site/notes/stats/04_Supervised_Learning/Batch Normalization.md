---
{"dg-publish":true,"permalink":"/stats/04-supervised-learning/batch-normalization/","tags":["probability","deep-learning","neural-networks","normalization"]}
---


## Definition

> [!abstract] Core Statement
> **Batch Normalization** normalizes layer inputs by ==standardizing activations across the mini-batch==, then applying learnable scale ($\gamma$) and shift ($\beta$) parameters. It accelerates training and acts as regularization.

---

> [!tip] Intuition (ELI5): The Class Curve
> Imagine each layer's outputs are exam scores. Some tests are harder (different means/variances). Batch Norm "curves" every test to have mean=0, variance=1, so the next layer always sees consistent input.

---

## How It Works

For each feature in a mini-batch:

1. **Compute batch statistics:**
$$
\mu_B = \frac{1}{m}\sum_{i=1}^m x_i \quad \sigma_B^2 = \frac{1}{m}\sum_{i=1}^m (x_i - \mu_B)^2
$$

2. **Normalize:**
$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$

3. **Scale and shift (learnable):**
$$
y_i = \gamma \hat{x}_i + \beta
$$

---

## Benefits

| Benefit | Explanation |
|---------|-------------|
| **Faster training** | Allows higher learning rates |
| **Regularization** | Adds noise via batch statistics |
| **Reduces internal covariate shift** | Stable input distributions |
| **Less sensitive to initialization** | More forgiving weight init |

---

## Python Implementation

```python
import torch
import torch.nn as nn
import tensorflow as tf

# ========== PYTORCH ==========
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)  # For 2D feature maps
        
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.bn2 = nn.BatchNorm1d(256)  # For fully connected
        
        self.fc2 = nn.Linear(256, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn2(x)
        x = torch.relu(x)
        
        x = self.fc2(x)
        return x

# ========== TENSORFLOW/KERAS ==========
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, 3, padding='same', input_shape=(32, 32, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    
    tf.keras.layers.Dense(10, activation='softmax')
])
```

---

## Placement: Before or After Activation?

| Order | Argument |
|-------|----------|
| **Conv → BN → ReLU** | Original paper, most common |
| **Conv → ReLU → BN** | Some argue works equally well |

**Practical advice:** Both work. Be consistent within your architecture.

---

## Training vs Inference

| Mode | Behavior |
|------|----------|
| **Training** | Uses batch statistics ($\mu_B$, $\sigma_B$) |
| **Inference** | Uses running averages (population statistics) |

```python
# PyTorch
model.train()  # Use batch statistics
model.eval()   # Use running averages

# Keras automatically handles this
```

---

## Variants

| Variant | Normalizes Over | Use Case |
|---------|-----------------|----------|
| **Batch Norm** | Batch dimension | CNNs, most common |
| **Layer Norm** | Feature dimension | RNNs, Transformers |
| **Instance Norm** | Single sample | Style transfer |
| **Group Norm** | Groups of channels | Small batch sizes |

---

## BatchNorm vs Dropout

| Aspect | BatchNorm | Dropout |
|--------|-----------|---------|
| **Regularization** | Mild | Strong |
| **Training speed** | Much faster | No change |
| **Together?** | Often redundant | Can conflict |

> [!tip] Modern Practice
> In CNNs, **BatchNorm alone** often works better than BatchNorm + Dropout.
> In Transformers, **LayerNorm** is preferred.

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Batch Size Too Small**
> - *Problem:* Noisy batch statistics → unstable training
> - *Solution:* Use GroupNorm or LayerNorm for batch_size < 16
>
> **2. Forgetting model.eval()**
> - *Problem:* Using batch statistics at inference
> - *Solution:* Always call `.eval()` or `.train(False)` before inference
>
> **3. Before vs After Activation Debate**
> - *Reality:* Both work, just be consistent

---

## Related Concepts

- [[stats/04_Supervised_Learning/Dropout\|Dropout]] — Alternative/complementary regularization
- [[stats/04_Supervised_Learning/Activation Functions\|Activation Functions]] — Usually applied after BN
- [[stats/03_Regression_Analysis/Regularization\|Regularization]] — BN provides implicit regularization

---

## References

- **Paper:** Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. *ICML*. [arXiv](https://arxiv.org/abs/1502.03167)
- **Paper:** Wu, Y., & He, K. (2018). Group Normalization. *ECCV*.
