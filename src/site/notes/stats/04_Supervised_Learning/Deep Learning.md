---
{"dg-publish":true,"permalink":"/stats/04-supervised-learning/deep-learning/","tags":["probability","machine-learning","deep-learning","neural-networks"]}
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
| **[[stats/04_Supervised_Learning/Transformers\|Transformers]]** | NLP, Vision |
| **[[stats/05_Unsupervised_Learning/Autoencoders\|Autoencoders]]** | Unsupervised |

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
| **[[stats/04_Supervised_Learning/Activation Functions\|Activation Functions]]** | Non-linearity (ReLU, sigmoid) |
| **[[stats/04_Supervised_Learning/Batch Normalization\|Batch Normalization]]** | Stabilizes training |
| **[[stats/04_Supervised_Learning/Dropout\|Dropout]]** | Regularization |
| **[[stats/04_Supervised_Learning/Gradient Boosting\|Gradient Boosting]]** | Often better for tabular |

---

## Related Concepts

- [[stats/04_Supervised_Learning/Transformers\|Transformers]] — State-of-the-art
- [[stats/05_Unsupervised_Learning/Autoencoders\|Autoencoders]] — Unsupervised DL
- [[stats/04_Supervised_Learning/Activation Functions\|Activation Functions]] — Enable non-linearity

---

## References

- **Book:** Goodfellow, I., et al. (2016). *Deep Learning*. MIT Press. [Free Online](https://www.deeplearningbook.org/)
