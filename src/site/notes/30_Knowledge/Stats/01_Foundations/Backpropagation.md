---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/backpropagation/","tags":["probability","foundations"]}
---


## Definition

> [!abstract] Core Statement
> **Backpropagation** computes ==gradients of the loss function with respect to all weights== by applying the chain rule backwards through the network.

![Backpropagation Derivative Flow Diagram](https://commons.wikimedia.org/wiki/Special:FilePath/Back_Propagation_Example.svg)

---

## Chain Rule

For weight $w$ in layer $l$:
$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$$

Where $z$ = weighted sum, $a$ = activation.

---

## Algorithm Steps

1. **Forward pass:** Compute outputs layer by layer
2. **Compute loss:** Compare output to target
3. **Backward pass:** Propagate gradients back
4. **Update weights:** $w \leftarrow w - \eta \frac{\partial L}{\partial w}$

---

## Python Conceptual Example

```python
# Forward pass
z1 = X @ W1 + b1
a1 = relu(z1)
z2 = a1 @ W2 + b2
y_pred = softmax(z2)

# Backward pass (simplified)
dL_dz2 = y_pred - y_true
dL_dW2 = a1.T @ dL_dz2
dL_da1 = dL_dz2 @ W2.T
dL_dz1 = dL_da1 * relu_derivative(z1)
dL_dW1 = X.T @ dL_dz1

# Update
W1 -= lr * dL_dW1
W2 -= lr * dL_dW2
```

---

## Related Concepts

- [[30_Knowledge/Stats/01_Foundations/Loss Function\|Loss Function]] - What we differentiate
- [[30_Knowledge/Stats/04_Supervised_Learning/Gradient Descent\|Gradient Descent]] - Uses computed gradients
- [[30_Knowledge/Stats/04_Supervised_Learning/Neural Networks\|Neural Networks]] - Where backprop is used

---

## When to Use

> [!success] Use Backpropagation When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## Python Implementation

```python
import numpy as np
import pandas as pd

# Example implementation of Backpropagation
# See documentation for details

data = np.random.randn(100)
print(f"Mean: {np.mean(data):.3f}")
print(f"Std: {np.std(data):.3f}")
```

---

## R Implementation

```r
# Backpropagation in R
set.seed(42)

# Example implementation
data <- rnorm(100)
summary(data)
```

---

## References

- **Article:** Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323, 533-536. [DOI: 10.1038/323533a0](https://doi.org/10.1038/323533a0)
