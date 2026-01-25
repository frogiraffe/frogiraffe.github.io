---
{"dg-publish":true,"permalink":"/stats/04-machine-learning/gradient-descent/","tags":["Math","Calculus","Machine-Learning","Optimization"]}
---

## Definition

> [!abstract] Core Statement
> **Gradient Descent** is an iterative first-order optimization algorithm used to find a ==local minimum== of a differentiable function. It takes steps proportional to the negative of the gradient (slope) of the function at the current point.
> 
> $$ \theta_{new} = \theta_{old} - \alpha \nabla J(\theta) $$

---

## Purpose

1.  **Minimize Loss:** Used to train ML models (Linear Regression, Neural Networks) by minimizing Mean Squared Error or Cross-Entropy.
2.  **Parameter Tuning:** Adjusts weights and biases until the model fits the data.

---

## Intuition: The Hiker in the Fog

Imagine you are on a mountain at night (foggy). You want to reach the lowest valley (minimum loss).
1.  **Check Slope:** You feel the ground with your foot to see which way is "down". (Calculate Gradient).
2.  **Take a Step:** You take a step in the downhill direction. (Update weights).
3.  **Step Size:** If you take tiny steps, you'll never get there. If you jump, you might fall off a cliff. (Learning Rate).
4.  **Repeat:** Keep doing this until the ground is flat (Convergence).

---

## Key Parameters

### 1. Learning Rate ($\alpha$)
The size of the step.
-   **Too Small:** Convergence takes forever.
-   **Too Large:** You overshoot the minimum and diverge (explode).

### 2. The Gradient ($\nabla J$)
The vector of partial derivatives. It points "uphill". We subtract it to go "downhill".

---

## Variants

| Variant | Description | Pros | Cons |
|---------|-------------|------|------|
| **Batch GD** | Uses **all** data for one step. | Stable convergence. | Slow; memory intensive. |
| **Stochastic GD (SGD)** | Uses **one** random sample per step. | Fast; escapes local minima. | Noisy/Jittery path. |
| **Mini-Batch GD** | Uses a batch (e.g., 32 samples). | **Best of both worlds.** | Requires tuning batch size. |

---

## Limitations & Pitfalls

> [!warning] Pitfalls
> 1.  **Local Minima:** In non-convex functions (Neural Nets), you might get stuck in a small valley, not the deepest one (Global Minimum). Motivation for momentum/Adam.
> 2.  **Saddle Points:** Points where slope is zero but it's not a minimum (flat plateau).
> 3.  **Scaling:** If features are on different scales (Age vs Income), the gradient path is a narrow ravine and descent is slow. **Always feature scale.**

---

## Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

# Comparison: Manual GD for y = x^2 (Min is at 0)

def cost_func(x):
    return x**2

def gradient(x):
    return 2*x  # Derivative of x^2

x_start = 10
learning_rate = 0.1
n_iterations = 20

x_path = [x_start]
x = x_start

for i in range(n_iterations):
    grad = gradient(x)
    x = x - learning_rate * grad
    x_path.append(x)

print(f"Final x: {x:.4f}") # Should be near 0

plt.plot(x_path, 'o-')
plt.title("Path to Minimum")
plt.xlabel("Iteration")
plt.ylabel("Value of x")
plt.show()
```

---

## Related Concepts

- [[stats/01_Foundations/Backpropagation\|Backpropagation]] - Using chain rule to calculate gradients in Neural Nets.
- [[stats/01_Foundations/Loss Function\|Loss Function]] - The function $J(\theta)$ we are minimizing.
- [[stats/04_Machine_Learning/Neural Networks\|Neural Networks]] - Heavy users of GD.
- [[stats/01_Foundations/Feature Scaling\|Feature Scaling]] - Critical pre-requisite.
