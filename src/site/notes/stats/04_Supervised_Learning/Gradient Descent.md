---
{"dg-publish":true,"permalink":"/stats/04-supervised-learning/gradient-descent/","tags":["Math","Calculus","Machine-Learning","Optimization"]}
---

## Definition

> [!abstract] Core Statement
> **Gradient Descent** is an iterative first-order optimization algorithm used to find a ==local minimum== of a differentiable function. It takes steps proportional to the negative of the gradient (slope) of the function at the current point.
> 
> $$ \theta_{new} = \theta_{old} - \alpha \nabla J(\theta) $$

![Gradient Descent Optimization](https://upload.wikimedia.org/wikipedia/commons/f/ff/Gradient_descent.svg)

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

## R Implementation

```r
# Simple Gradient Descent for f(x) = x^2
gradient_descent <- function(start_x, learning_rate, n_iter) {
  x <- start_x
  history <- numeric(n_iter)
  
  for(i in 1:n_iter) {
    grad <- 2 * x  # Derivative of x^2 is 2x
    x <- x - learning_rate * grad
    history[i] <- x
  }
  return(list(final_x = x, history = history))
}

# Run
res <- gradient_descent(start_x = 10, learning_rate = 0.1, n_iter = 20)
print(paste("Minimum found at:", round(res$final_x, 4)))
plot(res$history, type="b", main="Convergence Path", ylab="Value of x")
```

---

## Related Concepts

- [[stats/01_Foundations/Backpropagation\|stats/01_Foundations/Backpropagation]] - Using chain rule to calculate gradients in Neural Nets.
- [[stats/01_Foundations/Loss Function\|Loss Function]] - The function $J(\theta)$ we are minimizing.
- [[stats/04_Supervised_Learning/Neural Networks\|Neural Networks]] - Heavy users of GD.
- [[stats/01_Foundations/Feature Scaling\|Feature Scaling]] - Critical pre-requisite.

---

## References

- **Historical:** Cauchy, A. (1847). Méthode générale pour la résolution des systèmes d'équations simultanées. *Comptes Rendus*, 25, 536-538. [Link](https://gallica.bnf.fr/ark:/12148/bpt6k29824/f540.item)
- **Article:** Ruder, S. (2016). An overview of gradient descent optimization algorithms. [arXiv:1609.04747](https://arxiv.org/abs/1609.04747)
- **Book:** Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. [Book Website](https://www.deeplearningbook.org/)
