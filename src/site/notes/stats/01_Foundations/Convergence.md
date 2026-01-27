---
{"dg-publish":true,"permalink":"/stats/01-foundations/convergence/","tags":["Machine-Learning","Optimization","Theory"]}
---


## Definition

> [!abstract] Core Statement
> **Convergence** in optimization occurs when the ==algorithm approaches a solution== where further iterations produce negligible improvement.

---

## Stopping Criteria

| Criterion | Condition |
|-----------|-----------|
| **Gradient norm** | $\|\nabla L\| < \epsilon$ |
| **Parameter change** | $\|\theta_{t+1} - \theta_t\| < \epsilon$ |
| **Loss change** | $\|L_{t+1} - L_t\| < \epsilon$ |
| **Maximum iterations** | $t > t_{max}$ |

---

## Convergence Rate

| Rate | Definition | Example |
|------|------------|---------|
| **Linear** | $\|e_{t+1}\| \leq c \|e_t\|$ | Gradient descent |
| **Superlinear** | $\frac{\|e_{t+1}\|}{\|e_t\|} \to 0$ | BFGS |
| **Quadratic** | $\|e_{t+1}\| \leq c \|e_t\|^2$ | Newton's method |

---

## Non-Convergence Signs

- Loss oscillating or increasing
- Gradients exploding/vanishing
- Parameters stuck at boundary

---

## Python Example

```python
def gradient_descent(f, grad, x0, tol=1e-6, max_iter=1000):
    x = x0
    for i in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            print(f"Converged in {i} iterations")
            break
        x = x - 0.01 * g
    return x
```

---

## Related Concepts

- [[stats/01_Foundations/Optimization\|Optimization]] - Where convergence applies
- [[stats/04_Machine_Learning/Gradient Descent\|Gradient Descent]] - Common algorithm
- [[stats/01_Foundations/Loss Function\|Loss Function]] - What we're minimizing

---

## References

- **Book:** Nocedal, J., & Wright, S. (2006). *Numerical Optimization*. Springer. [Springer Link](https://link.springer.com/book/10.1007/978-0-387-40065-5)
