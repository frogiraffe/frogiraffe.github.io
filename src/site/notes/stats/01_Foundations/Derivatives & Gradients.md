---
{"dg-publish":true,"permalink":"/stats/01-foundations/derivatives-and-gradients/","tags":["Math","Calculus","Optimization"]}
---

## Definition

> [!abstract] Core Statement
> A **Derivative** measures the ==instantaneous rate of change== of a function with respect to one of its variables. It is the slope of the tangent line to the graph at a point. 
> A **Gradient** ($\nabla$) is a vector containing the derivatives for **all** variables in a multivariable function.

---

## Purpose

1.  **Optimization:** Finding where the slope is zero (Minima/Maxima) is the key to training AI.
2.  **Sensitivity Analysis:** "If I increase Price by \$1, how much does Demand change?"
3.  **Backpropagation:** How neural networks learn (propagating error backwards via the Chain Rule).

---

## Intuition

-   **Slope = 0:** You are at a peak (maximum) or valley (minimum).
-   **Slope > 0:** Value is increasing. To go down, move left.
-   **Slope < 0:** Value is decreasing. To go down, move right.

---

## Types

### 1. Ordinary Derivative $\frac{d f}{d x}$
Function has only one input variable ($y = f(x)$).
-   Example: $f(x) = x^2 \to f'(x) = 2x$.

### 2. Partial Derivative $\frac{\partial f}{\partial x}$
Function has multiple inputs ($z = f(x, y)$). We ask: "How does $z$ change if I wiggle $x$, **holding $y$ constant**?"
-   Example: $f(x, y) = x^2 + y^2$.
-   $\frac{\partial f}{\partial x} = 2x$ (Treat $y$ as a constant number like 5).
-   $\frac{\partial f}{\partial y} = 2y$.

### 3. The Gradient $\nabla f$
The vector of all partials: $\nabla f = [2x, 2y]$.
-   **Direction:** Points in the direction of **steepest ascent**.
-   **Magnitude:** How steep the slope is.

---

## Worked Example: Minimizing Cost

> [!example] Problem
> Cost Function: $J(w) = w^2 - 4w + 5$.
> Goal: Find $w$ that minimizes Cost.

1.  **Derivative:** $J'(w) = 2w - 4$.
2.  **Set to Zero:**
    $$ 2w - 4 = 0 \implies 2w = 4 \implies w = 2 $$
3.  **Conclusion:** The minimum cost occurs at $w=2$.
    (Check: $2^2 - 4(2) + 5 = 4 - 8 + 5 = 1$. Any other $w$ gives $>1$).

---

## Assumptions

- [ ] **Differentiability:** The function must be smooth (no sharp corners/kinks). E.g., ReLU activation has a "kink" at 0 which requires sub-gradients.
- [ ] **Continuity:** No gaps/jumps in the function.

---

## Limitations & Pitfalls

> [!warning] Pitfalls
> 1.  **Local vs Global:** Setting derivative to 0 finds *all* flat points (minima, maxima, saddle points). It doesn't guarantee the *best* one.
> 2.  **Vanishing Gradients:** In deep networks, if many derivatives < 1 are multiplied (Chain Rule), the product approaches zero. The network stops learning.
> 3.  **Exploding Gradients:** Conversely, if derivatives > 1, the product grows exponentially. Steps become huge and unstable.

---

## Python Implementation

```python
import sympy as sp

# Symbolic Math
w = sp.Symbol('w')
J = w**2 - 4*w + 5

# Calculate Derivative
derivative = sp.diff(J, w)
print(f"Derivative: {derivative}")

# Solve for 0
roots = sp.solve(derivative, w)
print(f"Critical Point at w = {roots[0]}")
```

---

## R Implementation

```r
# Symbolic Differentiation
f_sym <- expression(x^2 + 3*x + 5)
deriv_result <- D(f_sym, "x")

# Evaluate at x = 2
x <- 2
eval(deriv_result)

# Numerical Differentiation using numDeriv
# install.packages("numDeriv")
library(numDeriv)

f <- function(x) x^2 + 3*x + 5
grad(f, x=2)
```

---

## Related Concepts

- [[stats/04_Supervised_Learning/Gradient Descent\|Gradient Descent]] - Using derivates iteratively.
- [[stats/01_Foundations/Optimization\|Optimization]] - The broader field.
- [[stats/04_Supervised_Learning/Neural Networks\|Neural Networks]] - Use partial derivatives (weights).
- [[stats/08_Time_Series_Analysis/Taylor Series\|Taylor Series]] - Approximating functions using derivatives.

---

## References

- **Book:** Stewart, J. (2015). *Calculus: Early Transcendentals* (8th ed.). Cengage Learning. [Cengage Link](https://www.cengage.com/c/calculus-early-transcendentals-8e-stewart/9781285741550/)
- **Book:** Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. [Book Website](https://www.deeplearningbook.org/) (Chapter 6)
- **Book:** Strang, G. (2019). *Linear Algebra and Learning from Data*. Wellesley-Cambridge Press. [Wellesley-Cambridge](https://math.mit.edu/~gs/learningfromdata/)
