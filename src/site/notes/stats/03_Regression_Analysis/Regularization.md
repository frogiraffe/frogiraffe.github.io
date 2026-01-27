---
{"dg-publish":true,"permalink":"/stats/03-regression-analysis/regularization/","tags":["Machine-Learning","Model-Validation","Overfitting"]}
---

## Definition

> [!abstract] Core Statement
> **Regularization** is a technique used to **prevent overfitting** by adding a **penalty term** to the model's loss function. This penalty discourages complex models (large coefficients), biasedly "shrinking" estimates towards zero to reduce Variance.

$$ \text{Loss} = \text{Data Fit Error} + \lambda \times \text{Complexity Penalty} $$

---

## Purpose

1.  **Bias-Variance Trade-off:** Intentionally introduce a small amount of **Bias** to achieve a large reduction in **Variance**.
2.  **Generalization:** Helps the model perform better on unseen data.
3.  **Ill-Posed Problems:** Solves problems where there are more features than observations ($p > n$).

---

## Key Methods

| Method | Penalty | Effect | Usage |
|--------|---------|--------|-------|
| **[[stats/03_Regression_Analysis/Ridge Regression\|Ridge Regression]]** | L2 ($\sum \beta^2$) | Shrinks all coeffs; none to zero. | Multicollinearity, Dense data. |
| **[[stats/03_Regression_Analysis/Lasso Regression\|Lasso Regression]]** | L1 ($\sum |\beta|$) | Shrinks some to **exactly zero**. | Feature Selection, Sparse data. |
| **[[stats/01_Foundations/Elastic Net\|Elastic Net]]** | L1 + L2 | Best of both worlds. | Correlated features, Feature selection. |

---

## Conceptual Example: Polynomial Fitting

> [!example] Fitting a Line to Noisy Data
> Data: 10 points that roughly follow a line, but with noise.
> 
> 1.  **Linear Model:** Underfits slightly.
> 2.  **10th Degree Polynomial:** Hits every single point perfectly. $R^2 = 1.0$.
>     -   *Problem:* The curve goes wild between points. Huge variance.
>     -   Coefficients: $\beta_{10} = 5,000,000$.
> 
> 3.  **Regularized Polynomial:** Fits the curve, but penalty prevents $\beta = 5,000,000$.
>     -   Coefficients kept small. Curve is smooth.
>     -   Result: Good fit ($R^2=0.9$) and stable predictions.

---

## When to Use

> [!success] Always Consider Regularization When...
> - Model is **Overfitting** (Train score >> Test score).
> - Sample size is small relative to number of features.
> - Collinearity is high.
> - You want a robust deployment model.

---

## Related Concepts

- [[stats/01_Foundations/Bias-Variance Trade-off\|Bias-Variance Trade-off]]
- [[stats/04_Machine_Learning/Overfitting\|Overfitting]]
- [[stats/04_Machine_Learning/Cross-Validation\|Cross-Validation]] - Essential for choosing $\lambda$ (strength of penalty).

---

## References

- **Book:** Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. [Springer Link](https://link.springer.com/book/10.1007/978-0-387-84858-7)
- **Book:** James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.). Springer. [Springer Link](https://link.springer.com/book/10.1007/978-1-0716-1418-1)
- **Historical:** Tikhonov, A. N. (1963). On the solution of ill-posed problems and the method of regularization. *Soviet Mathematics*, 4, 1035-1038. [MathNet Link](http://mi.mathnet.ru/dan28329)
