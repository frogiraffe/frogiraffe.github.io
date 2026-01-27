---
{"dg-publish":true,"permalink":"/stats/04-supervised-learning/bias-variance-tradeoff/","tags":["Machine-Learning","Statistics","Model-Selection"]}
---


## Definition

> [!abstract] Core Statement
> The **Bias-Variance Tradeoff** describes the tension between a model's ability to ==fit the training data (low bias)== and ==generalize to new data (low variance)==. Optimal models balance both sources of error.

---

> [!tip] Intuition (ELI5): The Dartboard
> **Bias** = How far from the bullseye your *average* throw lands (systematic error)
> **Variance** = How spread out your throws are (inconsistency)
> 
> A good thrower is both accurate (low bias) AND consistent (low variance).

---

## Error Decomposition

$$
\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}
$$

| Component | Source | Fixable? |
|-----------|--------|----------|
| **Bias²** | Model too simple | Yes (add complexity) |
| **Variance** | Model too complex | Yes (regularize, more data) |
| **Noise** | Data randomness | No |

---

## Visual Understanding

```
          Low Variance         High Variance
         ┌─────────────┐      ┌─────────────┐
Low      │  ●  ●       │      │ ●       ●   │
Bias     │    ⊙  ●     │      │   ⊙         │
         │  ●    ●     │      │     ●   ●   │
         └─────────────┘      └─────────────┘
             IDEAL            OVERFITTING
         ┌─────────────┐      ┌─────────────┐
High     │         ● ● │      │ ●       ●   │
Bias     │      ⊙      │      │   ⊙  ●   ●  │
         │         ●●  │      │ ●     ●     │
         └─────────────┘      └─────────────┘
         UNDERFITTING          WORST
         
⊙ = Target (True Value)  ● = Predictions
```

---

## Symptoms and Solutions

| Problem | Symptoms | Solutions |
|---------|----------|-----------|
| **High Bias** | Training error high, Test error high | More features, more complexity, less regularization |
| **High Variance** | Training error low, Test ≫ Train | More data, regularization, simpler model, dropout |

---

## Python Demonstration

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score

# ========== GENERATE DATA ==========
np.random.seed(42)
X = np.linspace(0, 1, 30).reshape(-1, 1)
y = np.sin(4 * X).ravel() + np.random.normal(0, 0.3, 30)

# ========== BIAS-VARIANCE ACROSS COMPLEXITY ==========
degrees = range(1, 15)
train_errors = []
cv_errors = []

for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    train_errors.append(1 - model.score(X_poly, y))
    cv_errors.append(-cross_val_score(model, X_poly, y, 
                                       cv=5, scoring='neg_mean_squared_error').mean())

# ========== PLOT ==========
plt.figure(figsize=(10, 6))
plt.plot(degrees, train_errors, 'b-o', label='Training Error')
plt.plot(degrees, cv_errors, 'r-o', label='CV Error')
plt.axvline(x=4, color='g', linestyle='--', label='Optimal Complexity')
plt.xlabel('Polynomial Degree (Model Complexity)')
plt.ylabel('Error')
plt.title('Bias-Variance Tradeoff')
plt.legend()
plt.annotate('High Bias\n(Underfitting)', xy=(2, 0.3), fontsize=10)
plt.annotate('High Variance\n(Overfitting)', xy=(11, 0.3), fontsize=10)
plt.show()
```

---

## How to Diagnose

| Scenario | Train Error | Test Error | Diagnosis |
|----------|-------------|------------|-----------|
| Both high | 15% | 16% | High Bias |
| Train low, test high | 2% | 15% | High Variance |
| Both low, close | 3% | 5% | Good fit! |

Use **[[stats/04_Supervised_Learning/Learning Curves\|Learning Curves]]** for visual diagnosis.

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Only Focusing on Training Accuracy**
> - *Problem:* Ignoring test performance
> - *Solution:* Always use cross-validation or held-out test set
>
> **2. Over-regularizing**
> - *Problem:* Too much L2/dropout increases bias
> - *Solution:* Tune regularization strength
>
> **3. Small Dataset + Complex Model**
> - *Problem:* Guaranteed high variance
> - *Solution:* Start simple, add complexity carefully

---

## Related Concepts

- [[stats/04_Supervised_Learning/Overfitting\|Overfitting]] — High variance manifestation
- [[stats/04_Supervised_Learning/Underfitting\|Underfitting]] — High bias manifestation
- [[stats/03_Regression_Analysis/Regularization\|Regularization]] — Variance reduction technique
- [[stats/04_Supervised_Learning/Learning Curves\|Learning Curves]] — Diagnostic tool
- [[stats/04_Supervised_Learning/Cross-Validation\|Cross-Validation]] — Variance estimation

---

## References

- **Book:** Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Chapter 7.
- **Paper:** Geman, S., Bienenstock, E., & Doursat, R. (1992). Neural networks and the bias/variance dilemma. *Neural Computation*, 4(1), 1-58.
