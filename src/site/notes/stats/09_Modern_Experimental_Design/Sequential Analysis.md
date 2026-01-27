---
{"dg-publish":true,"permalink":"/stats/09-modern-experimental-design/sequential-analysis/","tags":["Experimental-Design","Hypothesis-Testing","Statistics"]}
---


## Definition

> [!abstract] Core Statement
> **Sequential Analysis** is a method of statistical inference where the sample size is not fixed in advance. Instead, data are evaluated as they are collected, and the experiment stops as soon as a ==pre-defined significance level== is reached. It is most commonly implemented via the **Sequential Probability Ratio Test (SPRT)**.

![Decision Diagram for the Ratio Test](https://upload.wikimedia.org/wikipedia/commons/4/4a/Decision_diagram_for_the_ratio_test.svg)

---

> [!tip] Intuition (ELI5): The Soup Taster
> Imagine you are tasting a soup to see if it's too salty ($H_1$) or just right ($H_0$). 
> - **Fixed Sample:** You decide to eat exactly 10 spoons before making a decision.
> - **Sequential Analysis:** If the first spoon is incredibly salty, you stop immediately and send it back. You didn't need 10 spoons to know it's bad!
> - This saves time and "soup" (traffic/money).

---

## Why Use It?

- **Efficiency:** On average, it requires 50% fewer samples than fixed tests to reach the same power.
- **Ethics:** In clinical trials, you stop as soon as a drug is proven to be dangerous or highly effective to save lives.
- **Speed:** In tech, it allows for faster product iterations.

---

## Sequential Probability Ratio Test (SPRT)

Developed by Abraham Wald, the test calculates a likelihood ratio $LR$ at each step:

$$ LR = \frac{P(\text{Data} | H_1)}{P(\text{Data} | H_0)} $$

### The Decision Rule
- **Upper Boundary ($B$):** If $LR \geq \frac{1-\beta}{\alpha}$, reject $H_0$ (Success!).
- **Lower Boundary ($A$):** If $LR \leq \frac{\beta}{1-\alpha}$, accept $H_0$ (Fail/Neutral).
- **Middle:** If $A < LR < B$, continue collecting data.

---

## The "Peeking" Problem

> [!warning] Caution: Continuous Peeking
> In standard A/B testing (like a t-test), "peeking" at the data and stopping when $p < 0.05$ is illegal! It drastically increases the **Type I Error Rate**. 
> Sequential Analysis uses specific mathematical boundaries to account for this peeking, making it statistically valid to stop early.

---

## Python Example (Logic)

```python
import numpy as np

alpha = 0.05
beta = 0.20
h0_prob = 0.5
h1_prob = 0.6 # Expecting 10% lift

# Boundaries
A = beta / (1 - alpha)
B = (1 - beta) / alpha

log_lr = 0
for data_point in incoming_stream:
    # Update log-likelihood ratio
    if data_point == 1:
        log_lr += np.log(h1_prob / h0_prob)
    else:
        log_lr += np.log((1 - h1_prob) / (1 - h0_prob))
    
    # Check boundaries
    if np.exp(log_lr) >= B:
        print("Stop! Reject H0")
        break
    elif np.exp(log_lr) <= A:
        print("Stop! Accept H0")
        break
```

---

## Related Concepts

- [[stats/02_Hypothesis_Testing/A-B Testing\|A-B Testing]]
- [[stats/02_Hypothesis_Testing/Type I & Type II Errors\|Type I & Type II Errors]]
- [[stats/02_Hypothesis_Testing/Power Analysis\|Power Analysis]]
- [[Bayesian AB Testing\|Bayesian AB Testing]]
