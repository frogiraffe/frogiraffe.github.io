---
{"dg-publish":true,"permalink":"/stats/10-ethics-and-biases/differential-privacy/","tags":["Ethics","Privacy","Data-Science","Cryptography"]}
---

## Definition

> [!abstract] Core Statement
> **Differential Privacy (DP)** is a mathematical framework for ensuring that the result of a data analysis does not reveal if any ==specific individual== was part of the dataset. It works by adding a controlled amount of **statistical noise** to the outputs or inputs.

---

> [!tip] Intuition (ELI5): The Secret Cookie Thief
> To find out the percentage of cookie thieves without outing anyone:
> 1. Flip a coin. If **Heads**, tell the truth.
> 2. If **Tails**, flip again: Heads means say "Yes", Tails means say "No".
> If you say "Yes," I don't know if you're a thief or just got a coin flip! But with 100 people, math can still calculate the **total** percentage accurately while keeping everyone's identity secret.

> [!example] Real-Life Example: The US Census
> The US Census Bureau adds "statistical noise" to counts in tiny towns. If a town has only one 90-year-old billionaire, an attacker shouldn't be able to identify them from the data. The noise makes individual data "jittery" while keeping the total country data accurate.

---

## Purpose

1.  **Privacy Loss Protection:** Quantifying exactly how much "privacy" is lost ($\epsilon$) when a database is queried.
2.  **Safe Data Sharing:** Allowing researchers to analyze sensitive data (e.g., medical records, census data) without violating individual privacy.
3.  **Defense against Linkage Attacks:** Preventing attackers from identifying people by combining multiple datasets.

---

## Epsilon ($\epsilon$): The Privacy Budget

| Value | Interpretation | Privacy Level |
| :--- | :--- | :--- |
| **Low $\epsilon$** (e.g., 0.01) | High noise, low accuracy. | **Strict Privacy** |
| **Medium $\epsilon$** (e.g., 1.0) | Balanced noise and accuracy. | **Standard** |
| **High $\epsilon$** (e.g., 10.0) | Low noise, high accuracy. | **Weak Privacy** |

---

## Theoretical Background: The Randomized Response

The core logic of DP often starts with the **Laplace Mechanism**:
$$ M(d) = f(d) + \text{Laplace}(\frac{\Delta f}{\epsilon}) $$
Where:
- $M(d)$ is the private output.
- $f(d)$ is the true answer.
- $\Delta f$ is the **Sensitivity** (how much the answer changes if one person is removed).

---

## Python Implementation: Laplace Mechanism

```python
import numpy as np

def private_mean(data, epsilon):
    """Calculates a differentially private mean."""
    n = len(data)
    true_sum = np.sum(data)
    
    # Sensitivity of SUM if values are between [0, 1] is 1.0
    sensitivity = 1.0
    
    # Add Laplace Noise
    noise = np.random.laplace(0, sensitivity / epsilon)
    private_sum = true_sum + noise
    
    return private_sum / n

# Dataset (e.g., salary in units of 100k)
salaries = np.array([0.5, 0.8, 1.2, 0.9, 0.6])

print(f"True Mean: {np.mean(salaries):.3f}")
print(f"DP Mean (e=0.1): {private_mean(salaries, 0.1):.3f}")
print(f"DP Mean (e=1.0): {private_mean(salaries, 1.0):.3f}")
```

---

## Related Concepts

- [[stats/10_Ethics_and_Biases/Algorithmic Bias\|Algorithmic Bias]] - DP can sometimes introduce bias if the noise affects minority groups disproportionately.
- [[stats/04_Supervised_Learning/Privacy-Preserving ML\|Privacy-Preserving ML]] - Using DP during model training (e.g., DP-SGD).
- [[stats/04_Supervised_Learning/Federated Learning\|Federated Learning]] - Often combined with DP.

---

## References

- **Book:** Dwork, C., & Roth, A. (2014). The Algorithmic Foundations of Differential Privacy. *Foundations and Trends in Theoretical Computer Science*. [Full Text PDF](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)
- **Article:** Wood, A., et al. (2018). Differential Privacy: A Primer for a Non-Technical Audience. *Vanderbilt Journal of Entertainment & Technology Law*. [JSTOR](https://www.jstor.org/stable/26529739)
- **Historical:** Dwork, C. (2006). Differential Privacy. *ICALP*. [Springer](https://link.springer.com/chapter/10.1007/11787006_1)
