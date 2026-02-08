---
{"dg-publish":true,"permalink":"/30-knowledge/stats/06-experimental-design/multi-armed-bandits-mab/","tags":["experimental-design"]}
---


## Definition

> [!abstract] Core Statement
> The **Multi-Armed Bandit (MAB)** problem is a classic reinforcement learning framework that models the ==Exploration-Exploitation trade-off==. An agent must decide which "arm" (action) to pull to maximize total reward over time, balancing the need to find the best arm (exploration) with the desire to pull the known best arm (exploitation).

![Visualization of Thompson Sampling](https://upload.wikimedia.org/wikipedia/commons/4/42/Visualization_of_Thompson_sampling.gif)

---

> [!tip] Intuition (ELI5): The Slot Machine Dilemma
> Imagine you are in a casino with 5 slot machines. 
> - You don't know which one pays out the most.
> - **Exploration:** You try different machines to see which one is "lucky."
> - **Exploitation:** Once you find a machine that seems to pay well, you keep playing it to win big.
> If you explore too much, you waste money on bad machines. If you exploit too early, you might miss a machine that pays even better!

---

## Why Use MAB instead of A/B Testing?

| Feature | A/B Testing | Multi-Armed Bandit |
| :--- | :--- | :--- |
| **Goal** | Minimize risk and gather evidence. | Maximize reward during the test. |
| **Traffic Assignment** | Fixed (e.g., 50/50). | Dynamic (shifts to the winner). |
| **Automation** | Requires manual stop. | Automatically converges on the winner. |
| **Use Case** | Identifying long-term impact. | Optimization (ads, headlines, prices). |

---

## Common Strategies

### 1. $\epsilon$-Greedy (Epsilon-Greedy)
- With probability $1-\epsilon$, choose the best-performing arm so far (**Exploit**).
- With probability $\epsilon$, choose an arm at random (**Explore**).
- *Pros:* Simple. *Cons:* Keeps exploring even after the winner is clear.

### 2. Upper Confidence Bound (UCB)
- Choose the arm with the highest "optimistic" estimate: $\text{Mean} + \text{Uncertainty}$.
- As an arm is pulled more, its uncertainty shrinks.
- *Pros:* Mathematically grounded. *Cons:* Harder to implement for some distributions.

### 3. Thompson Sampling (Bayesian)
- Maintain a **probability distribution** (e.g., [[30_Knowledge/Stats/01_Foundations/Beta Distribution\|Beta Distribution]]) for each arm's success rate.
- Sample a value from each distribution and pick the arm with the highest sample.
- *Pros:* Often outperforms UCB/Epsilon-Greedy in practice.

---

## Python Implementation (Conceptual)

```python
import numpy as np

# Thompson Sampling Logic
def thompson_sampling(successes, failures):
    # successes/failures arrays for each arm
    num_arms = len(successes)
    samples = [np.random.beta(successes[i] + 1, failures[i] + 1) for i in range(num_arms)]
    return np.argmax(samples)

# Simulation loop
successes = np.zeros(num_arms)
failures = np.zeros(num_arms)

for _ in range(total_steps):
    arm = thompson_sampling(successes, failures)
    reward = pull_arm(arm) # Observe 0 or 1
    if reward:
        successes[arm] += 1
    else:
        failures[arm] += 1
```

---

## Related Concepts

- [[30_Knowledge/Stats/04_Supervised_Learning/Exploration-Exploitation Trade-off\|Exploration-Exploitation Trade-off]]
- [[30_Knowledge/Stats/04_Supervised_Learning/Contextual Bandits\|Contextual Bandits]] - When actions depend on user context.
- [[30_Knowledge/Stats/04_Supervised_Learning/Thompson Sampling\|Thompson Sampling]]
- [[30_Knowledge/Stats/01_Foundations/Beta Distribution\|Beta Distribution]]

## When to Use

> [!success] Use Multi-armed Bandits (MAB) When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## R Implementation

```r
# Multi-armed Bandits (MAB) in R
set.seed(42)

# Example implementation
data <- rnorm(100)
summary(data)
```

---

## References

1. See related concepts for further reading
