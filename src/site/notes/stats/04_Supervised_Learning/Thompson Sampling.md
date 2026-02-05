---
{"dg-publish":true,"permalink":"/stats/04-supervised-learning/thompson-sampling/","tags":["probability","bandits","bayesian","reinforcement-learning"]}
---


## Definition

> [!abstract] Core Statement
> **Thompson Sampling** is a Bayesian algorithm for the ==multi-armed bandit problem== that balances exploration and exploitation by sampling from posterior distributions of each arm's reward and choosing the arm with the highest sample.

---

> [!tip] Intuition (ELI5): The Optimistic Gambler
> Imagine choosing between slot machines. For each machine, you imagine what its payout COULD be (based on your experience + uncertainty). You play the one that looks best in your imagination. Uncertain machines get more chances.

---

## How It Works

1. **Maintain** a posterior distribution for each arm's reward probability
2. **Sample** one value from each arm's posterior
3. **Select** the arm with the highest sampled value
4. **Update** the posterior based on the observed reward
5. **Repeat**

---

## Python Implementation

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class ThompsonSamplingBernoulli:
    """Thompson Sampling for Bernoulli (click/no-click) rewards."""
    
    def __init__(self, n_arms):
        self.n_arms = n_arms
        # Beta prior: uniform (alpha=1, beta=1)
        self.alpha = np.ones(n_arms)  # Successes + 1
        self.beta = np.ones(n_arms)   # Failures + 1
        
    def select_arm(self):
        # Sample from each arm's posterior
        samples = np.array([
            np.random.beta(self.alpha[i], self.beta[i])
            for i in range(self.n_arms)
        ])
        return np.argmax(samples)
    
    def update(self, arm, reward):
        # Update posterior: Beta-Bernoulli conjugacy
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
    
    def get_estimates(self):
        return self.alpha / (self.alpha + self.beta)

# ========== SIMULATION ==========
np.random.seed(42)
true_probs = [0.2, 0.5, 0.3, 0.7]  # True reward probabilities
n_arms = len(true_probs)
n_rounds = 1000

ts = ThompsonSamplingBernoulli(n_arms)
rewards = []
chosen_arms = []

for t in range(n_rounds):
    arm = ts.select_arm()
    reward = np.random.binomial(1, true_probs[arm])
    ts.update(arm, reward)
    
    rewards.append(reward)
    chosen_arms.append(arm)

# ========== RESULTS ==========
print("Final Estimates:", ts.get_estimates())
print("True Probabilities:", true_probs)
print(f"Best Arm Chosen: {np.bincount(chosen_arms).argmax()}")
print(f"Cumulative Reward: {sum(rewards)}")

# ========== REGRET ANALYSIS ==========
optimal_reward = max(true_probs) * n_rounds
actual_reward = sum(rewards)
regret = optimal_reward - actual_reward
print(f"Regret: {regret:.2f}")

# ========== PLOT POSTERIOR EVOLUTION ==========
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for i, ax in enumerate(axes.flatten()):
    x = np.linspace(0, 1, 100)
    y = stats.beta.pdf(x, ts.alpha[i], ts.beta[i])
    ax.plot(x, y)
    ax.axvline(true_probs[i], color='red', linestyle='--', label='True')
    ax.set_title(f'Arm {i}: α={ts.alpha[i]:.0f}, β={ts.beta[i]:.0f}')
    ax.legend()
plt.tight_layout()
plt.show()
```

---

## Thompson Sampling vs Other Bandits

| Algorithm | Exploration | Pros | Cons |
|-----------|-------------|------|------|
| **Thompson Sampling** | Bayesian uncertainty | Natural uncertainty handling | Requires conjugate priors |
| **Epsilon-Greedy** | Random ε% | Simple | No smart exploration |
| **UCB** | Confidence bounds | Deterministic | Overexplores |

---

## For Gaussian Rewards

```python
class ThompsonSamplingGaussian:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.mu = np.zeros(n_arms)      # Mean
        self.tau = np.ones(n_arms)      # Precision (1/variance)
        self.counts = np.zeros(n_arms)
        
    def select_arm(self):
        samples = np.array([
            np.random.normal(self.mu[i], 1/np.sqrt(self.tau[i]))
            for i in range(self.n_arms)
        ])
        return np.argmax(samples)
    
    def update(self, arm, reward):
        n = self.counts[arm] + 1
        self.mu[arm] = (self.mu[arm] * self.counts[arm] + reward) / n
        self.tau[arm] = n  # Simplified; proper Bayesian update is more complex
        self.counts[arm] = n
```

---

## Applications

| Application | Arms | Reward |
|-------------|------|--------|
| **A/B Testing** | Variants | Conversion rate |
| **Ad Selection** | Ads | Click-through rate |
| **Recommendations** | Items | User engagement |
| **Clinical Trials** | Treatments | Treatment success |

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Non-Stationary Rewards**
> - *Problem:* True probabilities change over time
> - *Solution:* Discounted TS, sliding window
>
> **2. Delayed Rewards**
> - *Problem:* Feedback not immediate
> - *Solution:* Batch updates, careful timing
>
> **3. Context Ignored**
> - *Problem:* User features not used
> - *Solution:* Use [[stats/04_Supervised_Learning/Contextual Bandits\|Contextual Bandits]]

---

## Related Concepts

- [[stats/04_Supervised_Learning/Exploration-Exploitation Trade-off\|Exploration-Exploitation Trade-off]] — Core problem
- [[stats/06_Experimental_Design/Bayesian AB Testing\|Bayesian AB Testing]] — Related application
- [[stats/04_Supervised_Learning/Contextual Bandits\|Contextual Bandits]] — Extension with features

---

## References

- **Paper:** Thompson, W. R. (1933). On the likelihood that one unknown probability exceeds another. *Biometrika*, 25(3/4), 285-294.
- **Tutorial:** Russo, D., et al. (2018). A Tutorial on Thompson Sampling. *Foundations and Trends in Machine Learning*.
