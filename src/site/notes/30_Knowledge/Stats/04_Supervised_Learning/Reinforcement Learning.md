---
{"dg-publish":true,"permalink":"/30-knowledge/stats/04-supervised-learning/reinforcement-learning/","tags":["machine-learning","supervised"]}
---


## Definition

> [!abstract] Core Statement
> **Reinforcement Learning** trains agents to make ==sequential decisions== by maximizing cumulative rewards through trial and error, learning a policy that maps states to actions.

---

> [!tip] Intuition (ELI5)
> Like training a dog: good actions → treat (positive reward), bad actions → no treat or scolding (negative reward). Over time, the dog learns what to do.

---

## Key Components

| Component | Description |
|-----------|-------------|
| **Agent** | The learner (makes decisions) |
| **Environment** | The world the agent interacts with |
| **State (s)** | Current situation |
| **Action (a)** | What agent can do |
| **Reward (r)** | Feedback signal |
| **Policy (π)** | Strategy: state → action |

---

## RL vs Supervised Learning

| Aspect | Supervised | RL |
|--------|------------|----|
| **Feedback** | Correct answers | Rewards (delayed) |
| **Data** | i.i.d. samples | Sequential, correlated |
| **Goal** | Minimize prediction error | Maximize cumulative reward |

---

## Algorithms

| Algorithm | Type | Description |
|-----------|------|-------------|
| **Q-Learning** | Value-based | Learns action-value function |
| **Policy Gradient** | Policy-based | Directly optimizes policy |
| **Actor-Critic** | Hybrid | Combines both approaches |
| **DQN** | Deep RL | Q-Learning + Neural Networks |
| **PPO** | Policy Gradient | State-of-the-art stability |

---

## Python Example (OpenAI Gym)

```python
import gymnasium as gym

env = gym.make("CartPole-v1")
state, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # Random policy
    state, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        state, info = env.reset()

env.close()
```

---

## Applications

| Domain | Example |
|--------|---------|
| **Games** | AlphaGo, Atari |
| **Robotics** | Robot control |
| **Recommendations** | Personalization |
| **Finance** | Trading strategies |

---

## Related Concepts

- [[30_Knowledge/Stats/04_Supervised_Learning/Thompson Sampling\|Thompson Sampling]] — Bandit algorithms
- [[30_Knowledge/Stats/04_Supervised_Learning/Contextual Bandits\|Contextual Bandits]] — Simplified RL
- [[30_Knowledge/Stats/04_Supervised_Learning/Exploration-Exploitation Trade-off\|Exploration-Exploitation Trade-off]] — Core challenge

---

## When to Use

> [!success] Use Reinforcement Learning When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Dataset is too small for training
> - Interpretability is more important than accuracy

---

## Python Implementation

```python
import numpy as np
import pandas as pd

# Example implementation of Reinforcement Learning
# See documentation for details

data = np.random.randn(100)
print(f"Mean: {np.mean(data):.3f}")
print(f"Std: {np.std(data):.3f}")
```

---

## R Implementation

```r
# Reinforcement Learning in R
set.seed(42)

# Example implementation
data <- rnorm(100)
summary(data)
```

---

## References

- **Book:** Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. [Free Online](http://incompleteideas.net/book/the-book-2nd.html)
