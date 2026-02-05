---
{"dg-publish":true,"permalink":"/stats/10-ethics-and-biases/feedback-loops/","tags":["ethics","systems-thinking","ai-safety","reinforcement-learning"]}
---

## Definition

> [!abstract] Core Statement
> A **Feedback Loop** in data science occurs when the ==outputs of an algorithm influence the future inputs== of the same algorithm. If not managed, this can create a "self-fulfilling prophecy" where biases are reinforced and amplified over time.

---

> [!tip] Intuition (ELI5): The Snack Trap
> Imagine a computer guesses you like **Apples**. It *only* shows you Apple ads. Because you only see apples, you buy another apple. The computer thinks, "I was right! He *only* likes apples," and stops showing you anything else. You are trapped in a loop of your own past actions.

> [!example] Real-Life Example: Predictive Policing
> Algorithms send more police to **Neighborhood A** based on old data. Because there are more police there, they catch more petty crimes (like jaywalking) that aren't caught elsewhere. This "new crime data" makes the algorithm even more sure Neighborhood A is the problem, creating a cycle that makes the prediction "true" by force.

---

## Purpose

1.  **System Stability:** Preventing algorithms from diverging into extreme behaviors.
2.  **Mitigating Bias:** Understanding how "Predictive Policing" or "Credit Scoring" can lock groups into cycles of disadvantage.
3.  **Recommendation Safety:** Preventing "Echo Chambers" where users are trapped in narrow information loops.

---

## The Cycle of Reinforcement

| Step | Example: Predictive Policing |
| :--- | :--- |
| **1. Biased Input** | Historical arrest data is higher in Neighborhood A (due to over-policing). |
| **2. Algorithm Prediction** | The model predicts more crime will occur in Neighborhood A. |
| **3. Action (Feedback)** | Police dispatch more officers to Neighborhood A. |
| **4. New Observation** | More officers find more petty crimes. This becomes new "Training Data." |
| **5. Amplification** | The model is even more certain Neighborhood A is the problem. |

---

## Theoretical Background: Control Theory Logic

In control theory, this is a **Positive Feedback Loop** (which is usually destructive).
- **Negative Feedback:** Self-correcting (like a thermostat).
- **Positive Feedback:** Self-amplifying (like a microphone squeal). 
In statistics, this violates the assumption that observations are **independent and identically distributed (i.i.d.)**.

---

## Python Simulation: The Echo Chamber Effect

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulate a user's interest (0 to 1)
true_interest = 0.5
user_history = []
algorithmic_recommendations = []

# Loop: Algorithm recommends based on past average
for i in range(100):
    # Current belief of the algo
    current_algo_belief = np.mean(user_history) if user_history else 0.5
    
    # Algo recommends something near its belief
    recommendation = np.clip(current_algo_belief + np.random.normal(0, 0.1), 0, 1)
    algorithmic_recommendations.append(recommendation)
    
    # User interacts (sways slightly toward what is shown)
    user_interaction = np.clip(0.8 * true_interest + 0.2 * recommendation, 0, 1)
    user_history.append(user_interaction)

plt.figure(figsize=(10, 5))
plt.plot(user_history, label='User Behavior')
plt.plot(algorithmic_recommendations, label='Algorithm Recommendations', alpha=0.5)
plt.axhline(true_interest, color='red', linestyle='--', label='Original True Interest')
plt.title("Feedback Loop: User behavior converges toward Algorithm recommendations")
plt.legend()
plt.show()
```

---

## Related Concepts

- [[stats/10_Ethics_and_Biases/Algorithmic Bias\|Algorithmic Bias]] - The catalyst for dangerous loops.
- [[stats/04_Supervised_Learning/Reinforcement Learning\|Reinforcement Learning]] - Highly susceptible to feedback effects.
- [[stats/10_Ethics_and_Biases/Selection Bias\|Selection Bias]] - Feedback loops are essentially a dynamic selection bias.

---

## References

- **Article:** Lum, K., & Isaac, W. (2016). To predict and serve? *Significance*. [Wiley Link](https://rss.onlinelibrary.wiley.com/doi/full/10.1111/j.1740-9713.2016.00960.x)
- **Paper:** Ensign, D., et al. (2018). Runaway Feedback Loops in Predictive Policing. *FAT**. [arXiv](https://arxiv.org/abs/1706.09847)
- **Book:** Meadows, D. H. (2008). *Thinking in Systems: A Primer*. Chelsea Green. [Publisher](https://www.chelseagreen.com/product/thinking-in-systems/)
