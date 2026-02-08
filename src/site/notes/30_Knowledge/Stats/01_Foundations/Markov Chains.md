---
{"dg-publish":true,"permalink":"/30-knowledge/stats/01-foundations/markov-chains/","tags":["probability","foundations"]}
---

## Definition

> [!abstract] Core Statement
> A **Markov Chain** is a stochastic process where the probability of transitioning to the next state depends ==only on the current state==, not on the sequence of states that preceded it. This is known as the **Markov Property** (memorylessness).

$$P(X_{n+1} = j \mid X_n = i, X_{n-1}, \ldots, X_0) = P(X_{n+1} = j \mid X_n = i) = p_{ij}$$

---

## Purpose

1.  Model systems that evolve randomly over time with limited memory.
2.  Foundation for **MCMC** (Markov Chain Monte Carlo) sampling.
3.  Analyze long-run behavior of stochastic systems.
4.  Power algorithms like **PageRank** and **Hidden Markov Models**.

---

## When to Use

> [!success] Use Markov Chains When...
> - The future state depends **only on the present** (not history).
> - You need to model sequential processes (weather, stock prices, text).
> - Implementing MCMC for Bayesian inference.
> - Analyzing steady-state probabilities.

> [!failure] Challenges
> - Real-world processes often violate the Markov property.
> - High-dimensional state spaces are computationally expensive.
> - Convergence to stationary distribution can be slow.

---

## Theoretical Background

### Transition Matrix

For a discrete Markov chain with $n$ states, the **transition matrix** $P$ is:

$$P = \begin{pmatrix} p_{11} & p_{12} & \cdots & p_{1n} \\ p_{21} & p_{22} & \cdots & p_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ p_{n1} & p_{n2} & \cdots & p_{nn} \end{pmatrix}$$

Where:
- $p_{ij} = P(X_{n+1} = j \mid X_n = i)$
- Each row sums to 1: $\sum_j p_{ij} = 1$

### Key Properties

| Property | Description |
|----------|-------------|
| **Irreducible** | Every state can be reached from every other state. |
| **Aperiodic** | No cyclic behavior (GCD of return times = 1). |
| **Ergodic** | Irreducible + Aperiodic → unique stationary distribution. |
| **Absorbing** | Once entered, cannot be left ($p_{ii} = 1$). |

### Stationary Distribution

A probability distribution $\pi$ is **stationary** if:
$$\pi P = \pi$$

For an ergodic chain, $\pi$ is unique and:
$$\lim_{n \to \infty} P^n = \begin{pmatrix} \pi \\ \pi \\ \vdots \\ \pi \end{pmatrix}$$

---

## Worked Example: Weather Model

> [!example] Problem
> A city has two weather states: Sunny (S) and Rainy (R).
> - If today is Sunny, tomorrow is 70% Sunny, 30% Rainy.
> - If today is Rainy, tomorrow is 40% Sunny, 60% Rainy.
> 
> Find the long-run proportion of sunny days.

**Solution:**

1. **Transition Matrix:**
$$P = \begin{pmatrix} 0.7 & 0.3 \\ 0.4 & 0.6 \end{pmatrix}$$

2. **Find Stationary Distribution:**
   Solve $\pi P = \pi$ with $\pi_S + \pi_R = 1$:
   
   $$\pi_S = 0.7\pi_S + 0.4\pi_R$$
   $$\pi_R = 0.3\pi_S + 0.6\pi_R$$
   
   From the first equation: $0.3\pi_S = 0.4\pi_R \Rightarrow \pi_S = \frac{4}{3}\pi_R$
   
   With $\pi_S + \pi_R = 1$: $\frac{4}{3}\pi_R + \pi_R = 1 \Rightarrow \pi_R = \frac{3}{7}$

3. **Answer:**
   $$\pi = \left(\frac{4}{7}, \frac{3}{7}\right) \approx (0.571, 0.429)$$
   
   **Long-run: ~57% sunny days.**

---

## Python Implementation

```python
import numpy as np

# Transition matrix
P = np.array([
    [0.7, 0.3],  # From Sunny
    [0.4, 0.6]   # From Rainy
])

# Method 1: Power iteration
def stationary_power(P, n_iter=100):
    """Find stationary distribution via matrix powers."""
    P_n = np.linalg.matrix_power(P, n_iter)
    return P_n[0]  # Any row works for ergodic chains

# Method 2: Eigenvalue decomposition
def stationary_eigen(P):
    """Find stationary distribution via left eigenvector."""
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigenvalues - 1))  # Find eigenvalue = 1
    stationary = np.real(eigenvectors[:, idx])
    return stationary / stationary.sum()  # Normalize

print("Power method:", stationary_power(P))
print("Eigen method:", stationary_eigen(P))
# Output: [0.571, 0.429]

# Simulate a chain
def simulate_chain(P, start_state, n_steps):
    """Simulate Markov chain trajectory."""
    states = [start_state]
    for _ in range(n_steps):
        current = states[-1]
        next_state = np.random.choice(len(P), p=P[current])
        states.append(next_state)
    return states

trajectory = simulate_chain(P, start_state=0, n_steps=1000)
print(f"Empirical Sunny proportion: {sum(s==0 for s in trajectory)/len(trajectory):.3f}")
```

---

## R Implementation

```r
library(markovchain)

# Define transition matrix
P <- matrix(c(0.7, 0.3,
              0.4, 0.6), 
            nrow = 2, byrow = TRUE,
            dimnames = list(c("Sunny", "Rainy"), c("Sunny", "Rainy")))

# Create Markov chain object
mc <- new("markovchain", states = c("Sunny", "Rainy"),
          transitionMatrix = P)

# Stationary distribution
stationary <- steadyStates(mc)
print(stationary)
# Output: Sunny = 0.571, Rainy = 0.429

# Simulate chain
set.seed(42)
trajectory <- rmarkovchain(n = 1000, object = mc, t0 = "Sunny")
table(trajectory) / length(trajectory)
```

---

## ML Applications

| Application | How Markov Chains Are Used |
|-------------|---------------------------|
| **MCMC Sampling** | Generate samples from complex posterior distributions. |
| **PageRank** | Web as a Markov chain; stationary dist. = page importance. |
| **HMM** | Hidden states form a Markov chain; emissions are observed. |
| **Reinforcement Learning** | MDP states follow Markov dynamics. |
| **Text Generation** | n-gram models are (n-1)th order Markov chains. |

---

## Interpretation Guide

| Output | Interpretation |
|--------|----------------|
| **Stationary π = (0.6, 0.4)** | Long-run: 60% in state 1, 40% in state 2. |
| **Mixing Time = 50** | ~50 steps needed to approach stationarity. |
| **Eigenvalue gap large** | Fast convergence to stationary distribution. |
| **Absorbing state found** | Chain will eventually get stuck there. |

---

## Limitations

> [!warning] Pitfalls
> 1. **Markov Assumption Violated:** Real data often has longer memory.
> 2. **State Space Explosion:** Continuous or high-dimensional states are problematic.
> 3. **Non-Ergodic Chains:** May not have a unique stationary distribution.
> 4. **Slow Mixing:** MCMC may take very long to explore the state space.

---

## Related Concepts

- [[30_Knowledge/Stats/01_Foundations/Monte Carlo Simulation\|Monte Carlo Simulation]] - Random sampling methods.
- [[30_Knowledge/Stats/01_Foundations/Bayesian Statistics\|Bayesian Statistics]] - MCMC is key for Bayesian inference.
- [[30_Knowledge/Stats/04_Supervised_Learning/Reinforcement Learning\|Reinforcement Learning]] - MDPs are Markov chains with actions.
- [[30_Knowledge/Stats/01_Foundations/Law of Large Numbers\|Law of Large Numbers]] - Ergodic theorem is the Markov version.

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## References

- **Book:** Norris, J. R. (1998). *Markov Chains*. Cambridge University Press.
- **Book:** Levin, D. A., Peres, Y., & Wilmer, E. L. (2017). *Markov Chains and Mixing Times* (2nd ed.). AMS.
- **Article:** Geman, S., & Geman, D. (1984). Stochastic relaxation, Gibbs distributions, and the Bayesian restoration of images. *IEEE TPAMI*, 6(6), 721-741.
