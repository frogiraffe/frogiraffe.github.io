---
{"dg-publish":true,"permalink":"/stats/06-causal-inference/causal-inference/","tags":["Causal-Inference","Econometrics","Theory"]}
---


## Definition

> [!abstract] Overview
> **Causal Inference** is the scientific process of drawing conclusions about causal relationships from data. Unlike correlation ("Ice cream sales and murders are correlated"), causal inference asks: "Does X **cause** Y?" and "What would Y be if we intervened on X?" (Counterfactuals).

---

## 1. The Fundamental Problem

We cannot observe the same unit in two different states simultaneously. 

For a patient taking a drug ($X=1$), we observe the outcome $Y_1$. We **cannot** observe $Y_0$ (what would have happened if they didn't take it). $Y_0$ is the **Counterfactual**.

**Average Treatment Effect (ATE):**
$$ ATE = E[Y_1 - Y_0] $$

---

## 2. Correlation vs Causation

Comparison implies causation ONLY IF **all other variables are equal** (Ceteris Paribus).

**Selection Bias:** People who *choose* to take a medicine might be sicker than those who don't. Comparing them directly provides a biased estimate.

$$ \text{Correlation} = \text{Causation} + \text{Bias} $$

---

## 3. The Hierarchy of Evidence

1.  **Randomized Controlled Trial (RCT):** The Gold Standard. Randomization eliminates bias.
2.  **Natural Experiments:** Exploiting random events in nature (e.g., lottery winners).
3.  **Quasi-Experiments:** Using statistical methods to adjust for bias (PSM, DiD).
4.  **Observational Studies:** High risk of confounding.

---

## 4. Key Methods

- [[stats/06_Causal_Inference/Propensity Score Matching (PSM)\|Propensity Score Matching (PSM)]] - Creating artificial control groups.
- [[stats/06_Causal_Inference/Difference-in-Differences (DiD)\|Difference-in-Differences (DiD)]] - Comparing trends over time.
- [[stats/06_Causal_Inference/Instrumental Variables (IV)\|Instrumental Variables (IV)]] - Using external shocks to isolate causal parts of $X$.
- [[stats/03_Regression_Analysis/Regression Discontinuity Design (RDD)\|Regression Discontinuity Design (RDD)]] - Comparing units just above/below a cutoff.

---

## Related Concepts

- [[stats/01_Foundations/Confounding Variables\|Confounding Variables]]
- [[Simpson's Paradox\|Simpson's Paradox]]
- [[stats/01_Foundations/Structural Equation Modeling (SEM)\|Structural Equation Modeling (SEM)]]
- [[Judea Pearl\|Judea Pearl]] (Ladder of Causality)
