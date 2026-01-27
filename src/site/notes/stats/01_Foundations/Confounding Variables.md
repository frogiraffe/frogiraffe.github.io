---
{"dg-publish":true,"permalink":"/stats/01-foundations/confounding-variables/","tags":["Causal-Inference","Bias","Study-Design"]}
---


## Definition

> [!abstract] Core Statement
> A **Confounding Variable** is a variable that ==influences both the treatment and the outcome==, creating a spurious association.

$$X \leftarrow C \rightarrow Y$$

C confounds the X-Y relationship.

---

## Classic Example

**Ice cream sales** correlate with **drowning deaths**.

Confounder: **Hot weather** causes both more ice cream and more swimming.

---

## Detection

- Theoretical: Does variable plausibly affect both X and Y?
- Statistical: Compare adjusted vs. unadjusted estimates

---

## Solutions

| Method | Approach |
|--------|----------|
| **Randomization** | Breaks confounding link to X |
| **Stratification** | Analyze within confounder strata |
| **Matching** | Match treated/control on confounders |
| **Regression adjustment** | Include confounder as covariate |

---

## Python Example

```python
import statsmodels.api as sm

# Unadjusted
unadj = sm.OLS(y, sm.add_constant(treatment)).fit()

# Adjusted for confounder
adj = sm.OLS(y, sm.add_constant(df[['treatment', 'confounder']])).fit()

print(f"Unadjusted effect: {unadj.params['treatment']:.3f}")
print(f"Adjusted effect: {adj.params['treatment']:.3f}")
```

---

## Related Concepts

- [[stats/01_Foundations/Selection Bias\|Selection Bias]] - Different threat to validity
- [[stats/01_Foundations/Randomized Controlled Trials\|Randomized Controlled Trials]] - Eliminates confounding
- [[stats/07_Causal_Inference/Propensity Score Matching (PSM)\|Propensity Score Matching]] - Adjusts for confounders

---

## References

- **Book:** Hernán, M. A., & Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall. [Free Online Version](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/)
