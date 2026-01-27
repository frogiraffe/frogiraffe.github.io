---
{"dg-publish":true,"permalink":"/stats/01-foundations/selection-bias/","tags":["Bias","Causal-Inference","Study-Design"]}
---


## Definition

> [!abstract] Core Statement
> **Selection Bias** occurs when the ==sample is not representative of the target population== due to systematic differences in how subjects were selected or retained.

---

## Types

| Type | Example |
|------|---------|
| **Self-selection** | Volunteers differ from non-volunteers |
| **Survivorship** | Only successful cases are observed |
| **Attrition** | Dropouts differ from completers |
| **Healthy worker** | Employed people are healthier |
| **Berksonian** | Hospital samples distort disease association |

---

## Example: Survivorship Bias

WWII planes returning with bullet holes → reinforce those spots?

**Wrong!** Returning planes show where planes can survive hits. Reinforce spots with NO holes (downed planes were hit there).

---

## Detection & Prevention

| Prevention | Method |
|------------|--------|
| **Random sampling** | Every unit has known probability |
| **Intention to treat** | Analyze as randomized |
| **Sensitivity analysis** | Model selection mechanisms |

---

## Related Concepts

- [[stats/01_Foundations/Confounding Variables\|Confounding Variables]] - Another threat to validity
- [[stats/01_Foundations/Randomized Controlled Trials\|Randomized Controlled Trials]] - Reduces selection bias
- [[stats/01_Foundations/Sample Ratio Mismatch (SRM)\|Sample Ratio Mismatch (SRM)]] - A/B test selection issue

---

## References

- **Book:** Hernán, M. A., & Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall. [Harvard Link](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/)
