---
{"dg-publish":true,"permalink":"/stats/02-statistical-inference/absolute-risk-reduction/","tags":["Epidemiology","Medical-Statistics","Effect-Size"]}
---


## Definition

> [!abstract] Core Statement
> **Absolute Risk Reduction** is the ==difference in event rates== between control and treatment groups. It shows the actual decrease in risk, not just the relative change.

$$
\text{ARR} = \text{Risk}_{control} - \text{Risk}_{treatment}
$$

---

## Example

| Group | Events | Total | Risk |
|-------|--------|-------|------|
| Control | 10 | 100 | 10% |
| Treatment | 5 | 100 | 5% |

- **ARR** = 10% - 5% = **5%**
- **[[stats/02_Statistical_Inference/Relative Risk\|Relative Risk]]** = 5% / 10% = 0.5 (50% reduction)

> [!tip] Key Insight
> RR = 50% reduction sounds impressive, but ARR = 5% shows only 5 in 100 benefit.

---

## Number Needed to Treat (NNT)

$$
\text{NNT} = \frac{1}{\text{ARR}}
$$

In example: NNT = 1/0.05 = **20** (treat 20 patients to prevent 1 event)

---

## Python Implementation

```python
# Control: 10/100 events, Treatment: 5/100 events
control_risk = 10 / 100
treatment_risk = 5 / 100

arr = control_risk - treatment_risk
nnt = 1 / arr

print(f"Absolute Risk Reduction: {arr:.1%}")
print(f"Number Needed to Treat: {nnt:.0f}")
```

---

## ARR vs RRR

| Measure | Calculation | Interpretation |
|---------|-------------|----------------|
| **ARR** | Risk₀ - Risk₁ | Absolute difference |
| **RRR** | 1 - (Risk₁/Risk₀) | % reduction |
| **NNT** | 1/ARR | Patients per benefit |

> [!warning] Be Careful
> RRR sounds more impressive but ignores baseline risk.
> - Low baseline (1% → 0.5%): ARR = 0.5%, NNT = 200
> - High baseline (50% → 25%): ARR = 25%, NNT = 4

---

## Related Concepts

- [[stats/02_Statistical_Inference/Relative Risk\|Relative Risk]] — RR = Risk₁/Risk₀
- [[stats/01_Foundations/Odds Ratio\|Odds Ratio]] — For case-control studies
- [[stats/02_Statistical_Inference/Hazard Ratio\|Hazard Ratio]] — For survival data

---

## References

- **Book:** Sackett, D. L., et al. (1996). *Evidence-Based Medicine*. Churchill Livingstone.
