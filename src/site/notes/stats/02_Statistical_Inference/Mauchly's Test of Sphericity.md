---
{"dg-publish":true,"permalink":"/stats/02-statistical-inference/mauchly-s-test-of-sphericity/","tags":["ANOVA","Repeated-Measures","Diagnostics"]}
---


## Definition

> [!abstract] Core Statement
> **Mauchly's Test of Sphericity** tests whether the ==variances of differences between conditions are equal==, an assumption required for repeated-measures ANOVA.

$$
H_0: \text{Sphericity holds (variances of differences are equal)}
$$

---

## When To Use

| Situation | Action |
|-----------|--------|
| Repeated-measures ANOVA | Always check |
| p > 0.05 | Sphericity OK, proceed |
| p < 0.05 | Sphericity violated, use correction |

---

## Corrections for Violation

| Correction | When |
|------------|------|
| **Greenhouse-Geisser (ε)** | Always safe, conservative |
| **Huynh-Feldt** | Less conservative, if ε > 0.75 |
| **Lower-bound** | Most conservative |

---

## Python Implementation

```python
import pingouin as pg

# Repeated-measures ANOVA with sphericity test
aov = pg.rm_anova(data=df, dv='score', within='condition', subject='subject_id')
print(aov)

# Sphericity test (built into rm_anova output)
# Look at 'spher' and 'W-spher' columns
```

---

## R Implementation

```r
library(ez)

model <- ezANOVA(
  data = df,
  dv = score,
  wid = subject_id,
  within = condition
)

# Mauchly's test in output
print(model---
{"dg-publish":true,"permalink":"/stats/02-statistical-inference/mauchly-s-test-of-sphericity/","tags":["ANOVA","Repeated-Measures","Diagnostics"]}
---


## Definition

> [!abstract] Core Statement
> **Mauchly's Test of Sphericity** tests whether the ==variances of differences between conditions are equal==, an assumption required for repeated-measures ANOVA.

$$
H_0: \text{Sphericity holds (variances of differences are equal)}
$$

---

## When To Use

| Situation | Action |
|-----------|--------|
| Repeated-measures ANOVA | Always check |
| p > 0.05 | Sphericity OK, proceed |
| p < 0.05 | Sphericity violated, use correction |

---

## Corrections for Violation

| Correction | When |
|------------|------|
| **Greenhouse-Geisser (ε)** | Always safe, conservative |
| **Huynh-Feldt** | Less conservative, if ε > 0.75 |
| **Lower-bound** | Most conservative |

---

## Python Implementation

```python
import pingouin as pg

# Repeated-measures ANOVA with sphericity test
aov = pg.rm_anova(data=df, dv='score', within='condition', subject='subject_id')
print(aov)

# Sphericity test (built into rm_anova output)
# Look at 'spher' and 'W-spher' columns
```

---

## R Implementation

Mauchly's Test for Sphericity`)

# Corrections
print(model---
{"dg-publish":true,"permalink":"/stats/02-statistical-inference/mauchly-s-test-of-sphericity/","tags":["ANOVA","Repeated-Measures","Diagnostics"]}
---


## Definition

> [!abstract] Core Statement
> **Mauchly's Test of Sphericity** tests whether the ==variances of differences between conditions are equal==, an assumption required for repeated-measures ANOVA.

$$
H_0: \text{Sphericity holds (variances of differences are equal)}
$$

---

## When To Use

| Situation | Action |
|-----------|--------|
| Repeated-measures ANOVA | Always check |
| p > 0.05 | Sphericity OK, proceed |
| p < 0.05 | Sphericity violated, use correction |

---

## Corrections for Violation

| Correction | When |
|------------|------|
| **Greenhouse-Geisser (ε)** | Always safe, conservative |
| **Huynh-Feldt** | Less conservative, if ε > 0.75 |
| **Lower-bound** | Most conservative |

---

## Python Implementation

```python
import pingouin as pg

# Repeated-measures ANOVA with sphericity test
aov = pg.rm_anova(data=df, dv='score', within='condition', subject='subject_id')
print(aov)

# Sphericity test (built into rm_anova output)
# Look at 'spher' and 'W-spher' columns
```

---

## R Implementation

Sphericity Corrections`)
```

---

## Related Concepts

- [[stats/02_Statistical_Inference/One-Way ANOVA\|One-Way ANOVA]] — Between-subjects version
- [[stats/02_Statistical_Inference/Repeated Measures ANOVA\|Repeated Measures ANOVA]] — Where sphericity matters

---

## References

- **Paper:** Mauchly, J. W. (1940). Significance test for sphericity of a normal n-variate distribution. *Annals of Mathematical Statistics*, 11(2), 204-209.
