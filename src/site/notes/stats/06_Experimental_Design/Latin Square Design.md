---
{"dg-publish":true,"permalink":"/stats/06-experimental-design/latin-square-design/","tags":["Experimental-Design","DOE","Blocking"]}
---


## Definition

> [!abstract] Core Statement
> A **Latin Square Design** is an experimental arrangement where each treatment appears ==exactly once in each row and each column==, controlling for two blocking factors simultaneously while using fewer runs than a full factorial.

---

> [!tip] Intuition (ELI5): The Sudoku Experiment
> Imagine testing 4 fertilizers on a farm with 4 rows (soil type) and 4 columns (sunlight). A Latin Square ensures each fertilizer appears once per row and once per column — like a mini Sudoku! This way, you fairly test all fertilizers despite field variations.

---

## Purpose

1. **Double Blocking:** Control two nuisance variables (row and column effects)
2. **Efficiency:** Test $n$ treatments in only $n^2$ runs (vs $n^3$ for full design)
3. **Fairness:** Every treatment faces every level of each blocking factor exactly once

---

## When to Use

> [!success] Use Latin Square When...
> - You have **two nuisance factors** (not of primary interest)
> - Both blocking factors have the **same number of levels** as treatments
> - You assume **no interaction** between rows, columns, and treatments

> [!failure] Avoid Latin Square When...
> - Row × Treatment or Column × Treatment **interactions** are expected
> - Blocking factors have **different numbers of levels**
> - You need to test for **interaction effects** → Use [[stats/06_Experimental_Design/Factorial Design (2k)\|Factorial Design (2k)]]

---

## Structure

For $n$ treatments (A, B, C, ...), the design is an $n \times n$ grid:

### Example: 4 × 4 Latin Square

|  | Col 1 | Col 2 | Col 3 | Col 4 |
|--|-------|-------|-------|-------|
| **Row 1** | A | B | C | D |
| **Row 2** | B | C | D | A |
| **Row 3** | C | D | A | B |
| **Row 4** | D | A | B | C |

**Properties:**
- Each letter appears once per row ✓
- Each letter appears once per column ✓
- Total runs = $n^2 = 16$

---

## Statistical Model

$$
Y_{ijk} = \mu + \alpha_i + \beta_j + \tau_k + \epsilon_{ijk}
$$

Where:
- $\mu$ = Grand mean
- $\alpha_i$ = Row effect (nuisance)
- $\beta_j$ = Column effect (nuisance)
- $\tau_k$ = Treatment effect (of interest)
- $\epsilon_{ijk} \sim N(0, \sigma^2)$ = Error

### ANOVA Table

| Source | df | SS | MS | F |
|--------|----|----|-----|---|
| Rows | $n-1$ | $SS_R$ | $MS_R$ | $MS_R/MS_E$ |
| Columns | $n-1$ | $SS_C$ | $MS_C$ | $MS_C/MS_E$ |
| Treatments | $n-1$ | $SS_T$ | $MS_T$ | $MS_T/MS_E$ |
| Error | $(n-1)(n-2)$ | $SS_E$ | $MS_E$ | |
| Total | $n^2-1$ | $SS_{Total}$ | | |

---

## Worked Example

> [!example] Agricultural Trial
> **Research Question:** Compare 4 irrigation methods (A, B, C, D) across fields with varying soil (rows) and sunlight (columns).
> 
> **Design:**
> 
> |  | Low Sun | Med Sun | High Sun | Full Sun |
> |--|---------|---------|----------|----------|
> | **Sandy** | A (12) | B (15) | C (18) | D (14) |
> | **Clay** | B (16) | C (20) | D (17) | A (13) |
> | **Loam** | C (22) | D (19) | A (14) | B (18) |
> | **Peat** | D (16) | A (11) | B (17) | C (21) |
> 
> **Analysis:** ANOVA separates row, column, and treatment effects.

---

## Python Implementation

```python
import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# ========== DATA ==========
data = pd.DataFrame({
    'Row': ['Sandy', 'Sandy', 'Sandy', 'Sandy', 
            'Clay', 'Clay', 'Clay', 'Clay',
            'Loam', 'Loam', 'Loam', 'Loam',
            'Peat', 'Peat', 'Peat', 'Peat'],
    'Col': ['Low', 'Med', 'High', 'Full'] * 4,
    'Treatment': ['A', 'B', 'C', 'D', 
                  'B', 'C', 'D', 'A',
                  'C', 'D', 'A', 'B',
                  'D', 'A', 'B', 'C'],
    'Yield': [12, 15, 18, 14, 16, 20, 17, 13, 22, 19, 14, 18, 16, 11, 17, 21]
})

# ========== ANOVA ==========
model = ols('Yield ~ C(Row) + C(Col) + C(Treatment)', data=data).fit()
anova_table = anova_lm(model, typ=2)
print(anova_table)

# ========== TREATMENT MEANS ==========
print("\nTreatment Means:")
print(data.groupby('Treatment')['Yield'].mean())
```

---

## R Implementation

```r
# ========== DATA ==========
yield_data <- c(12, 15, 18, 14, 16, 20, 17, 13, 22, 19, 14, 18, 16, 11, 17, 21)

row <- factor(rep(1:4, each = 4))
col <- factor(rep(1:4, times = 4))
treatment <- factor(c("A", "B", "C", "D", 
                      "B", "C", "D", "A",
                      "C", "D", "A", "B",
                      "D", "A", "B", "C"))

df <- data.frame(Row = row, Col = col, Treatment = treatment, Yield = yield_data)

# ========== ANOVA ==========
model <- aov(Yield ~ Row + Col + Treatment, data = df)
summary(model)

# ========== POST-HOC ==========
TukeyHSD(model, "Treatment")
```

---

## Generating Latin Squares

```python
def random_latin_square(n):
    """Generate a random n×n Latin Square"""
    from itertools import permutations
    import random
    
    # Start with standard Latin square
    square = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Shuffle rows and columns
    random.shuffle(square)
    square = list(map(list, zip(*square)))  # Transpose
    random.shuffle(square)
    square = list(map(list, zip(*square)))  # Transpose back
    
    return square

# Example
ls = random_latin_square(4)
for row in ls:
    print([chr(65 + x) for x in row])
```

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Assuming No Interactions**
> - *Problem:* If Treatment × Row interaction exists, F-test for treatment is biased
> - *Solution:* Use Graeco-Latin squares or replicated Latin squares if interactions suspected
>
> **2. Size Constraint**
> - *Problem:* Can only test $n$ treatments with exactly $n$ rows and $n$ columns
> - *Solution:* Use Youden squares for rectangular designs
>
> **3. One Latin Square = One Replicate**
> - *Problem:* Low power with only $n^2$ observations
> - *Solution:* Use multiple Latin squares (replications)

---

## Related Concepts

**Prerequisites:**
- [[stats/02_Statistical_Inference/One-Way ANOVA\|One-Way ANOVA]] — Single factor analysis
- [[stats/06_Experimental_Design/Randomized Block Design\|Randomized Block Design]] — Simpler blocking

**Extensions:**
- **Graeco-Latin Square** — Three blocking factors
- **Youden Square** — Incomplete Latin square (fewer columns than rows)
- [[stats/06_Experimental_Design/Factorial Design (2k)\|Factorial Design (2k)]] — With interactions

---

## References

- **Book:** Montgomery, D. C. (2017). *Design and Analysis of Experiments* (9th ed.). Wiley. (Chapter 4) [Wiley Link](https://www.wiley.com/en-us/Design+and+Analysis+of+Experiments%2C+9th+Edition-p-9781119113478)
- **Historical:** Fisher, R. A. (1925). *Statistical Methods for Research Workers*. Oliver and Boyd.
- **Book:** Box, G. E. P., Hunter, J. S., & Hunter, W. G. (2005). *Statistics for Experimenters*. Wiley.
