---
{"dg-publish":true,"permalink":"/stats/07-ethics-and-biases/publication-bias/","tags":["Ethics","Meta-Analysis","Critical-Thinking","Reporting"]}
---

## Definition

> [!abstract] Core Statement
> **Publication Bias** (the "File Drawer Problem") is the phenomenon where studies with ==statistically significant or "positive" results== are more likely to be published than those with null or negative results. This creates a distorted view of the evidence, often overestimating effect sizes in scientific literature.

---

> [!tip] Intuition (ELI5): The Magic Trick
> Imagine a magician tries a trick 100 times. He fails 99 times but only posts the **one successful video** on YouTube. People think he's a master, but they don't see the "failed" videos hidden in the trash (the file drawer). 

> [!example] Real-Life Example: Pharmaceutical Trials
> A company runs 10 trials for a drug. In 8 trials, it fails, so they "file" them away. In 2 trials, it shows a tiny benefit by chance, and *those* are the ones doctors see in journals. This makes the drug look much more effective than it actually is.

---

## Purpose

1.  **Improving Meta-Analyses:** Understand why combining published studies can lead to false conclusions.
2.  **Scientific Integrity:** Promote the reporting of all findings, regardless of the $p$-value.
3.  **Critical Evaluation:** Recognizing that the absence of published negative results is itself a "red flag."

---

## Detection: The Funnel Plot

| Method | Logics | Interpretation |
| :--- | :--- | :--- |
| **Funnel Plot** | Scatter plot of treatment effect vs. study size (precision). | Asymmetric "funnel" suggests missing negative studies (usually small ones). |
| **Egger's Test** | Regression-based test for funnel plot asymmetry. | Significant intercept indicates potential bias. |
| **Fail-Safe N** | Number of null studies needed to reverse a significant meta-analysis. | Low N means the result is fragile. |

---

## Theoretical Background

### The File Drawer Effect
If researchers only submit significant results ($p < 0.05$), and journals only publish them, the literature becomes a collection of **Type I Errors** (False Positives) and over-optimistic effect sizes.
- **Winner's Curse:** The first published study on a new effect usually reports the largest effect size, which then shrinks in subsequent replications.

---

## R Simulation: Visualizing Publication Bias (Funnel Plot)

```r
library(meta)

# Simulate 50 small studies with NO real effect (mean = 0)
set.seed(42)
n_studies <- 50
effects <- rnorm(n_studies, mean = 0, sd = 0.5)
se <- runif(n_studies, 0.1, 0.5)

# P-values
z <- effects / se
p_vals <- 2 * (1 - pnorm(abs(z)))

# Reality: All studies
all_data <- data.frame(effects, se, p_vals)

# HACK: Only "publish" significant ones (p < 0.05) or those with strong trends
published <- all_data[all_data$p_vals < 0.2, ] # More lenient for simulation

# Funnel Plot
funnel(metagen(published$effects, published$se))
title("Funnel Plot: Asymmetry indicates Publication Bias")
```

---

## Related Concepts

- [[stats/07_Ethics_and_Biases/P-Hacking\|P-Hacking]] - The process used to generate the "publishable" results.
- [[stats/01_Foundations/Reproducibility Crisis\|Reproducibility Crisis]] - Partially caused by publication bias.
- [[Meta-Analysis\|Meta-Analysis]] - Highly sensitive to this bias.

---

## References

- **Historical:** Rosenthal, R. (1979). The file drawer problem and tolerance for null results. *Psychological Bulletin*. [APA PsycNet](https://doi.org/10.1037/0033-2909.86.3.638)
- **Article:** Ioannidis, J. P. (2005). Why most published research findings are false. *PLoS Medicine*. [DOI: 10.1371/journal.pmed.0020124](https://doi.org/10.1371/journal.pmed.0020124)
- **Book:** Rothstein, H. R., et al. (2005). *Publication Bias in Meta-Analysis: Prevention, Assessment and Adjustments*. Wiley. [Wiley Link](https://www.wiley.com/en-us/Publication+Bias+in+Meta+Analysis%3A+Prevention%2C+Assessment+and+Adjustments-p-9780470870143)
