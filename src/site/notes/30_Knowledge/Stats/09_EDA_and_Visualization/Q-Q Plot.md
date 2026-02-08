---
{"dg-publish":true,"permalink":"/30-knowledge/stats/09-eda-and-visualization/q-q-plot/","tags":["eda","visualization"]}
---

## Definition

> [!abstract] Core Statement
> A **Q-Q Plot (Quantile-Quantile Plot)** is a graphical tool to compare the distribution of a sample against a theoretical distribution (usually the [[30_Knowledge/Stats/01_Foundations/Normal Distribution\|Normal Distribution]]). It plots ==sample quantiles against theoretical quantiles==. If the data follows the theoretical distribution, points will fall on a straight diagonal line.

![Q-Q Plot Visualization](https://upload.wikimedia.org/wikipedia/commons/0/08/Normal_normal_qq.svg)

---

> [!tip] Intuition (ELI5): The Mirror Mirror
> A Q-Q plot is like a mirror for data. It asks: "Does this messy group of numbers look like the perfect Bell Curve?" If they match, they'll hold hands along a perfectly straight diagonal line. If they curve away, it shows you exactly how the data is "bent" compared to a normal distribution.

---

## Purpose

1.  Assess **normality** of data or residuals.
2.  Diagnose deviations: skewness, heavy tails, outliers.
3.  Complement statistical tests (e.g., [[30_Knowledge/Stats/02_Statistical_Inference/Shapiro-Wilk Test\|Shapiro-Wilk Test]]) with visual evidence.

---

## When to Use

> [!success] Use Q-Q Plot When...
> - Checking normality assumption for t-tests, ANOVA, or regression residuals.
> - Sample size is moderate to large (where histograms are also useful).
> - You want to understand *how* normality is violated (skew? tails?).

---

## Theoretical Background

### Construction

1.  Sort the data.
2.  Calculate sample quantiles (percentiles).
3.  Calculate theoretical quantiles from $N(0,1)$.
4.  Plot pairs: (Theoretical $Z$, Sample Value).
5.  Add a 45-degree reference line.

### Reading the Plot

| Pattern | Interpretation | Example |
|---------|----------------|---------|
| **Straight Line** | Data is Normal. | Normal residuals. |
| **S-Shape** | Light tails (Uniform). | Limited variation. |
| **Curve (Concave)** | Right Skew. | Income data. Use Log Transformations. |
| **Curve (Convex)** | Left Skew. | Ceiling effects. |
| **Tails Depart Upward** | Heavy right tail (Outliers). | Check for extreme values. |
| **Tails Depart Downward** | Heavy left tail. | Negative outliers. |

---

## Assumptions

Q-Q plot is a diagnostic tool, not a test; no formal assumptions. However, interpretation assumes data is continuous.

---

## Limitations

> [!warning] Pitfalls
> 1.  **Subjectivity:** "Straight enough" is a judgment call, especially for small $n$.
> 2.  **Small samples:** Random variation can look like deviation.
> 3.  **Not a formal test:** Use [[30_Knowledge/Stats/02_Statistical_Inference/Shapiro-Wilk Test\|Shapiro-Wilk Test]] for statistical confirmation.

---

## Python Implementation

```python
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Create Q-Q Plot
sm.qqplot(data, line='45', fit=True)
plt.title("Q-Q Plot")
plt.show()

# For regression residuals
sm.qqplot(model.resid, line='45', fit=True)
plt.title("Q-Q Plot of Residuals")
plt.show()
```

---

## R Implementation

```r
# Q-Q Plot for a vector
qqnorm(data, main = "Q-Q Plot")
qqline(data, col = "red", lwd = 2)

# For regression residuals
model <- lm(Y ~ X, data = df)
qqnorm(resid(model))
qqline(resid(model), col = "red")

# Or use built-in plot for lm objects
plot(model, 2)  # Standard plot #2 is Q-Q
```

---

## Interpretation Guide

| Visual | Interpretation |
|--------|----------------|
| Points on line | Data is approximately normal. |
| Points curve up at right | Right skew; consider log transform. |
| Points flatten at tails | Heavy tails (platykurtic/leptokurtic). |
| Single point far off | Outlier. Investigate. |

---

## Related Concepts

- [[30_Knowledge/Stats/02_Statistical_Inference/Shapiro-Wilk Test\|Shapiro-Wilk Test]] - Statistical test for normality.
- [[30_Knowledge/Stats/01_Foundations/Normal Distribution\|Normal Distribution]] - The reference distribution.
- [[30_Knowledge/Stats/01_Foundations/Log Transformation\|Log Transformation]] - Fix for skewness.

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Assumptions are violated
> - Alternative methods are more appropriate

---

## References

- **Article:** Wilk, M. B., & Gnanadesikan, R. (1968). Probability plotting methods for the analysis of data. *Biometrika*. [JSTOR](https://www.jstor.org/stable/2334444)
- **Book:** Thode, H. C. (2002). *Testing for Normality*. CRC Press. [Link](https://www.routledge.com/Testing-for-Normality/Thode/p/book/9780824796136)
- **Book:** Cleveland, W. S. (1993). *Visualizing Data*. Hobart Press. [Link](https://books.google.com.tr/books/about/Visualizing_Data.html?id=V-9SAAAAMAAJ)