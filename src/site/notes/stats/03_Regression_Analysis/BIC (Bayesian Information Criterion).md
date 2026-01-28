---
{"dg-publish":true,"permalink":"/stats/03-regression-analysis/bic-bayesian-information-criterion/","tags":["Regression","Model-Selection","Information-Theory"]}
---


## Definition

> [!abstract] Overview
> The **Bayesian Information Criterion (BIC)** is a criterion for model selection among a finite set of models; prediction errors are penalized more heavily for model complexity than in [[stats/03_Regression_Analysis/AIC (Akaike Information Criterion)\|AIC (Akaike Information Criterion)]].

$$
BIC = k \ln(n) - 2 \ln(\hat{L})
$$

Where:
*   $k$ = number of parameters
*   $n$ = number of data points
*   $\hat{L}$ = maximized value of likelihood function

## Related Concepts
*   [[stats/03_Regression_Analysis/AIC (Akaike Information Criterion)\|AIC (Akaike Information Criterion)]]
*   [[stats/01_Foundations/Likelihood Function\|Likelihood Function]]
