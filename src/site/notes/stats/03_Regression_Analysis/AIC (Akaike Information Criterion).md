---
{"dg-publish":true,"permalink":"/stats/03-regression-analysis/aic-akaike-information-criterion/","tags":["probability","model-selection","information-theory"]}
---


## Definition

> [!abstract] Overview
> The **Akaike Information Criterion (AIC)** is an estimator of prediction error and thereby relative quality of statistical models for a given set of data. Given a collection of models for the data, AIC estimates the quality of each model, relative to each of the other models.

$$
AIC = 2k - 2\ln(\hat{L})
$$

Where:
*   $k$ is the number of estimated parameters in the model.
*   $\hat{L}$ is the maximum value of the likelihood function for the model.

## Purpose
*   **Model Selection:** Helps choose the best model by balancing **goodness of fit** (likelihood) against **model complexity** (number of parameters).
*   **Penalty for Complexity:** It penalizes overfitting by adding $2k$, discouraging models with too many parameters that don't significantly improve the fit.

## Interpretation
*   **Lower is Better:** The model with the lowest AIC is preferred.
*   **Relative Metric:** The absolute value of AIC doesn't matter; only the differences between AIC values of candidate models matter.

## Comparison with BIC
*   [[stats/03_Regression_Analysis/BIC (Bayesian Information Criterion)\|BIC (Bayesian Information Criterion)]] imposes a stronger penalty for model complexity than AIC.
*   AIC tends to pick more complex models, while BIC tends to pick simpler models.

## Related Concepts
*   [[stats/01_Foundations/Likelihood Function\|Likelihood Function]]
*   [[stats/03_Regression_Analysis/BIC (Bayesian Information Criterion)\|BIC (Bayesian Information Criterion)]]
*   [[Stepwise Regression\|Stepwise Regression]]
