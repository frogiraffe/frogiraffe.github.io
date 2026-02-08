---
{"dg-publish":true,"permalink":"/30-knowledge/stats/04-supervised-learning/feature-engineering/","tags":["machine-learning","supervised"]}
---


## Definition

> [!abstract] Core Statement
> **Feature Engineering** is the process of using ==domain knowledge== to create, transform, and select features that improve machine learning model performance. It's often the difference between a mediocre and excellent model.

---

> [!tip] Intuition (ELI5): The Detective's Clues
> Raw data is like a crime scene. Feature engineering is the detective work — noticing that "time between events" matters more than raw timestamps, or that "ratio of income to debt" reveals more than either alone.

---

## Why It Matters

> [!important] "Applied machine learning is basically feature engineering."
> — Andrew Ng

| Impact | Example |
|--------|---------|
| **Linear model + good features** | Often beats deep learning with raw data |
| **Domain knowledge** | Creates features algorithms can't discover |
| **Interpretability** | Engineered features make sense to stakeholders |

---

## Feature Engineering Techniques

### 1. Numerical Transformations

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({
    'income': [30000, 50000, 75000, 120000],
    'age': [25, 35, 45, 55],
    'purchase_count': [0, 5, 15, 50]
})

# ========== LOG TRANSFORM (SKEWED DATA) ==========
df['log_income'] = np.log1p(df['income'])  # log(1+x) handles zeros

# ========== POLYNOMIAL FEATURES ==========
df['age_squared'] = df['age'] ** 2

# ========== BINNING ==========
df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 100], 
                         labels=['young', 'middle', 'senior'])

# ========== RATIOS & INTERACTIONS ==========
df['income_per_year'] = df['income'] / df['age']
df['income_x_purchases'] = df['income'] * df['purchase_count']
```

### 2. Categorical Encoding

```python
# ========== ONE-HOT ENCODING ==========
df_encoded = pd.get_dummies(df, columns=['category'], drop_first=True)

# ========== TARGET ENCODING (CAREFUL: LEAKAGE RISK) ==========
from category_encoders import TargetEncoder
encoder = TargetEncoder()
df['category_target'] = encoder.fit_transform(df['category'], y)

# ========== FREQUENCY ENCODING ==========
freq = df['category'].value_counts(normalize=True)
df['category_freq'] = df['category'].map(freq)

# ========== ORDINAL ENCODING ==========
order = {'low': 1, 'medium': 2, 'high': 3}
df['level_encoded'] = df['level'].map(order)
```

### 3. Datetime Features

```python
df['date'] = pd.to_datetime(df['date'])

# ========== EXTRACT COMPONENTS ==========
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek  # 0=Monday
df['hour'] = df['date'].dt.hour
df['is_weekend'] = df['date'].dt.dayofweek >= 5

# ========== CYCLICAL ENCODING (FOR SEASONALITY) ==========
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# ========== TIME SINCE EVENT ==========
df['days_since_signup'] = (pd.Timestamp.now() - df['signup_date']).dt.days
```

### 4. Text Features

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# ========== BASIC ==========
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()
df['has_exclamation'] = df['text'].str.contains('!').astype(int)

# ========== TF-IDF ==========
tfidf = TfidfVectorizer(max_features=100)
tfidf_features = tfidf.fit_transform(df['text'])
```

### 5. Aggregation Features

```python
# ========== GROUP STATISTICS ==========
# Customer-level features from transaction data
customer_stats = df.groupby('customer_id').agg({
    'amount': ['mean', 'std', 'min', 'max', 'sum', 'count'],
    'date': ['min', 'max']
}).reset_index()

customer_stats.columns = ['customer_id', 
                          'avg_amount', 'std_amount', 'min_amount', 
                          'max_amount', 'total_amount', 'transaction_count',
                          'first_date', 'last_date']

# ========== ROLLING FEATURES (TIME SERIES) ==========
df['rolling_mean_7d'] = df['value'].rolling(window=7).mean()
df['rolling_std_7d'] = df['value'].rolling(window=7).std()
df['lag_1'] = df['value'].shift(1)
df['pct_change'] = df['value'].pct_change()
```

---

## R Implementation

```r
library(dplyr)
library(recipes)

# ========== USING RECIPES ==========
recipe_obj <- recipe(target ~ ., data = train_data) %>%
  # Imputation
  step_impute_median(all_numeric()) %>%
  step_impute_mode(all_nominal()) %>%
  # Transformations
  step_log(income, offset = 1) %>%
  step_poly(age, degree = 2) %>%
  step_normalize(all_numeric_predictors()) %>%
  # Encoding
  step_dummy(all_nominal_predictors()) %>%
  # Feature selection
  step_nzv(all_predictors()) %>%  # Remove near-zero variance
  step_corr(all_numeric_predictors(), threshold = 0.9)

# Prep and bake
prepped <- prep(recipe_obj)
train_processed <- bake(prepped, new_data = train_data)
```

---

## Domain-Specific Features

| Domain | Feature Ideas |
|--------|---------------|
| **E-commerce** | Time since last purchase, avg basket size, return rate |
| **Finance** | Debt-to-income, payment-to-income, credit utilization |
| **Healthcare** | BMI, age×risk factors, days since last visit |
| **Marketing** | Recency, Frequency, Monetary (RFM) |
| **NLP** | Sentiment score, named entities, reading level |

---

## Feature Selection

After engineering, select the best features:

```python
# ========== CORRELATION-BASED ==========
corr_matrix = df.corr()
high_corr = np.where(np.abs(corr_matrix) > 0.9)

# ========== FEATURE IMPORTANCE ==========
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X, y)
importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

# ========== RECURSIVE FEATURE ELIMINATION ==========
from sklearn.feature_selection import RFE
selector = RFE(estimator=rf, n_features_to_select=20)
selector.fit(X, y)
selected_features = X.columns[selector.support_]
```

---

## Common Pitfalls

> [!warning] Real-World Traps
>
> **1. Target Leakage**
> - *Problem:* Using information that wouldn't be available at prediction time
> - *Example:* Using "time_to_conversion" to predict conversion
> - *Solution:* Only use features available BEFORE the prediction point
>
> **2. High Cardinality Categoricals**
> - *Problem:* One-hot encoding creates thousands of sparse columns
> - *Solution:* Use target encoding, embedding, or frequency encoding
>
> **3. Over-engineering**
> - *Problem:* Too many features → overfitting, slow training
> - *Solution:* Feature selection, regularization, cross-validation

---

## Related Concepts

- [[30_Knowledge/Stats/01_Foundations/Feature Scaling\|Feature Scaling]] — Normalize feature ranges
- [[30_Knowledge/Stats/04_Supervised_Learning/Imbalanced Data\|Imbalanced Data]] — May need specialized features
- [[30_Knowledge/Stats/04_Supervised_Learning/Hyperparameter Tuning\|Hyperparameter Tuning]] — After feature engineering
- [[30_Knowledge/Stats/04_Supervised_Learning/SHAP Values\|SHAP Values]] — Understand feature importance

---

## When to Use

> [!success] Use Feature Engineering When...
> - Refer to standard documentation
> - Refer to standard documentation

---

## When NOT to Use

> [!danger] Do NOT Use When...
> - Dataset is too small for training
> - Interpretability is more important than accuracy

---

## Python Implementation

```python
import numpy as np
import pandas as pd

# Example implementation of Feature Engineering
# See documentation for details

data = np.random.randn(100)
print(f"Mean: {np.mean(data):.3f}")
print(f"Std: {np.std(data):.3f}")
```

---

## References

- **Book:** Zheng, A., & Casari, A. (2018). *Feature Engineering for Machine Learning*. O'Reilly.
- **Course:** [Feature Engineering on Coursera](https://www.coursera.org/learn/feature-engineering)
- **Kaggle:** [Feature Engineering Techniques](https://www.kaggle.com/learn/feature-engineering)
