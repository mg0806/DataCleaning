# Supermarket Sales Data Analysis and Preprocessing

This repository contains a Jupyter Notebook (`210410107067-Manohar-Gupta.ipynb`) focused on cleaning, preprocessing, and analyzing supermarket sales data. The notebook performs various steps like handling missing values, normalizing data, feature scaling, dimensionality reduction using PCA, and feature selection to help improve further analysis or model-building efforts.

## Table of Contents
1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Data Preparation](#data-preparation)
4. [Data Cleaning and Imputation](#data-cleaning-and-imputation)
5. [Feature Scaling](#feature-scaling)
6. [Dimensionality Reduction using PCA](#dimensionality-reduction-using-pca)
7. [Feature Selection](#feature-selection)
8. [Summary of Insights](#summary-of-insights)
9. [Challenges](#challenges)

---

## Overview
The primary goal of this project is to analyze and preprocess a supermarket sales dataset to prepare it for machine learning or statistical modeling. The notebook automates the following tasks:
- **Handling missing data**.
- **Scaling numeric features**.
- **Applying Principal Component Analysis (PCA)** for dimensionality reduction.
- **Selecting important features** using statistical techniques like `SelectKBest`.

The notebook is structured in a way to allow easy understanding and modification based on the dataset you use.

---

## Requirements
To run this notebook, you will need the following Python libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `opencv-python`
  
Install the required libraries using the following:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn opencv-python
```

---

## Data Preparation
We start by uploading and loading the dataset. In this notebook, data is uploaded via Google Colab's file upload mechanism. It assumes the dataset is in Excel format, but you can modify it to handle other formats like CSV.

```python
# File upload and loading
uploaded = files.upload()
df = pd.read_excel(list(uploaded.keys())[0])  # Loading dataset
```

---

## Data Cleaning and Imputation
### Missing Values:
- The dataset may contain missing values, denoted by `'?'`. These are replaced with `NaN` values.
- Missing numeric values are imputed using the **median**, while missing categorical values are imputed using the **most frequent value**.

```python
from sklearn.impute import SimpleImputer

# Impute missing values
df.replace('?', np.nan, inplace=True)
imputer_median = SimpleImputer(strategy='median')
df[numeric_columns] = imputer_median.fit_transform(df[numeric_columns])

imputer_mode = SimpleImputer(strategy='most_frequent')
df[categorical_columns] = imputer_mode.fit_transform(df[categorical_columns])
```

---

## Feature Scaling
### Standardization:
Feature scaling is applied to numeric columns using **StandardScaler**. This step ensures that all features have a similar scale, which is crucial for many machine learning models and techniques like PCA.

```python
from sklearn.preprocessing import StandardScaler

# Feature scaling for numeric columns
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[numeric_columns])
```

---

## Dimensionality Reduction using PCA
We apply **Principal Component Analysis (PCA)** to reduce the dimensionality of the dataset while retaining as much variance as possible. PCA helps simplify the dataset and is particularly useful when dealing with a large number of features.

```python
from sklearn.decomposition import PCA

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
pca_data = pca.fit_transform(df_scaled)
```

We also visualize the explained variance of the first two components to evaluate how much information is retained.

---

## Feature Selection
Using `SelectKBest` and `f_regression`, we identify the most important features impacting the target variable (in this case, `Sales`). This step reduces noise in the dataset by focusing only on relevant features.

```python
from sklearn.feature_selection import SelectKBest, f_regression

# Selecting the most important features impacting 'Sales'
selector = SelectKBest(score_func=f_regression, k='all')
selector.fit(X, y)
```

---

## Summary of Insights
1. **Missing Data**: Imputation of missing values was necessary to ensure the dataset was complete. Numeric values were imputed using the median, while categorical values were imputed using the mode.
2. **Outlier Detection**: A boxplot visualization revealed outliers in some numeric columns, though no extreme skew was observed.
3. **Scaling**: Feature scaling was applied to ensure all numeric columns were on a similar scale, crucial for machine learning techniques.
4. **Dimensionality Reduction**: PCA showed that the first two components explained a significant portion of the variance in the data, which helps in dimensionality reduction.
5. **Feature Selection**: SelectKBest revealed the most important features that impact sales, providing insights into which variables are most influential.

---

## Challenges
1. **Missing Values**: Some columns contained significant missing values, requiring careful imputation strategies to avoid bias.
2. **PCA**: Choosing the right number of components for PCA was challenging. We used the explained variance to strike a balance between dimensionality reduction and information retention.
3. **Data Scaling**: Ensuring the alignment of data scaling with model requirements was crucial to avoid issues like data leakage.

---

## How to Use
1. **Run the notebook** on your local environment or in Google Colab.
2. **Upload your dataset** when prompted.
3. Follow the steps in the notebook to perform **data cleaning**, **feature scaling**, **PCA**, and **feature selection**.

---
