# Customer Churn Analysis

## Project Overview

Customer churn is a major business problem for subscription-based companies. Understanding which customers are most likely to leave can help businesses prioritize retention efforts and reduce customer loss.

This project analyzes customer churn using a telecommunications customer dataset containing **7,043 customers**.

The analysis combines exploratory data analysis, statistical comparisons, machine learning, model evaluation, and business interpretation to answer three main questions:

1. What characteristics are associated with customer churn?
2. Can customer churn be predicted using machine learning?
3. Which customers should a business prioritize for retention?

---

## Business Problem

The objective is to identify patterns associated with customer churn and develop predictive models that can help a business identify customers who may be at higher risk of leaving.

The project evaluates several machine learning approaches and compares them using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Because different business decisions have different costs, the project also examines the trade-off between identifying more potential churners and generating false alarms.

---

## Dataset

The dataset contains **7,043 customer records** and **21 original variables**.

The variables describe:

- Customer demographics
- Contract type
- Tenure
- Internet service
- Additional services
- Payment method
- Monthly charges
- Total charges
- Churn status

The target variable is:

**Churn**

It was converted into a binary target for machine learning:

- `0` = Retained
- `1` = Churned

---

## Data Cleaning

The initial audit identified:

- 7,043 rows
- 21 columns
- 0 duplicate rows
- 0 standard missing-value cells
- 7,043 unique customer IDs
- 11 invalid `TotalCharges` values

The 11 invalid `TotalCharges` values occurred for customers with zero months of tenure. These values were converted to missing values and subsequently handled during cleaning.

The cleaned dataset retained all **7,043 customers**.

Additional preparation included:

- Converting `TotalCharges` to numeric
- Encoding `Churn` as a binary variable
- Separating numerical and categorical features
- Preparing the data for machine learning
- Using a stratified train/test split

The final modeling dataset contained:

- **5,634 training customers**
- **1,409 testing customers**

The churn rate remained **26.54%** in both datasets.

---

## Exploratory Data Analysis

### Overall Churn

The dataset contains:

| Customer status | Customers | Percentage |
|---|---:|---:|
| Retained | 5,174 | 73.46% |
| Churned | 1,869 | 26.54% |

The overall customer churn rate is **26.54%**.

---

## Key Churn Findings

### Contract Type

| Contract | Churn Rate |
|---|---:|
| Month-to-month | **42.71%** |
| One year | 11.27% |
| Two year | **2.83%** |

Month-to-month customers have substantially higher churn than customers on longer contracts.

---

### Customer Tenure

| Tenure | Churn Rate |
|---|---:|
| 0–6 months | **52.94%** |
| 7–12 months | 35.89% |
| 13–24 months | 28.71% |
| 25–36 months | 21.63% |
| 37–60 months | 16.62% |
| 61–72 months | **6.61%** |

The highest-risk group is customers within their first six months.

Churn decreases substantially as customer tenure increases.

---

### Monthly Charges

| Monthly Charges | Churn Rate |
|---|---:|
| $0–$30 | **9.80%** |
| $30–$50 | 30.80% |
| $50–$70 | 20.76% |
| $70–$90 | **37.80%** |
| $90+ | 32.78% |

Customers paying between $70 and $90 per month have the highest churn rate among the charge groups analyzed.

---

### Internet Service

| Internet Service | Churn Rate |
|---|---:|
| Fiber optic | **41.89%** |
| DSL | 18.96% |
| No internet service | **7.40%** |

Fiber-optic customers show substantially higher churn than DSL customers.

This finding identifies an important area for further investigation rather than establishing that fiber optic service itself causes churn.

---

### Payment Method

| Payment Method | Churn Rate |
|---|---:|
| Electronic check | **45.29%** |
| Mailed check | 19.11% |
| Bank transfer (automatic) | 16.71% |
| Credit card (automatic) | **15.24%** |

Customers using electronic checks have the highest churn rate.

---

### Support and Security Services

Customers using additional support and security services generally show lower churn rates.

Examples include:

| Service | Without Service | With Service |
|---|---:|---:|
| Online Security | 41.77% | 14.61% |
| Tech Support | 41.64% | 15.17% |
| Online Backup | 39.93% | 21.53% |
| Device Protection | 39.13% | 22.50% |

These relationships are useful for identifying customer segments for further investigation, although they should not automatically be interpreted as causal effects.

---

# Machine Learning

Three primary classification approaches were evaluated:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting

Hyperparameter tuning was also performed for Random Forest and Gradient Boosting.

---

## Logistic Regression

The improved Logistic Regression model achieved:

| Metric | Result |
|---|---:|
| Accuracy | **80.70%** |
| Precision | **66.04%** |
| Recall | 56.15% |
| F1 Score | 0.6069 |
| ROC-AUC | 0.8422 |

Logistic Regression produced the highest overall accuracy among the tested models.

---

## Random Forest

The Random Forest model achieved:

| Metric | Result |
|---|---:|
| Accuracy | 75.87% |
| Precision | 53.07% |
| Recall | **78.61%** |
| F1 Score | **0.6336** |
| ROC-AUC | 0.8437 |

Random Forest produced substantially higher recall than Logistic Regression.

This means it identified a much larger proportion of customers who actually churned, although it also generated more false positives.

---

## Gradient Boosting

The original Gradient Boosting model achieved:

| Metric | Result |
|---|---:|
| Accuracy | 79.63% |
| Precision | 64.95% |
| Recall | 50.53% |
| F1 Score | 0.5684 |
| ROC-AUC | 0.8438 |

After hyperparameter tuning:

| Metric | Tuned Result |
|---|---:|
| Accuracy | 79.56% |
| Precision | 65.14% |
| Recall | 49.47% |
| F1 Score | 0.5623 |
| ROC-AUC | **0.8460** |

The tuned Gradient Boosting model achieved the highest ROC-AUC among the evaluated models.

---

## Model Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | **80.70%** | **66.04%** | 56.15% | 0.6069 | 0.8422 |
| Random Forest | 75.87% | 53.07% | **78.61%** | **0.6336** | 0.8437 |
| Gradient Boosting | 79.63% | 64.95% | 50.53% | 0.5684 | 0.8438 |
| Tuned Random Forest | 75.37% | 52.41% | 78.61% | 0.6289 | 0.8455 |
| Tuned Gradient Boosting | 79.56% | 65.14% | 49.47% | 0.5623 | **0.8460** |

---

# Error Analysis

The Logistic Regression model's test-set confusion matrix was:

| | Predicted Retained | Predicted Churned |
|---|---:|---:|
| **Actual Retained** | 927 | 108 |
| **Actual Churned** | 165 | 209 |

This means:

- **927** customers were correctly identified as retained.
- **209** customers who churned were correctly identified.
- **165** churners were missed.
- **108** retained customers were incorrectly flagged as potential churners.

This demonstrates why accuracy alone is not sufficient for evaluating a churn model.

---

# Prediction Threshold Analysis

The standard classification threshold is `0.50`.

At this threshold:

- Accuracy: **80.62%**
- Recall: **55.88%**

A lower threshold of `0.30` was also evaluated.

At `0.30`:

- Accuracy: **75.02%**
- Precision: **52.02%**
- Recall: **75.67%**
- F1 Score: **0.6166**

Lowering the threshold identifies substantially more potential churners, but also creates more false alarms.

Therefore, the appropriate threshold depends on the business objective.

If missing a customer at risk of churn is expensive, a lower threshold may be preferable.

If retention resources are limited and false alarms are costly, a higher threshold may be more appropriate.

---

# Key Business Insights

The analysis identified several important customer segments associated with higher churn:

1. **Month-to-month customers** have a 42.71% churn rate.
2. **Customers within their first six months** have a 52.94% churn rate.
3. Customers paying **$70–$90 per month** have a 37.80% churn rate.
4. **Fiber-optic customers** have a 41.89% churn rate.
5. Customers using **electronic checks** have a 45.29% churn rate.
6. Customers using additional security and support services generally show lower churn rates.

The strongest pattern is the relationship between churn and **early customer tenure**.

---

# Business Recommendations

Based on the analysis, a business could:

### 1. Prioritize early-tenure customers

Develop onboarding and engagement programs specifically for customers during their first six months.

### 2. Target month-to-month customers

Identify month-to-month customers with additional risk indicators and consider appropriate incentives for longer-term contracts.

### 3. Investigate high-charge customer segments

Review pricing, perceived value, service quality, and customer expectations among customers with higher monthly charges.

### 4. Investigate the fiber-optic customer experience

The high churn rate among fiber-optic customers warrants further investigation into service quality, pricing, technical issues, and customer expectations.

### 5. Investigate electronic-check customers

The high churn rate associated with electronic checks may indicate differences in customer behavior or payment experience that deserve further analysis.

### 6. Use predictive churn scores

Rather than treating every customer equally, retention teams can prioritize customers according to their predicted probability of churn.

### 7. Select the model according to the business objective

Logistic Regression provides the highest accuracy, while Random Forest provides substantially higher churn recall.

Therefore, the best model depends on the cost of missed churners versus unnecessary retention interventions.

---

# Final Conclusion

Customer churn in this dataset is strongly associated with contract type, tenure, monthly charges, internet service, payment method, and the use of support and security services.

The modeling results show that there is no single metric that completely defines the best churn model.

**Logistic Regression achieved the highest accuracy at 80.70%.**

**Random Forest achieved the highest churn recall at 78.61% and the highest F1 score among the primary models.**

**Tuned Gradient Boosting achieved the highest ROC-AUC at 0.8460.**

These differences demonstrate why model selection should be driven by the business objective rather than accuracy alone.

---

# Project Structure

```text
customer-churn-analysis/
│
├── Data/
│   └── Telco-Customer-Churn.csv
│
├── notebooks/
│   └── customer_churn_analysis.ipynb
│
├── outputs/
│   ├── figures/
│   └── results/
│
├── src/
│   └── data_audit.py
│
├── README.md
└── .gitignore
```

---

# Tools and Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- VS Code
- Git / GitHub

---

# Project Outcome

This project demonstrates an end-to-end customer churn analysis workflow:

**Data Audit → Data Cleaning → Exploratory Analysis → Feature Preparation → Model Development → Model Comparison → Hyperparameter Tuning → Error Analysis → Business Recommendations**

The final analysis provides both predictive modeling results and actionable business insights that can be used to prioritize customer retention efforts.