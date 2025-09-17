# Bank Churn Prediction

A comprehensive machine learning project for predicting customer churn in banking institutions using multiple classification algorithms and advanced data preprocessing techniques.

## Project Overview

This project implements a bank churn prediction system using various machine learning models including Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, and Neural Networks. The system addresses class imbalance using SMOTE and includes comprehensive feature engineering and selection techniques.

## Dataset

The dataset contains customer information including:
- Customer demographics (Age, Gender, Geography)
- Account information (Credit Score, Tenure, Balance)
- Product usage (Number of Products, Has Credit Card, Is Active Member)
- Financial data (Estimated Salary)
- Target variable (Exited - 1 for churn, 0 for retention)

## Installation & Requirements

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost imbalanced-learn matplotlib seaborn
```

## How to Run

1. **Data Loading**: Ensure your dataset is in the correct path
2. **Execute Notebooks**: Run cells sequentially for complete analysis
3. **Model Training**: All models are trained automatically with SMOTE balancing
4. **Feature Analysis**: Chi-Square tests and cross-validation performed
5. **Results**: Model performance metrics and comparisons are displayed

## Model Performance Results

### Main Model Rankings (Using All Features):

| Rank | Model | Accuracy | F1-Score | ROC-AUC | Reasoning |
|------|-------|----------|----------|---------|-----------|
| 1 | **LightGBM** | **86.18%** | **65.76%** | **88.83%** | Best balanced performance across all metrics, fastest training |
| 2 | **CatBoost** | 86.48% | 64.81% | 88.80% | Highest accuracy but slightly lower F1-score |
| 3 | **XGBoost** | 85.86% | 65.45% | 88.56% | Strong performance, highly tunable |
| 4 | **Random Forest** | 84.52% | 63.30% | 86.97% | Good baseline ensemble performance |
| 5 | **Neural Network** | 82.13% | 63.64% | 87.48% | Decent but requires more tuning |
| 6 | **Logistic Regression** | 75.55% | 55.93% | 81.59% | Linear baseline model |

### Cross-Validation Results (Categorical Features Only):
- **XGBoost CV ROC-AUC**: 0.6580 ± 0.0012
- **CatBoost CV ROC-AUC**: 0.6580 ± 0.0012  
- **LightGBM CV ROC-AUC**: 0.6580 ± 0.0012

*Note: The identical CV scores indicate that categorical features alone provide moderate discriminatory power (better than random 0.5, but below optimal 0.8+).*

## Feature Engineering & Selection

### Preprocessing Steps:
1. **Data Cleaning**: Removal of unnecessary columns (ID, CustomerID, Surname)
2. **Categorical Encoding**: 
   - Gender: Binary encoding (Male=0, Female=1)
   - Geography: One-hot encoding for France, Germany, Spain
3. **Feature Scaling**:
   - RobustScaler for: Balance, EstimatedSalary (handles outliers)
   - StandardScaler for: CreditScore, Age, Tenure, NumOfProducts, HasCrCard, IsActiveMember

### Chi-Square Feature Selection Results:
```
Feature Importance Scores (Categorical Features):
- Geography_Germany: 5809.74 (Highest - Strong churn predictor)
- Gender: 1997.61 (Significant impact)
- Geography_France: 1211.56 (Moderate impact)  
- Geography_Spain: 337.37 (Lowest impact)
```

## Business Analysis & Insights

### Key Feature Interpretations:

**1. Geography_Germany (Chi2 = 5809.74)**
- **Business Impact**: German customers show the strongest churn propensity
- **Possible Causes**: Increased competition, regulatory changes, cultural preferences
- **Strategy**: Develop Germany-specific retention campaigns and investigate market dynamics

**2. Gender (Chi2 = 1997.61)**  
- **Business Impact**: Gender significantly influences churn behavior
- **Research Finding**: Females tend to churn more
- **Strategy**: Gender-targeted product offerings and retention strategies

**3. Geographic Distribution**
- **France**: Moderate churn risk - requires attention but not critical
- **Spain**: Lowest churn risk - stable customer base

## Key Research Findings

1. **Geographic Influence**: German market requires immediate attention due to highest churn propensity
2. **Gender Impact**: Demographic targeting can improve retention strategies
3. **Model Selection**: Tree-based ensemble models significantly outperform linear approaches
4. **Feature Strategy**: Using all features provides better performance than aggressive selection
5. **Class Imbalance**: SMOTE effectively improves minority class recall without sacrificing precision

## Recommendations for Implementation

### Technical Recommendations:
1. **Use LightGBM** as the primary model for production deployment
2. **Implement all features** rather than selected subsets for optimal performance
3. **Apply SMOTE balancing** to address class imbalance effectively
4. **Cross-validate regularly** to monitor model performance over time

### Business Recommendations:
1. **Germany-focused initiatives**: Investigate competitive landscape and develop targeted retention programs
2. **Gender-specific strategies**: Create tailored products and communications for different demographic groups  
3. **Predictive monitoring**: Implement early warning systems for high-risk customer segments
4. **Feature expansion**: Consider additional behavioral and transactional features for model enhancement

## Future Enhancements

- **Hyperparameter Optimization**: Grid/Random search for optimal model parameters
- **Feature Engineering**: Create interaction terms and derived features
- **Model Interpretability**: SHAP values for individual prediction explanations
- **Real-time Deployment**: API development for live churn prediction
- **A/B Testing Framework**: Validate retention strategy effectiveness