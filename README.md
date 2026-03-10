# Telco Customer Churn Prediction

## Project Overview
This project predicts customer churn for a telecom company using machine learning.  
Churn prediction helps companies identify customers likely to leave so they can take proactive retention measures.

## Objective
- Build a machine learning model to predict churn
- Identify high-risk customers
- Analyze features driving churn
- Provide actionable business insights

## Dataset
- Source: [Kaggle Telco Churn Dataset](https://www.kaggle.com/datasets/alfathterry/telco-customer-churn-11-1-3)
- Size: 7043 customers, 43 features
- Includes demographics, service usage, and customer satisfaction

## Methodology
1. Data Cleaning & Preprocessing
   - Handle missing values  
   - Encode categorical variables  

2. Exploratory Data Analysis (EDA)
   - Analyze distributions, correlations, and churn patterns  

3. Machine Learning
   - Model: Random Forest Classifier  
   - Evaluation: Accuracy, Precision, Recall, F1-score, ROC-AUC  

4. Feature Importance Analysis 
   - Identify top factors driving churn

5. Churn Risk Prediction
   - Generate churn probabilities for all customers  
   - Highlight high-risk customers for retention actions  

## Key Results
- Accuracy: 95%  
- ROC-AUC: 0.99  
- Top Features Influencing Churn: Contract type, Tenure, Monthly Charges, Satisfaction Score

## Visualizations
- Feature Importance
- Churn Probability Distribution
- High-Risk Customer Table

## Tools & Libraries
- Python: pandas, numpy, scikit-learn, seaborn, matplotlib
- Jupyter Notebook

## Next Steps / Business Use
- Integrate model into a dashboard for retention campaigns  
- Monitor churn trends over time  
- Use insights to improve customer satisfaction