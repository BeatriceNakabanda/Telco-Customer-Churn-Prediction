# Telco Customer Churn Prediction App

This project predicts whether a telecom customer is likely to churn using a Random Forest Machine Learning model trained on a Telco customer dataset.

The project has two main components:

1. Model Analysis – Insights and predictions generated from the original dataset used to train and evaluate the model.
2. Interactive Prediction App – A Streamlit web application where users can input customer information and receive a real-time churn risk prediction.

---

# Live App

Try the interactive Streamlit application:

https://customer-churn-prediction-t.streamlit.app/

Users can enter customer attributes such as contract type, tenure, and monthly charges to receive a predicted churn probability.

---

# Dataset

The model is trained on the Telco Customer Churn dataset, which contains customer information such as:

- Demographics
- Account information
- Contract details
- Services subscribed
- Billing information
- Churn status

The dataset is used to train the machine learning model and evaluate its performance.

---

# Model Insights (Dataset-Based Analysis)

The following visualizations are generated from the training dataset to help understand churn behavior and model performance.

### Feature Importance

Shows which customer attributes have the largest influence on churn prediction.

![Feature Importance](figures/feature_importance.png)

---

### Churn Distribution

Displays the distribution of churn vs non-churn customers in the dataset.

![Churn Distribution](figures/churn_distribution.png)

---

### Confusion Matrix

Evaluates the model's classification performance on the test dataset.

![Confusion Matrix](figures/confusion_matrix.png)

---

# Features

- Predicts customer churn probability
- Interactive Streamlit web app
- Displays Feature Importance
- Shows Churn Distribution
- Provides Model Evaluation Metrics
- Built using a Random Forest Machine Learning model

---

# Machine Learning Workflow

1. Data preprocessing
2. Handling missing values
3. Feature encoding
4. Train/Test split
5. Model training using Random Forest Classifier
6. Model evaluation
7. Deployment using Streamlit

---

# Tech Stack

- Python
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- Git & GitHub

---

# Run Locally

Clone the repository

```bash
git clone https://github.com/YOURUSERNAME/telco-churn-prediction.git
cd telco-churn-prediction