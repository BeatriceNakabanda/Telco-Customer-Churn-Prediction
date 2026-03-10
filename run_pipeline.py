import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.explainability import shap_summary
from src.data_preprocessing import (
    load_data, fill_missing, encode_binary, encode_multiclass,
    encode_target, drop_unnecessary, save_processed
)
from src.model_training import (
    split_scale, train_random_forest, evaluate, save_model, predict_churn_risk
)
from src.visualisation import (
    plot_confusion_matrix, plot_feature_importance, plot_churn_risk, plot_roc_curve
)
from sklearn.model_selection import train_test_split

import joblib

# -------------------------------
# Step 1: Load and preprocess data
# -------------------------------
data_path = "data/raw/telco.csv"
df = load_data(data_path)

# Fill missing values
df = fill_missing(df)

# Encode features
binary_cols = [
    'Phone Service', 'Paperless Billing', 'Referred a Friend', 'Multiple Lines',
    'Online Security', 'Online Backup', 'Device Protection Plan', 'Premium Tech Support',
    'Streaming TV', 'Streaming Movies', 'Streaming Music', 'Unlimited Data'
]
multi_cols = ['Gender', 'Internet Service', 'Internet Type', 'Contract', 'Payment Method']

df = encode_binary(df, binary_cols)
df = encode_multiclass(df, multi_cols)
df = encode_target(df)

# Drop unnecessary columns
df = drop_unnecessary(df)

# Save processed data
save_processed(df)

# -------------------------------
# Step 2: Define features & target
# -------------------------------
y = df['Churn']
X = df.drop(columns=['Churn'])

# Convert any remaining categorical columns
X = pd.get_dummies(X, drop_first=True)

# -------------------------------
# Step 3: Split and scale data
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save feature columns (needed for app)
feature_columns = list(X.columns)  # plain list
joblib.dump(feature_columns, "outputs/models/feature_columns.pkl")

# --- Save medians for numeric features ---
import numpy as np
numeric_cols = X.select_dtypes(include=np.number).columns
feature_medians = X[numeric_cols].median().to_dict()
joblib.dump(feature_medians, "outputs/models/feature_medians.pkl")

# -------------------------------
# Step 4: Train model
# -------------------------------
# model = train_random_forest(X_train, y_train)

# --- Train model ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Save model ---
joblib.dump(model, "outputs/models/rf_model.pkl")

# Save the feature columns used during training
feature_columns = list(X.columns)   # convert Index → list
joblib.dump(feature_columns, "outputs/models/feature_columns.pkl")

# -------------------------------
# Step 5: Evaluate model
# -------------------------------
cm, report = evaluate(model, X_test, y_test)
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

# Save model and scaler
save_model(model)

# Create dataframe with churn risk predictions
df_full = pd.concat([X, y], axis=1)

# Visualize results
plot_confusion_matrix(model, X_test, y_test)
plot_feature_importance(model, pd.DataFrame(X_train, columns=X.columns))
# Predict churn probability
df_full["Churn_Risk"] = model.predict_proba(pd.DataFrame(X, columns=X.columns))[:,1]

plot_churn_risk(df_full)

# -------------------------------
# Step 6: Predict churn risk for all customers
# -------------------------------
df_full = pd.read_csv("data/raw/telco.csv")  # Reload raw to keep IDs etc.
df_full_processed = encode_binary(df_full, binary_cols)
df_full_processed = encode_multiclass(df_full_processed, multi_cols)
df_full_processed = encode_target(df_full_processed)
df_full_processed = drop_unnecessary(df_full_processed)

# Align columns with training features
df_full_features = df_full_processed.reindex(columns=X.columns, fill_value=0)
df_full_with_risk = predict_churn_risk(model, df_full_features, df_full)

#Confusion Matrix
#[[1027    8]
# [  53  321]]
#Stayed → Correct = 1027
#Stayed → Predicted Churn 8
#Churn → Missed 53
#Churn → Correct 321

plot_roc_curve(model, X_test, y_test)

shap_summary(model, X_train)

df_full["Churn_Risk"] = model.predict_proba(X)[:,1]

high_risk = df_full[df_full["Churn_Risk"] > 0.7]

high_risk.to_csv("outputs/tables/high_risk_customers.csv", index=False)

