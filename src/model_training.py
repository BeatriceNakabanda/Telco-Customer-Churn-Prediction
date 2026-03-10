from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import pandas as pd
from src.visualisation import plot_confusion_matrix, plot_feature_importance, plot_churn_risk

def split_scale(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return cm, report

# def save_model(model, scaler, path_model="outputs/models/rf_model.pkl", path_scaler="outputs/models/scaler.pkl"):
#     joblib.dump(model, path_model)
#     joblib.dump(scaler, path_scaler)
def save_model(model, path_model="outputs/models/rf_model.pkl"):
    joblib.dump(model, path_model)

def predict_churn_risk(model, X_full, df):
    X_full = X_full.reindex(columns=X_full.columns, fill_value=0)
    churn_probabilities = model.predict_proba(X_full.values)[:, 1]
    df['Churn_Risk'] = churn_probabilities
    df.sort_values(by='Churn_Risk', ascending=False, inplace=True)
    df.head(10).to_csv("outputs/tables/high_risk_customers.csv", index=False)
    return df