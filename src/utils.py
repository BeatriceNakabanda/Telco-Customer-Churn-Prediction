import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#Sort Features by Importance in ascending order top 10
def plot_feature_importance(model, X_train, top_n=10):
    importances = pd.Series(model.feature_importances_, index=X_train.columns) 
    top_features = importances.sort_values(ascending=False).head(top_n)
    top_features.sort_values().plot(kind='barh', figsize=(8,6))
    plt.title("Top Features Influencing Churn")
    plt.show()
    return top_features

def churn_risk_distribution(df, risk_col='Churn_Risk'):
    sns.histplot(df[risk_col], bins=30, kde=True)
    plt.title("Distribution of Customer Churn Risk")
    plt.xlabel("Churn Probability")
    plt.ylabel("Number of Customers")
    plt.show()