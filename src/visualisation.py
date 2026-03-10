import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc

def plot_confusion_matrix(model, X_test, y_test, path="outputs/figures/confusion_matrix.png"):
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def plot_feature_importance(model, X_train, top_n=10, path="outputs/figures/feature_importance.png"):
    importance = pd.Series(model.feature_importances_, index=X_train.columns)
    top_features = importance.sort_values(ascending=False).head(top_n)
    top_features.sort_values().plot(kind='barh')
    plt.title("Top Features Influencing Churn")
    plt.xlabel("Importance Score")
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    top_features.to_csv("outputs/tables/top_features.csv")

def plot_churn_risk(df, path="outputs/figures/churn_risk_distribution.png"):
    sns.histplot(df['Churn_Risk'], bins=30)
    plt.title("Distribution of Customer Churn Risk")
    plt.xlabel("Churn Probability")
    plt.ylabel("Number of Customers")
    plt.savefig(path, bbox_inches='tight')
    plt.close()


#Show ROC Curve to see how well your model separates churn vs non-churn customers.
def plot_roc_curve(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:,1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("outputs/figures/roc_curve.png")
    plt.close()