import shap 
import matplotlib.pyplot as plt

def shap_summary(model, X_train):

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    shap.summary_plot(shap_values, X_train, show=False)

    plt.savefig("outputs/figures/shap_summary.png")
    plt.close()


#SHAP is a model explainability tool that shows how each feature contributes to a machine learning prediction. 
# It helps us understand why the model predicts outcomes such as customer churn.