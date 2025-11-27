# explanations.py
import shap
import matplotlib.pyplot as plt
import pandas as pd

def explain_model_shap_tree(model, numeric_cols, X_sample, save_path="shap_summary.png"):
    """
    Uses SHAP TreeExplainer for tree-based models (RandomForest).
    model: sklearn Pipeline with scaler + RandomForest
    numeric_cols: list of numeric column names used
    X_sample: DataFrame (test set or a sample)
    save_path: where to save a PNG summary plot
    """
    clf = model.named_steps['clf']
    scaler = model.named_steps.get('scaler', None)

    if scaler is not None:
        X_trans = scaler.transform(X_sample[numeric_cols])
    else:
        X_trans = X_sample[numeric_cols].values

    explainer = shap.TreeExplainer(clf)
    shap_values_all = explainer.shap_values(X_trans)

    # TreeExplainer returns list [values_for_class0, values_for_class1] for classifiers
    if isinstance(shap_values_all, (list, tuple)):
        shap_values = shap_values_all[1]
    else:
        shap_values = shap_values_all

    X_trans_df = pd.DataFrame(X_trans, columns=numeric_cols)

    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_trans_df, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    return save_path, shap_values
