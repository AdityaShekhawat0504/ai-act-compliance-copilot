# model_training.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

def train_random_forest(X_train, y_train, n_estimators=200, random_state=42):
    """
    Trains a RandomForestClassifier on numeric features.
    Returns trained pipeline and the list of numeric columns used.
    """
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        ))
    ])

    pipeline.fit(X_train[numeric_cols], y_train)
    return pipeline, numeric_cols

def evaluate_model(model, numeric_cols, X_test, y_test):
    """
    Evaluates the model on test data and returns metrics, predictions, and probabilities.
    """
    probs = model.predict_proba(X_test[numeric_cols])[:, 1]
    preds = model.predict(X_test[numeric_cols])

    metrics = {
        'accuracy': float(accuracy_score(y_test, preds)),
        'roc_auc': float(roc_auc_score(y_test, probs)),
        'report': classification_report(y_test, preds, output_dict=True)
    }
    return metrics, preds, probs

def get_feature_importances(model, numeric_cols):
    """
    Extract feature importances from RandomForest inside pipeline.
    Returns a sorted list of (feature, importance).
    """
    clf = model.named_steps['clf']
    importances = clf.feature_importances_
    feat_imp = list(zip(numeric_cols, importances))
    feat_imp_sorted = sorted(feat_imp, key=lambda x: x[1], reverse=True)
    return feat_imp_sorted
