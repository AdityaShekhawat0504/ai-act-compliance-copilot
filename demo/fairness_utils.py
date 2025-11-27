# fairness_utils.py
import numpy as np

def statistical_parity_difference(y_pred, protected_attr):
    """
    SPD = P(pred=1 | protected=1) - P(pred=1 | protected=0)
    y_pred: 1D array of predicted labels (0/1)
    protected_attr: aligned array (0/1)
    """
    y_pred = np.asarray(y_pred)
    prot = np.asarray(protected_attr)

    mask1 = prot == 1
    mask0 = prot == 0

    p1 = y_pred[mask1].mean() if mask1.sum() > 0 else 0.0
    p0 = y_pred[mask0].mean() if mask0.sum() > 0 else 0.0

    return float(p1 - p0)
