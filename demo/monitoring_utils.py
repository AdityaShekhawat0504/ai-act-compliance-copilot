# monitoring_utils.py
import os
import datetime
import pandas as pd

LOG_PATH = "monitoring_log.csv"


def log_audit_run(model_name, domain, accuracy, roc_auc, spd, risk_level="N/A"):
    """
    Append a single audit run to the monitoring_log.csv file.
    """
    ts = datetime.datetime.utcnow().isoformat(timespec="seconds")

    row = {
        "timestamp": ts,
        "model_name": model_name,
        "domain": domain,
        "accuracy": float(accuracy),
        "roc_auc": float(roc_auc),
        "spd": float(spd),
        "risk_level": risk_level,
    }

    if os.path.exists(LOG_PATH):
        df = pd.read_csv(LOG_PATH)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(LOG_PATH, index=False)


def load_monitoring_log():
    """
    Load the monitoring log as a DataFrame, or return None if it doesn't exist.
    """
    if not os.path.exists(LOG_PATH):
        return None
    return pd.read_csv(LOG_PATH)
