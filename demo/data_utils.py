# data_utils.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def generate_synthetic_credit(n=2000, random_state=42):
    rng = np.random.RandomState(random_state)
    age = rng.randint(18, 75, n)
    income = rng.normal(40000, 15000, n).astype(int)
    loan_amount = rng.normal(8000, 6000, n).clip(500, 50000).astype(int)
    tenure = rng.randint(6, 60, n)
    # Protected attribute (binary) for fairness demo
    gender = rng.choice([0, 1], n, p=[0.52, 0.48])  # 0=female, 1=male

    # Synthetic default rule + noise
    score = (income / 1000) - (loan_amount / 2000) + (age / 10) + rng.normal(0, 2, n)
    default = (score < 10).astype(int)

    df = pd.DataFrame({
        'age': age,
        'income': income,
        'loan_amount': loan_amount,
        'tenure': tenure,
        'gender': gender,
        'default': default
    })

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=['default']),
        df['default'],
        test_size=0.2,
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = generate_synthetic_credit(500)
    print(X_train.shape, X_test.shape)
