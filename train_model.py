import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# Sample fake data (replace with real dataset if you have it locally)
df = pd.DataFrame({
    'AMT_INCOME_TOTAL': [50000, 60000, 35000, 80000],
    'AMT_CREDIT_SUM': [10000, 20000, 5000, 12000],
    'DAYS_CREDIT': [-365, -730, -1095, -1460],
    'TARGET': [0, 0, 1, 0]
})

X = df.drop('TARGET', axis=1)
y = df['TARGET']

# Create preprocessing + model pipeline
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

# Train
pipe.fit(X, y)

# Save model with column names
joblib.dump((pipe, list(X.columns)), 'credit_model.joblib')

print("✅ Model saved as credit_model.joblib")
