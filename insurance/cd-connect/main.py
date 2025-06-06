import pandas as pd
import numpy as np

df = pd.read_csv('insurance.csv')

print(df.head(2))

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

label_encoders = {}
for col in ['sex', 'smoker', 'region']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop('charges', axis=1)
y = df['charges']   

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

from sklearn.metrics import r2_score

lr_preds = lr_model.predict(X_test)
rf_preds = rf_model.predict(X_test)

print("Linear Regression Results")
print("R2 Score:", r2_score(y_test, lr_preds))

print("Random Forest Results")
print("R2 Score:", r2_score(y_test, rf_preds))