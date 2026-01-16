import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Load Dataset
df = pd.read_csv("Chronic_Kidney_Dsease_data.csv")  # Updated file path

# Feature Selection (Dropping Non-Relevant Columns)
drop_columns = ["PatientID", "DoctorInCharge"]  # Remove non-essential columns
df = df.drop(columns=drop_columns)

# Define Features and Target
features = df.columns.tolist()
features.remove("Diagnosis")  # Target Variable
target = "Diagnosis"

X = df[features]
y = df[target]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train XGBoost Model
xgb_model = xgb.XGBClassifier(
    max_depth=5, learning_rate=0.05, n_estimators=200, subsample=0.8, colsample_bytree=0.8,
    importance_type='gain', random_state=42
)

xgb_model.fit(X_train, y_train)

# Predictions
y_pred = xgb_model.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Feature Importance Analysis
feature_importance = pd.DataFrame({'Feature': features, 'Importance': xgb_model.feature_importances_})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
print("\nFeature Importance Ranking:")
print(feature_importance)
