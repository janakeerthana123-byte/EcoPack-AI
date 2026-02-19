import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor

# Load dataset
df = pd.read_csv("ml_training_dataset.csv")

# -----------------------------
# Define Features and Targets
# -----------------------------
X = df.drop(columns=["cost_estimated", "co2_estimated"])
y_cost = df["cost_estimated"]
y_co2 = df["co2_estimated"]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_cost_train, y_cost_test = train_test_split(
    X, y_cost, test_size=0.2, random_state=42
)

_, _, y_co2_train, y_co2_test = train_test_split(
    X, y_co2, test_size=0.2, random_state=42
)

# -----------------------------
# Scaling
# -----------------------------
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# -----------------------------
# 1️⃣ Train Cost Model
# -----------------------------
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    n_jobs=-1,
    random_state=42
)

rf_model.fit(X_train_scaled, y_cost_train)

# Predictions
y_cost_pred = rf_model.predict(X_test_scaled)

# Evaluation
print("----- COST MODEL -----")
print("RMSE:", np.sqrt(mean_squared_error(y_cost_test, y_cost_pred)))
print("MAE:", mean_absolute_error(y_cost_test, y_cost_pred))
print("R2:", r2_score(y_cost_test, y_cost_pred))

# Save model
joblib.dump(rf_model, "cost_model.pkl")

# -----------------------------
# 2️⃣ Train CO2 Model
# -----------------------------
xgb_model = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    n_jobs=-1,
    random_state=42
)

xgb_model.fit(X_train_scaled, y_co2_train)

# Predictions
y_co2_pred = xgb_model.predict(X_test_scaled)

# Evaluation
print("----- CO2 MODEL -----")
print("RMSE:", np.sqrt(mean_squared_error(y_co2_test, y_co2_pred)))
print("MAE:", mean_absolute_error(y_co2_test, y_co2_pred))
print("R2:", r2_score(y_co2_test, y_co2_pred))

# Save model
joblib.dump(xgb_model, "co2_model.pkl")

print("Models trained and saved successfully.")