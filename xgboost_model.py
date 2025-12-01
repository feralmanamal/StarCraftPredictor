import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv('starcraft.csv')
df = df.replace('?', np.nan)
df['TotalHours'] = pd.to_numeric(df['TotalHours'], errors='coerce')
df = df.dropna()

features = ['APM', 'SelectByHotkeys', 'ActionLatency', 'GapBetweenPACs', 'NumberOfPACs', 'TotalHours']

X = df[features]
y = df['LeagueIndex']

X['EfficiencyIndex'] = X['APM'] / X['ActionLatency']
X.drop(columns=['APM', 'ActionLatency'], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) 

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test)

xgb_model = xgb.XGBRegressor(objective ='reg:squarederror', early_stopping_rounds=500, eval_metric = 'mae', n_estimators=3000, random_state=42, learning_rate=0.001, max_depth=5)

xgb_model.fit(x_train_scaled, y_train, eval_set=[(x_test_scaled, y_test)], verbose=25)

xgb_predictions = xgb_model.predict(x_test_scaled)

xgb_mae = mean_absolute_error(y_test, xgb_predictions)

print(f"Random Forest MAE: 0.30685")
print(f"XGBoost Final MAE: {xgb_mae:.5f}")


