import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

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

scaler = joblib.load('scaler_ver1.joblib')

x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test)

xgb_model = xgb.XGBRegressor(
    objective ='reg:squarederror', 
    early_stopping_rounds=500, 
    eval_metric = 'mae', 
    n_estimators=3000, 
    random_state=42, 
    learning_rate=0.001,
    n_jobs = 1 
    max_depth=5
)

xgb_model.fit(x_train_scaled, y_train, eval_set=[(x_test_scaled, y_test)], verbose=25)

xgb_predictions = xgb_model.predict(x_test_scaled)

xgb_mae = mean_absolute_error(y_test, xgb_predictions)

print(f"XGBoost Final MAE: {xgb_mae:.5f}")

# "saving" the model

with open('xgboost_model_ver1', 'wb') as f:
    pickle.dump(xgb_model, f)

joblib.dump(xgb_model, 'fnn_model_ver1.joblib')

print(f"Model trained with {xgb_model.best_ntree_limit} trees.")
xgb_model.save_model('starcraft_xgboost_model.json')
