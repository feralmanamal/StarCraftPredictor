import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import joblib

# imports the dataset and drops rows with missing values (loss of ~30 rows)
df = pd.read_csv('starcraft.csv')
df = df.replace('?', np.nan)
df = df.dropna()

features = ['APM', 'SelectByHotkeys', 'ActionLatency', 'GapBetweenPACs', 'NumberOfPACs', 'TotalHours']

X = df[features]
y = df['LeagueIndex']

X['EfficiencyIndex'] = X['APM'] / X['ActionLatency']
X.drop(columns=['APM', 'ActionLatency'], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) 
# 80% train, 20% test

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.fit_transform(X_test)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_features='sqrt')
rf_model.fit(x_train_scaled, y_train)

predictions = rf_model.predict(x_train_scaled)

mae = mean_absolute_error(y_train, predictions)
print(f"Random Forest MAE: {mae}")

# begin putting validation things here...
feature_importances = rf_model.feature_importances_
print(feature_importances)




#joblib.dump(rf_model, 'rf_model_ver1.joblib')
#joblib.dump(scaler, 'scaler_ver1.joblib')

#loaded_rf_model = joblib.load('rf_model_ver1.joblib')
#loaded_scaler = joblib.load('scaler_ver1.joblib')