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

features = ['APM', 'SelectByHotkeys', 'ActionLatency', 'GapBetweenPACs', 'NumberOfPACs']

X = df[features]
y = df['LeagueIndex']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) 
# 80% train, 20% test

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.fit_transform(X_test)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(x_train_scaled, y_train)

predictions = rf_model.predict(x_train_scaled)

mae = mean_absolute_error(y_train, predictions)
print(f"Random Forest MAE: {mae}")

# begin putting validation things here...
feature_importances = rf_model.feature_importances_
print(feature_importances)

residuals = y_train - predictions

# --- 3. Create the Residual Plot ---
plt.figure(figsize=(10, 6))

# Plot the residuals (errors) against the predicted values
# A good model will have these points randomly scattered around the zero line.
plt.scatter(predictions, residuals, alpha=0.5)

# Add the zero reference line
plt.hlines(y=0, xmin=predictions.min(), xmax=predictions.max(), color='red', linestyle='--')

plt.title('Residual Analysis: Errors vs. Predicted League Index')
plt.xlabel('Predicted League Index (P)')
plt.ylabel('Residuals (Actual - Predicted)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()


userAPM = input("Enter your APM: ")
userSelectByHotkeys = input("Enter how often you select using hotkeys per timestamp: ")
userActionLatency = input("Enter your action latency (how long it takes you to react): ")
userGapBetweenPACs = input("Enter your gap between Perception Action cycles: ")
userNumberOfPACs = input("Enter your number of Perception Action cycles per timestamp: ")


test_df = pd.DataFrame({
    'APM': [userAPM], 
    'SelectByHotkeys': [userSelectByHotkeys], 
    'ActionLatency': [userActionLatency], 
    'GapBetweenPACs': [userGapBetweenPACs], 
    'NumberOfPACs': [userNumberOfPACs]  
}) 


new_player_scaled = scaler.transform(test_df)

prediction = rf_model.predict(new_player_scaled)
print(prediction)



#joblib.dump(rf_model, 'rf_model_ver1.joblib')
#joblib.dump(scaler, 'scaler_ver1.joblib')

#loaded_rf_model = joblib.load('rf_model_ver1.joblib')
#loaded_scaler = joblib.load('scaler_ver1.joblib')