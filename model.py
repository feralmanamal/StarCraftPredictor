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

rankLabels = ["Bronze", "Silver", "Gold", "Platinum", "Diamond", "Master", "Grandmaster"]

features = ['APM', 'SelectByHotkeys', 'ActionLatency', 'GapBetweenPACs', 'NumberOfPACs']

X = df[features]
y = df['LeagueIndex']

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.10, random_state=42) 
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.20, random_state=42)
# ultimately 72% train, 18% val, 10% test



scaler = StandardScaler()
x_scaled = scaler.fit_transform(X)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(x_scaled, y)

predictions = rf_model.predict(x_scaled)

mae = mean_absolute_error(y, predictions)
print(f"Random Forest MAE: {mae}")


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