import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# imports the dataset and drops rows with missing values (loss of ~30 rows)
df = pd.read_csv('starcraft.csv')
df = df.replace('?', np.nan)
df = df.dropna()

features = ['APM', 'SelectByHotkeys', 'ActionLatency', 'GapBetweenPACs', 'HoursPerWeek', 'NumberOfPACs']

x = df[features]
y = df['LeagueIndex']

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

#model = Sequential()
#model.add(Dense(64, input_dim=x_scaled.shape[1], activation='relu'))
#model.add(Dropout(0.2)) # randomly disables 20% of neurons to prevent overfitting
#model.add(Dense(32, activation='relu'))
#model.add(Dropout(0.2)) # randomly disables 20% of neurons to prevent overfitting
#model.add(Dense(1, activation='linear'))
#model.compile(optimizer='adam', loss='mse', metrics=['mae'])
#model.fit(x_scaled, y, epochs=50, batch_size=10, validation_split=0.2)


# note: when interpreting results of the model, val_mae is the decimal percentage loss of how accurate in predicting the LeagueIndex



rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

rf_model.fit(x_scaled, y)

predictions = rf_model.predict(x_scaled)

mae = mean_absolute_error(y, predictions)
print(f"Random Forest MAE: {mae}")


userAPM = input("Enter your APM: ")
userSelectByHotkeys = input("Enter how often you select using hotkeys per timestamp: ")
userActionLatency = input("Enter your action latency (how long it takes you to react): ")
userGapBetweenPACs = input("Enter your gap between Perception Action cycles: ")
userHoursPerWeek = input("Enter how many hours a week you play: ")
userNumberOfPACs = input("Enter your number of Perception Action cycles per timestamp: ")


test_df = pd.DataFrame({
    'APM': [userAPM], 
    'SelectByHotkeys': [userSelectByHotkeys], 
    'ActionLatency': [userActionLatency], 
    'GapBetweenPACs': [userGapBetweenPACs], 
    'NumberOfPACs': [userNumberOfPACs]  # <--- CHECK YOUR CODE: You likely trained with a 6th feature.
})  # expected output is 5


new_player_scaled = scaler.transform(test_df)

prediction = rf_model.predict(new_player_scaled)
print(prediction)

