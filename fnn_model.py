import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
import matplotlib.pyplot as plt

# imports the dataset and drops rows with missing values (loss of ~30 rows)
df = pd.read_csv('starcraft.csv')
df = df.replace('?', np.nan)
df = df.dropna()

features = ['APM', 'SelectByHotkeys', 'ActionLatency', 'GapBetweenPACs', 'NumberOfPACs', 'TotalHours']

X = df[features]
y = df['LeagueIndex']

X['EfficiencyIndex'] = X['APM'] / X['ActionLatency']
X.drop(columns=['APM', 'ActionLatency'], inplace=True)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(X)

model = Sequential()
model.add(Dense(64, input_dim=x_scaled.shape[1], activation='relu'))
model.add(Dropout(0.2)) # randomly disables 20% of neurons to prevent overfitting
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = model.fit(x_scaled, y, epochs=50, batch_size=10, validation_split=0.2)

predictions = model.predict(x_scaled)

mae = mean_absolute_error(y, predictions)
print(f"FNN Model MAE: {mae}")