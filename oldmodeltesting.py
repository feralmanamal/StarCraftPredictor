import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
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

model = Sequential()
model.add(Dense(64, input_dim=x_scaled.shape[1], activation='relu'))
model.add(Dropout(0.2)) # randomly disables 20% of neurons to prevent overfitting
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2)) # randomly disables 20% of neurons to prevent overfitting
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(x_scaled, y, epochs=50, batch_size=10, validation_split=0.2)


sample_inputs = x_scaled[:10]
actual_ranks = y[:10]

# 2. Ask the model to predict
predictions = model.predict(sample_inputs)

# 3. Print them side by side
print("Actual  |  Predicted  |  Rounded")
print("---------------------------------")
for i in range(10):
    val = actual_ranks.iloc[i] # or actual_ranks[i] depending on format
    pred = predictions[i][0]
    print(f"   {val}    |    {pred:.2f}     |     {round(pred)}")